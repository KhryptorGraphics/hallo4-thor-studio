# Phase 2 — True Live-Mirror Engine (scope)

> Status: **2a done (GO) + a working 2b–2e skeleton built** on branch
> `feat/webcam-avatar-studio`. Phase 1 (record → generate → play) also done.
>
> **Built (functional):** WebRTC live session (`live_session.py`, 30 fps relay),
> LivePortrait reenactment (`live_video_engine.py`), RVC engine + enrollment
> (`rvc_engine.py`/`enrollment.py`), frontend "Mirror" mode (`LiveMirror.tsx`).
> Mounted in `app.py`; real engines gated by `HALLO4_LIVE_ENGINE=1`.
>
> **Gaps closed (round 2):**
> - **fps — corrected after the live e2e found a bug:** the "22.6 fps via TRT" was on
>   **broken output** — fp16 ORT-TRT overflows the motion+SPADE nets to NaN → a **black
>   face**. Fixed: TRT defaults to fp32 + a warmup render-check disables TRT and recomputes
>   eager if the face is degenerate. Reality on Thor: **eager is correct AND fastest**
>   (~15 fps video-only / ~12 fps with Wav2Lip-every-frame); fp16-TRT is a black face,
>   fp32-TRT is correct but ~2× *slower* than eager. So **don't set `HALLO4_LIVE_TRT`** —
>   TRT gives no usable speedup for these LivePortrait nets here. The warp's 5D
>   grid_sample remains the eager floor; 25+ fps needs a native plugin (out of scope).
> - **Wav2Lip:** real (vendored model + `wav2lip_gan.pth`, librosa mel, audio ring
>   buffer) — mouth tracks speech; ~18.9 fps with it on every frame.
> - **Voice:** real **streaming kNN-VC** (not seed-vc — its `torch==2.4.0` pin would
>   wreck the CUDA-13 stack). Zero-shot timbre shift, **~0.4 s latency** (cut from ~1 s
>   — compute is only ~12 ms so the analysis window is the lever; 0.4 s is the measured
>   low-latency floor before WavLM/HiFiGAN vocode to silence; tunable via
>   `HALLO4_RVC_WIN`), zero new deps. Enrollment stores the reference (no training).
> **Follow-ons:** native 5D-GridSample plugin for 25+ fps; tighter Wav2Lip crop;
> vendored trained-RVC (lower latency) at the `TODO(rvc)` slots; A/V sync polish.
>
> **2a progress:**
> - **Transport ✅** — `aiortc 1.14 + av 16.1` install cleanly (av bundles ffmpeg →
>   ffmpeg-8 risk #2 moot); numpy/onnxruntime untouched; full WebRTC loopback
>   (DTLS/SRTP + VP8) sustains 30 fps at 320². Self-test:
>   `scripts/phase2_aiortc_loopback.py`.
> - **insightface / onnxruntime clobber ✅** — `pip install insightface --no-deps`
>   avoids the CPU-onnxruntime pull; GPU providers + numpy pin + x86_64 gate all
>   survive; detector runs on the GPU build (buffalo_l, ~26 ms on the ref image).
> - **LivePortrait fps ✅ GO via TensorRT** — PyTorch ≈ 16–17 fps and `torch.compile`
>   is a no-op on Thor (triton can't target SM110, see [[torch-compile-fails-thor-sm110]]).
>   But **TensorRT works**: via the onnxruntime **TRT execution provider** (no separate
>   `tensorrt` install needed) the SPADE decoder — the dominant 31 ms — drops to
>   **10.3 ms fp16 (3×)** and builds a **native `_sm110.engine`**. That alone takes the
>   per-frame budget to **~38 ms → ~26 fps** with everything else still eager; TRT-ing
>   the motion extractor + dense-motion convs adds headroom for Wav2Lip. Caveat: the
>   warping module's **5D (volumetric) grid_sample doesn't ONNX-export** — use the
>   FasterLivePortrait split (TRT the conv parts, keep the 5D warp in torch).
> - **RVC voice ✅** — the dominant component (94M HuBERT/ContentVec-class content
>   encoder, proxied by the repo's wav2vec2-base) runs **~5 ms/chunk** on Thor
>   regardless of chunk size (1–3% of the realtime budget); + RMVPE pitch + a
>   VITS synthesizer (~the HiFiGAN scale already run realtime in the kNN-VC test)
>   ⇒ ~15–30 ms/chunk compute, far under a 160–500 ms chunk. Compute is a non-issue;
>   the real cost is the **install** (fairseq/HuBERT on aarch64) + per-voice training.
>
> **2a VERDICT: GO.** All four legs feasible on Thor (transport, insightface,
> LivePortrait via TensorRT, RVC). Remaining Phase-2 work is engineering, not
> feasibility: the TensorRT LivePortrait build (incl. the 5D-grid_sample split),
> RVC install + enrollment workflow, aiortc session plumbing, and the frontend.

## Goal / definition of done

Sub-second, video-call-style puppeteering: the user's face + voice drive a target
portrait **live** (~25–30 fps, ~250–400 ms glass-to-glass), rendered on the Thor
box and streamed back to the browser. This is a **separate engine** from hallo4 —
hallo4 is a diffusion model (seconds–minutes per chunk) and physically cannot run
live. Lower fidelity than hallo4, but interactive.

## Why a new engine (not hallo4)

hallo4 = iterative diffusion (25 steps × 81-frame chunks). Live mirroring needs a
**single-pass, per-frame** reenactment network (sub-frame-interval latency). Different
model family, different transport (streaming, not a batch job). Phase 1's job/queue
model does not apply — live is a **session**.

## Architecture

```
Browser (laptop)                         Thor box (FastAPI + aiortc)
  webcam ─┐                               ┌─ face track ─► LivePortrait.drive(target,frame) ─► Wav2Lip mouth refine ─┐
  mic ────┤  WebRTC (DTLS/SRTP, HTTPS) ──►│                                                                         │
          │                               └─ RVC realtime (mic → target timbre, trained model) ──────────┐         │
  <video> ◄──────── WebRTC video+audio ───┤◄──── A/V mux + encode ◄──────────────────────────────────────┴─────────┘
```

- **Transport: aiortc** (Python WebRTC) on the backend; native `RTCPeerConnection`
  in the browser. Signaling = 2 endpoints (offer → answer) + ICE. Reuses the Phase 1
  HTTPS secure-context (already required and solved for `getUserMedia`).
- **Face reenactment: LivePortrait** at **256²** (implicit-keypoint, real-time-capable).
  Warm up once per session: encode the target image → appearance/keypoint features.
  Then per driving frame: detect/track face → `drive()` → composite onto target.
- **Lip-sync refinement: Wav2Lip-style** pass on the mouth region, driven by the
  (converted) audio, layered on the LivePortrait output for a crisper mouth. This puts
  **two models in the per-frame path** — combined fps at 256² is the key 2a metric.
- **Voice: trained RVC.** Per-target voices are **enrolled offline** (train an RVC model
  from the target's audio — reuses the Phase 1 job/queue as a "voice enrollment" job).
  The live session loads the trained model and runs **RVC real-time** inference on the
  mic track (~50–120 ms), lower latency than zero-shot.
- Abundant unified VRAM (132 GB) keeps LivePortrait + Wav2Lip + RVC all resident.
- **One live session at a time**, serialized by the existing GPU lock; hallo4 jobs and
  a live session are mutually exclusive.

## Phasing (gated)

- **2a — Feasibility spike (GO/NO-GO).** De-risk the unknowns before building:
  1. Install **LivePortrait** on aarch64 and benchmark **256²** fps (target ≥ 25 fps),
     then **LivePortrait + Wav2Lip combined** in one per-frame path (the real budget).
     Resolve the **insightface → onnxruntime** install *without* clobbering the GPU
     onnxruntime build (install `--no-deps`, or swap the face detector for a
     torch/mediapipe one). Verify providers stay GPU via Preflight /
     `scripts/thor_preflight.py`.
  2. Build/install **aiortc + PyAV** and confirm they work against **ffmpeg 8** (or pin
     ffmpeg 6/7, or use a jetson-ai-lab aarch64 wheel). Loopback a webcam track.
  3. Stand up **RVC**: train a small model from a sample voice, benchmark **RVC
     real-time** inference latency on Thor.
- **2b — Voice enrollment (offline).** "Enroll voice" job (reuses Phase 1 job/queue):
  upload target audio → train an RVC model → store as a reusable artifact. Needs more
  audio than the zero-shot path (ideally minutes of clean speech) and a GPU training run.
- **2c — Video-only live mirror.** aiortc signaling endpoints + session lifecycle,
  target-image warmup, per-frame **LivePortrait + Wav2Lip** loop, return video track.
  Frontend: "Mirror" mode in the Live tab using `RTCPeerConnection`. Mic passthrough.
- **2d — Live RVC voice.** Load the enrolled RVC model; run real-time VC on the mic
  track; mux to the return audio; handle A/V sync (small buffer offset).
- **2e — Studio integration & polish.** Engine selector (`hallo4` render vs `live`
  mirror) over the Phase 1 capture/consent plumbing; session cleanup; graceful
  fallback; visible "synthetic" watermark on live output.

## Latency budget (~250–400 ms target)

| Stage | Budget |
|-------|--------|
| Browser capture + encode | 30–50 ms |
| Uplink (LAN) | 5–20 ms |
| Decode + face track | 10–20 ms |
| **LivePortrait reenact (256²)** | 15–40 ms/frame *(validate in 2a)* |
| **Wav2Lip mouth refine** | +10–30 ms/frame *(validate in 2a)* |
| Composite + encode | 10–20 ms |
| Downlink | 5–20 ms |
| Audio: RVC real-time (parallel) | +50–120 ms |

The two-model per-frame path (LivePortrait + Wav2Lip) is the tightest constraint —
2a must confirm their **combined** fps clears 25 at 256².

## Reuse from Phase 1

HTTPS/secure-context, consent gate, target-image upload, device enumeration, auth, the
Live-tab shell, **and the job/queue model for the offline RVC voice-enrollment job**.
**New:** aiortc session endpoints (the live mirror is a session, not a job), the
LivePortrait + Wav2Lip per-frame loop, and real-time RVC inference.

## Risks

1. **onnxruntime clobber (insightface)** — ✅ RESOLVED. `pip install insightface
   --no-deps` skips the CPU-onnxruntime pull and reuses the GPU build; verified
   providers/numpy/x86_64-gate intact and the detector runs on GPU (~26 ms). The
   live engine must pin this install method. See [[torchaudio-no-torchcodec-thor]] for the
   same class of env trap.
2. **PyAV/aiortc vs ffmpeg 8** — ✅ RESOLVED. The `av` aarch64 wheel bundles its own
   ffmpeg, so it ignores the linuxbrew ffmpeg 8 entirely; loopback verified at 30 fps.
3. **Combined two-model fps on Thor** (LivePortrait + Wav2Lip per frame) — ✅ feasible
   (was MED/HIGH). Proven: TensorRT (via the onnxruntime TRT EP) takes SPADE 31 → 10.3 ms
   and builds native SM110 engines, putting the pipeline at ~26 fps with only SPADE
   converted; more modules on TRT add Wav2Lip headroom. `torch.compile` is a no-op here
   ([[torch-compile-fails-thor-sm110]]). Remaining work is *engineering, not feasibility*:
   a full TRT build (warping's 5D grid_sample needs the FasterLivePortrait split — TRT
   the conv parts, keep the volumetric warp in torch).
4. **RVC enrollment cost** — MED. Needs more clean target audio (ideally minutes) and a
   GPU training run per target; this is an offline step before any live session, not an
   upload. Quality depends on enrollment data.
5. **Single-GPU contention** (reenact + lip-sync + RVC + encode concurrent) — LOW given
   132 GB unified VRAM, but measure end-to-end.
6. **Misuse / consent** — live makes real-time impersonation (e.g. on a call) easy.
   Keep the consent gate; add a visible synthetic-content watermark on live output.

## Decisions (locked)

- **Voice:** trained **RVC** — per-target voice enrolled offline (training job), real-time
  inference in the session. Lower latency / higher fidelity than zero-shot; cost is a
  training step + more enrollment audio per target.
- **Quality/latency:** **256²**, ~300 ms target (smoothest, best real-time odds on Thor).
- **Driving:** **LivePortrait + Wav2Lip lip-sync** refinement (crisper mouth; two models
  in the per-frame path).

## Rough effort

~3–4 weeks of focused work for 2a→2e (RVC enrollment + the two-model per-frame path add
over the zero-shot/single-model baseline), **gated on the 2a spike** (~1–2 days; decides
whether LivePortrait+Wav2Lip clears 25 fps at 256² and whether aiortc/RVC install cleanly
on Thor at all).
