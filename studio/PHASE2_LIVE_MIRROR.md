# Phase 2 — True Live-Mirror Engine (scope)

> Status: **scoping**. Not built. Phase 1 (record → generate → play on hallo4 with
> voice cloning) is done on branch `feat/webcam-avatar-studio`.
>
> **2a progress (transport leg ✅):** `aiortc 1.14 + av 16.1` install cleanly on
> Thor — av ships a prebuilt aarch64 wheel with **bundled ffmpeg**, so the
> ffmpeg-8 risk (#2) is moot; numpy 1.26.4 and onnxruntime CUDA/TensorRT
> providers untouched; studio x86_64-wheel gate stays green. A full in-process
> WebRTC loopback (DTLS/SRTP + VP8 encode→decode) sustains 30 fps at 320².
> Self-test: `scripts/phase2_aiortc_loopback.py`. **Still open in 2a:**
> LivePortrait+insightface install & fps, and RVC.

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

1. **onnxruntime clobber (insightface)** — HIGH. Mitigate: `pip install insightface
   --no-deps` + keep onnxruntime-gpu pinned, or replace the detector. Gate on Preflight
   showing GPU providers + no x86_64 wheels. See [[torchaudio-no-torchcodec-thor]] for the
   same class of env trap.
2. **PyAV/aiortc vs ffmpeg 8** — ✅ RESOLVED. The `av` aarch64 wheel bundles its own
   ffmpeg, so it ignores the linuxbrew ffmpeg 8 entirely; loopback verified at 30 fps.
3. **Combined two-model fps on Thor** (LivePortrait + Wav2Lip per frame) — MED/HIGH.
   Blackwell + TensorRT EP should suffice but is unproven; 2a measures the *combined*
   path, not each model alone.
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
