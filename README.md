<h1 align="center">Hallo4 Thor Studio</h1>

<p align="center">
Drive a target portrait from your webcam and microphone — the image moves how you
move, speaks what you say, in a target voice — running on an <b>NVIDIA Jetson AGX
Thor</b>. A self-hosted studio built on the <a href="https://github.com/fudan-generative-vision/hallo4">Hallo4</a>
(Wan2.1) portrait-animation model, with an added real-time live-mirror engine.
</p>

> ⚠️ **Responsible use.** This is portrait-animation + voice-cloning technology
> (synthetic media / "deepfake"). Use it only on **your own likeness or with the
> explicit, informed consent** of the person whose image and voice you animate.
> The studio requires a per-session consent confirmation. Do not use it for
> impersonation, fraud, harassment, or any non-consensual content.

## Two modes

| Mode | What it does | How it runs |
|------|--------------|-------------|
| **Render** (offline) | Upload/record a driving clip + a target image + audio → a generated video of the target moving and lip-syncing, optionally **in a cloned voice**. | Hallo4 diffusion (Wan2.1-1.3B), chunked; **minutes per clip** on Thor. Finished chunks stream into the preview. |
| **Mirror** (real-time) | Sit at your webcam → a target portrait is driven **live** over WebRTC, with lip-sync and voice conversion. | **LivePortrait** reenactment + **Wav2Lip** mouth + streaming **kNN-VC** voice. ~**12–15 fps**, ~**0.4 s** voice latency. |

Both modes share one web app: capture from **your browser** (any computer on the
LAN) or from a **camera attached to the Thor box**; upload a target image and a
target voice sample; watch the result. Heavy compute stays on the GPU host.

### Honest limits (measured on Thor)
- **Mirror is not a perfect 25 fps mirror.** LivePortrait's warping net (a 5D
  `grid_sample`) is an eager ~22 ms wall; eager is correct and fastest here.
  `torch.compile` doesn't work on Thor's SM110, and `onnxruntime`-TensorRT **fp16
  overflows these nets to a black face** while fp32-TRT is slower than eager — so
  the engine runs **eager** by default (don't set `HALLO4_LIVE_TRT`). 25+ fps would
  need a native 5D-`grid_sample` CUDA plugin.
- **Voice** is zero-shot **kNN-VC** (no per-voice training); ~0.4 s window latency,
  tunable via `HALLO4_RVC_WIN`. A trained-RVC path (lower latency/higher fidelity)
  is a documented follow-on.

## Quickstart

```bash
# 1. Environment + models (Jetson AGX Thor, aarch64 / CUDA 13)
bash scripts/setup_thor_env.sh
bash scripts/download_models.sh
conda activate hallo4-thor

# 2. HTTPS cert (browser camera/mic need a secure context over the LAN)
bash scripts/make_studio_cert.sh

# 3. Build the frontend the backend serves
cd studio/frontend && npm install && npm run build && cd ../..

# 4. Run (real engines on; eager is the correct/fast path — no HALLO4_LIVE_TRT)
HALLO4_LIVE_ENGINE=1 \
  uvicorn studio.backend.hallo4_studio.app:app --host 0.0.0.0 --port 8443 \
  --ssl-keyfile studio_data/certs/studio.key --ssl-certfile studio_data/certs/studio.crt
```

Open `https://<thor-lan-ip>:8443/`, accept the self-signed cert, and use the
**Live Studio** tab.

**Deploy as a persistent service** (systemd `--user`, HTTPS, auto-generated auth
token, survives reboot — no sudo):

```bash
bash scripts/deploy_studio.sh
# manage: systemctl --user {status,restart,stop} hallo4-studio
```

Full setup, flags, and the live-mirror architecture:
- [`studio/README.md`](studio/README.md) — setup, HTTPS, capture flows, the
  resident-model option (`HALLO4_INPROCESS_ENGINE=1`).
- [`studio/PHASE2_LIVE_MIRROR.md`](studio/PHASE2_LIVE_MIRROR.md) — the live-mirror
  engine design, the Thor feasibility findings, and remaining follow-ons.

## Developers

- Khryptor@giggadev

## Acknowledgements & licenses

Built on excellent open research — please honor each project's license:

- **[Hallo4](https://github.com/fudan-generative-vision/hallo4)** (Fudan University et al.,
  SIGGRAPH Asia 2025, [arXiv:2505.23525](https://arxiv.org/pdf/2505.23525)) — the
  audio-driven portrait-animation model this studio wraps. Hallo4 is a derivative of
  **[Wan2.1-1.3B](https://github.com/Wan-Video/Wan2.1)**; use must comply with the
  [WAN LICENSE](https://github.com/Wan-Video/Wan2.1/blob/main/LICENSE.txt).
- **[LivePortrait](https://github.com/KwaiVGI/LivePortrait)** (Kuaishou) — real-time
  face reenactment in the Mirror engine.
- **[Wav2Lip](https://github.com/Rudrabha/Wav2Lip)** — lip-sync. The model code is
  vendored under `studio/backend/hallo4_studio/vendor/`; it is licensed for
  **research / non-commercial** use. Check its license before any commercial use.
- **[kNN-VC](https://github.com/bshall/knn-vc)** — zero-shot voice conversion
  (WavLM + HiFi-GAN).
- **[insightface](https://github.com/deepinsight/insightface)** — face detection;
  **[aiortc](https://github.com/aiortc/aiortc)** — WebRTC transport.

This repository is a self-hosted integration/UI on top of the above; it does not
relicense or redistribute their model weights (weights are downloaded at setup and
are git-ignored).
