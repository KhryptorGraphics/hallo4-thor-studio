# Hallo4 Thor Studio

Local web studio for running Hallo4 on Jetson AGX Thor. Drive a **target person's
portrait** with a webcam + mic and have the portrait move how you move and speak
what you say — optionally **in the target's own voice** (zero-shot voice cloning).

> Hallo4 is a diffusion video model: it generates in chunks, not in real time.
> The **Live Studio** is a *record → generate → play* loop — you record a short
> take, the Thor box renders it, and finished chunks stream into the preview as
> they complete. It is near-real-time, not a live mirror.

## Setup

```bash
bash scripts/setup_thor_env.sh
bash scripts/download_models.sh
```

## Run

```bash
conda activate hallo4-thor
# HTTP (localhost only — camera/mic work only on localhost):
uvicorn studio.backend.hallo4_studio.app:app --host 127.0.0.1 --port 8000

cd studio/frontend && npm install && npm run build   # build the SPA the backend serves
```

### HTTPS (required to use a webcam/mic from another computer)

Browsers only expose `getUserMedia` (camera/mic) in a **secure context**. Over a
plain `http://<lan-ip>` origin the camera is blocked, so to capture from a laptop
browser on the LAN you must serve the studio over HTTPS (or use `localhost`).

```bash
bash scripts/make_studio_cert.sh                 # self-signed cert for this host's LAN IP
uvicorn studio.backend.hallo4_studio.app:app --host 0.0.0.0 --port 8443 \
    --ssl-keyfile studio_data/certs/studio.key --ssl-certfile studio_data/certs/studio.crt
```

Open `https://<lan-ip>:8443/` and accept the self-signed certificate once. The
backend serves the built frontend, so this single URL is the whole app.

For the Vite dev server instead of the built SPA: `cd studio/frontend && npm run dev`
(binds `0.0.0.0:5173`, proxies the API). The backend's CORS already allows
localhost and private-LAN origins; add others via `HALLO4_STUDIO_ALLOW_ORIGINS`.

## Live Studio tab

1. **Capture source** — choose where the driving webcam + mic are:
   - **This computer** — the browser captures from the devices on whatever machine
     you opened the studio on (your laptop). Enable the camera, pick devices,
     **Record take**, **Stop & upload**. (Needs the HTTPS/secure context above.)
   - **Thor box** — capture from a webcam/mic plugged into the GPU host. Refresh
     devices, **Start capture**, **Stop & use**. (Recorded server-side via ffmpeg
     `v4l2`+`alsa`.)
2. **Target person** — upload the reference **image** (the face to animate) and a
   short **voice sample** of the target. Tick **Clone target voice** and the
   **consent** box.
3. **Animate** — submits a job; finished chunks stream into the preview, then the
   final muxed video plays. The take's motion + your words drive the portrait;
   with voice cloning on, the audio is converted to the target's timbre first
   (kNN-VC) and that converted audio is both lip-synced and muxed into the output.

## Voice cloning

Zero-shot voice conversion uses **kNN-VC** (`vace/models/utils/voice_convert.py`),
pulled via `torch.hub` on first use (WavLM + HiFi-GAN, cached under
`~/.cache/torch/hub`). No new pip dependency and no `onnxruntime` — it will not
disturb the locally-built `onnxruntime-gpu`. A longer (30s+) clean reference clip
gives a better target timbre. If conversion fails the job falls back to the user's
own voice (logged). Override the device with `HALLO4_VC_DEVICE`.

## Performance: resident model

By default each job runs as a subprocess that reloads ~6 GB of weights — fine for
batch use, slow for the interactive Live loop. Set `HALLO4_INPROCESS_ENGINE=1`
when launching the backend to keep `WanVace` resident in the server process
(`vace.vace_wan_inference.build_pipeline` caches it), so only the first clip pays
the load cost:

```bash
HALLO4_INPROCESS_ENGINE=1 uvicorn studio.backend.hallo4_studio.app:app --host 0.0.0.0 --port 8443 \
    --ssl-keyfile studio_data/certs/studio.key --ssl-certfile studio_data/certs/studio.crt
```

Trade-offs: the model shares the web-server process (a model crash takes the
server with it), and an in-process job can't be cancelled mid-round (cancel is
honoured at round boundaries). Jobs are still serialized by the GPU lock.

## Auth

Set `HALLO4_STUDIO_TOKEN` (bearer) or `HALLO4_STUDIO_USER`/`HALLO4_STUDIO_PASSWORD`
(basic) before exposing the studio beyond localhost. Consent is required for any
voice-clone job.
