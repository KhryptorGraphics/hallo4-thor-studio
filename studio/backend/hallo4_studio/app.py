from __future__ import annotations

import asyncio
import base64
import glob
import json
import os
import platform
import re
import secrets
import shutil
import signal
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator


REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = Path(os.getenv("HALLO4_STUDIO_DATA", REPO_ROOT / "studio_data")).resolve()
UPLOADS_DIR = DATA_ROOT / "uploads"
JOBS_DIR = DATA_ROOT / "jobs"
UPLOAD_INDEX = DATA_ROOT / "uploads.json"

DEFAULT_CKPT_DIR = "pretrained_models/Wan2.1_Encoders"
DEFAULT_MODEL_PATH = "pretrained_models/hallo4/model_weight.ckpt"
DEFAULT_AUDIO_SEPARATOR = "pretrained_models/audio_separator/Kim_Vocal_2.onnx"
DEFAULT_WAV2VEC = "pretrained_models/wav2vec2-base-960h"
TERMINAL_STATUSES = {"succeeded", "failed", "cancelled"}


class UploadInfo(BaseModel):
    id: str
    filename: str
    content_type: str | None = None
    kind: str = "asset"
    path: str
    size: int
    url: str
    created_at: str


class ArtifactInfo(BaseModel):
    name: str
    size: int
    kind: str
    url: str
    created_at: str


class JobRequest(BaseModel):
    prompt: str | None = "a person is talking"
    source_video: str | None = None
    reference_images: list[str] = Field(default_factory=list)
    audio: str | None = None
    size: str = "480*832"
    frame_num: int = 81
    start_inf_frame: int = 0
    n_motion_frame: int = 1
    seed: int = 2025
    sample_solver: Literal["unipc", "dpm++"] = "unipc"
    sample_steps: int | None = None
    max_round: int | None = None
    sample_shift: float | None = None
    sample_guide_scale: float = 6.0
    offload_model: bool = False
    t5_cpu: bool = False
    batch_prompt_file: str | None = None
    model_name: str = "vace-1.3B"
    ckpt_dir: str = DEFAULT_CKPT_DIR
    model_path: str = DEFAULT_MODEL_PATH
    audio_separator_model_path: str = DEFAULT_AUDIO_SEPARATOR
    wav2vec_model_path: str = DEFAULT_WAV2VEC
    wav2vec_features: Literal["all", "last"] = "all"
    # Live studio additions
    target_voice: str | None = None          # upload id / path of the target speaker's reference clip
    voice_convert: bool = False              # clone the target's voice onto the driving speech
    voice_topk: int = 4                      # kNN-VC neighbours (higher = smoother timbre)
    stream_chunks: bool = False              # save per-round chunks for progressive playback
    consent: bool = False                    # user confirms consent to use this likeness/voice

    @field_validator("frame_num")
    @classmethod
    def validate_frame_num(cls, value: int) -> int:
        if value < 1:
            raise ValueError("frame_num must be positive")
        if (value - 1) % 4 != 0:
            raise ValueError("frame_num must be 4n+1")
        return value

    @field_validator("n_motion_frame")
    @classmethod
    def validate_motion_frames(cls, value: int) -> int:
        if value < 1:
            raise ValueError("n_motion_frame must be positive")
        return value


class JobInfo(BaseModel):
    id: str
    status: Literal["queued", "running", "succeeded", "failed", "cancelled"]
    created_at: str
    updated_at: str
    request: JobRequest
    command: list[str]
    artifacts: list[ArtifactInfo] = Field(default_factory=list)
    logs_tail: list[str] = Field(default_factory=list)
    error: str | None = None
    returncode: int | None = None
    progress: str | None = None


class RuntimeInfo(BaseModel):
    project_root: str
    data_root: str
    python: str
    platform: dict[str, str]
    cuda: dict[str, Any]
    torch: dict[str, Any]
    flash_attention: dict[str, Any]
    models: list[dict[str, Any]]
    packages: dict[str, Any]
    auth_enabled: bool
    active_job: str | None


class JobState:
    def __init__(self, info: JobInfo) -> None:
        self.info = info
        self.events: list[dict[str, Any]] = []
        self.process: asyncio.subprocess.Process | None = None
        self.cancel_requested = False

    def append_log(self, line: str) -> None:
        clean = line.rstrip()
        if not clean:
            return
        self.info.logs_tail.append(clean)
        self.info.logs_tail = self.info.logs_tail[-300:]
        self.info.updated_at = now_iso()
        round_match = re.search(r"\b(\d+/\d+)\b", clean)
        if round_match:
            self.info.progress = round_match.group(1)
        event = {"time": self.info.updated_at, "type": "log", "message": clean}
        self.events.append(event)
        write_job(self)

    def set_status(self, status: JobInfo.model_fields["status"].annotation, error: str | None = None) -> None:
        self.info.status = status
        self.info.error = error
        self.info.updated_at = now_iso()
        self.events.append({"time": self.info.updated_at, "type": "status", "status": status, "error": error})
        write_job(self)


class StudioState:
    def __init__(self) -> None:
        self.gpu_lock = asyncio.Lock()
        self.jobs: dict[str, JobState] = {}
        self.uploads: dict[str, UploadInfo] = {}
        self.active_job: str | None = None
        self.voice_converter: Any = None  # lazily-loaded VoiceConverter, kept resident across jobs
        self.capture: dict[str, Any] = {}  # active server-side capture process, if any


state = StudioState()
app = FastAPI(title="Hallo4 Thor Studio", version="0.1.0")
_extra_origins = [o.strip() for o in os.getenv("HALLO4_STUDIO_ALLOW_ORIGINS", "").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5173", "http://localhost:5173", *_extra_origins],
    # Allow the Vite dev server over http/https from localhost or a private LAN address,
    # so getUserMedia (which needs a secure context) works when the studio is opened on
    # another computer's browser over the network.
    allow_origin_regex=r"https?://(127\.0\.0\.1|localhost|192\.168\.\d+\.\d+|10\.\d+\.\d+\.\d+|172\.(1[6-9]|2\d|3[01])\.\d+\.\d+)(:\d+)?",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def auth_enabled() -> bool:
    return bool(os.getenv("HALLO4_STUDIO_TOKEN") or (os.getenv("HALLO4_STUDIO_USER") and os.getenv("HALLO4_STUDIO_PASSWORD")))


async def require_auth(request: Request) -> None:
    token = os.getenv("HALLO4_STUDIO_TOKEN")
    expected_user = os.getenv("HALLO4_STUDIO_USER")
    expected_password = os.getenv("HALLO4_STUDIO_PASSWORD")
    if not auth_enabled():
        return

    supplied_query_token = request.query_params.get("token")
    if token and supplied_query_token and secrets.compare_digest(supplied_query_token, token):
        return

    authorization = request.headers.get("authorization", "")
    if token and authorization.startswith("Bearer "):
        supplied = authorization.removeprefix("Bearer ").strip()
        if secrets.compare_digest(supplied, token):
            return

    if expected_user and expected_password and authorization.startswith("Basic "):
        try:
            decoded = base64.b64decode(authorization.removeprefix("Basic ").strip()).decode("utf-8")
            supplied_user, supplied_password = decoded.split(":", 1)
        except Exception:  # noqa: BLE001
            supplied_user, supplied_password = "", ""
        if secrets.compare_digest(supplied_user, expected_user) and secrets.compare_digest(supplied_password, expected_password):
            return

    raise HTTPException(status_code=401, detail="Authentication required", headers={"WWW-Authenticate": "Bearer"})


def ensure_dirs() -> None:
    for path in (DATA_ROOT, UPLOADS_DIR, JOBS_DIR):
        path.mkdir(parents=True, exist_ok=True)


def safe_filename(filename: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(filename).name).strip("._")
    return cleaned or "upload.bin"


def artifact_kind(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".mp4", ".mov", ".webm", ".mkv"}:
        return "video"
    if suffix in {".png", ".jpg", ".jpeg", ".webp"}:
        return "image"
    if suffix in {".wav", ".mp3", ".flac", ".m4a"}:
        return "audio"
    if suffix in {".json", ".txt", ".log"}:
        return "text"
    return "file"


def list_artifacts(job_id: str) -> list[ArtifactInfo]:
    artifact_dir = JOBS_DIR / job_id / "artifacts"
    if not artifact_dir.exists():
        return []
    artifacts: list[ArtifactInfo] = []
    for path in sorted(artifact_dir.iterdir()):
        if not path.is_file():
            continue
        stat = path.stat()
        artifacts.append(
            ArtifactInfo(
                name=path.name,
                size=stat.st_size,
                kind=artifact_kind(path),
                url=f"/api/artifacts/{job_id}/{path.name}",
                created_at=datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
            )
        )
    return artifacts


def write_uploads() -> None:
    ensure_dirs()
    UPLOAD_INDEX.write_text(json.dumps({k: v.model_dump() for k, v in state.uploads.items()}, indent=2), encoding="utf-8")


def write_job(job: JobState) -> None:
    job_dir = JOBS_DIR / job.info.id
    job_dir.mkdir(parents=True, exist_ok=True)
    job.info.artifacts = list_artifacts(job.info.id)
    payload = job.info.model_dump()
    payload["events"] = job.events[-1000:]
    (job_dir / "job.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_state() -> None:
    ensure_dirs()
    if UPLOAD_INDEX.exists():
        try:
            raw_uploads = json.loads(UPLOAD_INDEX.read_text(encoding="utf-8"))
            state.uploads = {key: UploadInfo(**value) for key, value in raw_uploads.items()}
        except Exception:
            state.uploads = {}

    for job_json in JOBS_DIR.glob("*/job.json"):
        try:
            payload = json.loads(job_json.read_text(encoding="utf-8"))
            events = payload.pop("events", [])
            info = JobInfo(**payload)
            if info.status in {"queued", "running"}:
                info.status = "failed"
                info.error = "Backend restarted before job completed."
            job = JobState(info)
            job.events = events
            job.info.artifacts = list_artifacts(info.id)
            state.jobs[info.id] = job
        except Exception:
            continue


def resolve_path(value: str | None, *, must_exist: bool = True) -> Path | None:
    if not value:
        return None
    if value in state.uploads:
        path = Path(state.uploads[value].path)
    else:
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = REPO_ROOT / path
    path = path.resolve()
    if must_exist and not path.exists():
        raise HTTPException(status_code=400, detail=f"Path does not exist: {value}")
    return path


def resolve_model_path(value: str) -> Path:
    path = resolve_path(value, must_exist=False)
    assert path is not None
    if path.exists():
        return path
    if path.name == "model_weight.ckpt":
        pth = path.with_suffix(".pth")
        if pth.exists():
            return pth
    return path


def validate_job_request(payload: JobRequest) -> dict[str, Path | list[Path] | None]:
    batch_file = resolve_path(payload.batch_prompt_file) if payload.batch_prompt_file else None
    if batch_file:
        return {
            "batch_prompt_file": batch_file,
            "source_video": None,
            "reference_images": [],
            "audio": None,
            "target_voice": None,
            "ckpt_dir": resolve_path(payload.ckpt_dir, must_exist=False),
            "model_path": resolve_model_path(payload.model_path),
            "audio_separator_model_path": resolve_path(payload.audio_separator_model_path, must_exist=False),
            "wav2vec_model_path": resolve_path(payload.wav2vec_model_path, must_exist=False),
        }

    source_video = resolve_path(payload.source_video)
    audio = resolve_path(payload.audio)
    reference_images = [resolve_path(item) for item in payload.reference_images]
    missing = []
    if source_video is None:
        missing.append("source_video")
    if audio is None:
        missing.append("audio")
    if not reference_images:
        missing.append("reference_images")
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing required generation inputs: {', '.join(missing)}")

    target_voice = None
    if payload.voice_convert:
        if not payload.consent:
            raise HTTPException(
                status_code=400,
                detail="Consent confirmation required to animate this likeness and clone this voice.",
            )
        if not payload.target_voice:
            raise HTTPException(status_code=400, detail="voice_convert requires a target_voice reference clip.")
        target_voice = resolve_path(payload.target_voice)

    return {
        "batch_prompt_file": None,
        "source_video": source_video,
        "reference_images": [p for p in reference_images if p is not None],
        "audio": audio,
        "target_voice": target_voice,
        "ckpt_dir": resolve_path(payload.ckpt_dir, must_exist=False),
        "model_path": resolve_model_path(payload.model_path),
        "audio_separator_model_path": resolve_path(payload.audio_separator_model_path, must_exist=False),
        "wav2vec_model_path": resolve_path(payload.wav2vec_model_path, must_exist=False),
    }


def append_arg(command: list[str], flag: str, value: Any | None) -> None:
    if value is None:
        return
    command.extend([flag, str(value)])


def build_command(job_id: str, payload: JobRequest, paths: dict[str, Any]) -> list[str]:
    save_dir = JOBS_DIR / job_id / "artifacts"
    save_dir.mkdir(parents=True, exist_ok=True)

    python_bin = os.getenv("HALLO4_PYTHON", sys.executable)
    command = [
        python_bin,
        "-m",
        "vace.vace_wan_inference",
        "--model_name",
        payload.model_name,
        "--size",
        payload.size,
        "--frame_num",
        str(payload.frame_num),
        "--start_inf_frame",
        str(payload.start_inf_frame),
        "--n_motion_frame",
        str(payload.n_motion_frame),
        "--base_seed",
        str(payload.seed),
        "--sample_solver",
        payload.sample_solver,
        "--sample_guide_scale",
        str(payload.sample_guide_scale),
        "--save_dir",
        str(save_dir),
        "--ckpt_dir",
        str(paths["ckpt_dir"]),
        "--model_path",
        str(paths["model_path"]),
        "--audio_separator_model_path",
        str(paths["audio_separator_model_path"]),
        "--wav2vec_model_path",
        str(paths["wav2vec_model_path"]),
        "--wav2vec_features",
        payload.wav2vec_features,
    ]
    append_arg(command, "--sample_steps", payload.sample_steps)
    append_arg(command, "--max_round", payload.max_round)
    append_arg(command, "--sample_shift", payload.sample_shift)

    if payload.offload_model:
        command.append("--offload_model")
    if payload.t5_cpu:
        command.append("--t5_cpu")
    if payload.stream_chunks:
        command.append("--save_chunks")

    if paths["batch_prompt_file"]:
        command.extend(["--prompt", str(paths["batch_prompt_file"])])
    else:
        command.extend(
            [
                "--prompt",
                payload.prompt or "a person is talking",
                "--src_video",
                str(paths["source_video"]),
                "--src_ref_images",
                ",".join(str(path) for path in paths["reference_images"]),
                "--src_audio",
                str(paths["audio"]),
            ]
        )
    return command


def to_wav16k(src: Path, dst: Path) -> Path:
    """Transcode any audio/video source to mono 16 kHz WAV (ffmpeg). Reused for
    the driving mic clip (often webm/opus) and the target voice reference."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", str(src), "-vn", "-ac", "1", "-ar", "16000", str(dst)],
        check=True,
        cwd=REPO_ROOT,
    )
    return dst


def get_voice_converter(topk: int):
    """Lazily build and cache the kNN-VC voice converter (kept resident across jobs)."""
    if state.voice_converter is None:
        from vace.models.utils.voice_convert import VoiceConverter

        device = os.getenv("HALLO4_VC_DEVICE", "cuda")
        state.voice_converter = VoiceConverter(device=device, topk=topk)
    state.voice_converter.topk = topk
    return state.voice_converter


def prepare_audio_sync(job: JobState) -> list[str]:
    """Blocking audio pre-step: transcode the driving clip to WAV and (optionally)
    clone the target's voice onto it. Returns a (possibly patched) command whose
    ``--src_audio`` points at the prepared WAV. On any failure, logs and returns
    the original command so the job still runs in the user's own voice."""
    req = job.info.request
    command = list(job.info.command)
    if "--src_audio" not in command:
        return command  # batch-prompt-file path has no inline audio
    idx = command.index("--src_audio") + 1
    audio_in = Path(command[idx])
    needs_prep = req.voice_convert or audio_in.suffix.lower() != ".wav"
    if not audio_in.exists() or not needs_prep:
        return command

    art = JOBS_DIR / job.info.id / "artifacts"
    try:
        driving = to_wav16k(audio_in, art / "driving.wav")
        job.append_log(f"Prepared driving audio -> {driving.name}")
        src_audio = driving
        if req.voice_convert and req.target_voice:
            target_src = resolve_path(req.target_voice)
            target_wav = to_wav16k(target_src, art / "target_voice.wav")
            job.append_log("Cloning target voice with kNN-VC (first run loads WavLM)...")
            converter = get_voice_converter(req.voice_topk)
            converted = converter.convert(str(driving), str(target_wav), str(art / "voice.wav"))
            src_audio = Path(converted)
            job.append_log("Voice conversion complete -> voice.wav")
        command[idx] = str(src_audio)
    except Exception as exc:  # noqa: BLE001
        job.append_log(f"Audio prep failed ({type(exc).__name__}: {exc}); using original audio.")
        return list(job.info.command)
    return command


async def run_job(job: JobState) -> None:
    if job.cancel_requested:
        job.set_status("cancelled")
        return

    async with state.gpu_lock:
        if job.cancel_requested:
            job.set_status("cancelled")
            return
        state.active_job = job.info.id
        job.set_status("running")
        env = os.environ.copy()
        env.setdefault("TORCH_CUDA_ARCH_LIST", "11.0")
        env.setdefault("PYTHONUNBUFFERED", "1")
        log_file = JOBS_DIR / job.info.id / "runner.log"
        try:
            # Pre-step: transcode driving audio and optionally clone the target voice.
            # Runs in a thread so the event loop (and SSE log stream) stays responsive.
            job.info.command = await asyncio.to_thread(prepare_audio_sync, job)
            if job.cancel_requested:
                job.set_status("cancelled")
                return
            process = await asyncio.create_subprocess_exec(
                *job.info.command,
                cwd=str(REPO_ROOT),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                env=env,
            )
            job.process = process
            with log_file.open("a", encoding="utf-8") as logs:
                assert process.stdout is not None
                while True:
                    line = await process.stdout.readline()
                    if not line:
                        break
                    text = line.decode("utf-8", errors="replace")
                    logs.write(text)
                    logs.flush()
                    job.append_log(text)
                    if job.cancel_requested and process.returncode is None:
                        process.terminate()
                returncode = await process.wait()
            job.info.returncode = returncode
            job.info.artifacts = list_artifacts(job.info.id)
            if job.cancel_requested:
                job.set_status("cancelled")
            elif returncode == 0:
                job.set_status("succeeded")
            else:
                job.set_status("failed", f"Inference exited with code {returncode}.")
        except Exception as exc:  # noqa: BLE001
            job.set_status("failed", f"{type(exc).__name__}: {exc}")
        finally:
            if state.active_job == job.info.id:
                state.active_job = None
            job.process = None
            job.info.artifacts = list_artifacts(job.info.id)
            write_job(job)


def shell_version(command: list[str]) -> str | None:
    try:
        return subprocess.check_output(command, cwd=REPO_ROOT, text=True, stderr=subprocess.STDOUT, timeout=8).strip()
    except Exception:
        return None


def model_checks() -> list[dict[str, Any]]:
    candidates = [
        ("Hallo4 checkpoint", [REPO_ROOT / "pretrained_models/hallo4/model_weight.ckpt", REPO_ROOT / "pretrained_models/hallo4/model_weight.pth"]),
        ("Wan VAE", [REPO_ROOT / "pretrained_models/Wan2.1_Encoders/Wan2.1_VAE.pth"]),
        ("Wan T5", [REPO_ROOT / "pretrained_models/Wan2.1_Encoders/models_t5_umt5-xxl-enc-bf16.pth"]),
        ("Audio separator", [REPO_ROOT / DEFAULT_AUDIO_SEPARATOR]),
        ("Wav2Vec", [REPO_ROOT / DEFAULT_WAV2VEC / "config.json", REPO_ROOT / "pretrained_models/wav2vec2-base-960h/config.json"]),
    ]
    checks = []
    for label, paths in candidates:
        found = next((path for path in paths if path.exists()), None)
        checks.append({"name": label, "ok": found is not None, "path": str(found.relative_to(REPO_ROOT)) if found else str(paths[0].relative_to(REPO_ROOT))})
    return checks


def package_checks() -> dict[str, Any]:
    offenders: list[str] = []
    for root in [Path(item) for item in sys.path if item.endswith("site-packages")]:
        if not str(root).startswith(sys.prefix):
            continue
        for wheel in root.glob("*.dist-info/WHEEL"):
            try:
                text = wheel.read_text(errors="ignore")
            except OSError:
                continue
            if "x86_64" in text or "linux_x86_64" in text:
                offenders.append(str(wheel))
    return {"no_x86_64_wheels": not offenders, "x86_64_wheels": offenders[:20]}


@app.on_event("startup")
async def startup() -> None:
    load_state()


@app.get("/api/health", dependencies=[Depends(require_auth)])
async def health() -> dict[str, Any]:
    return {"ok": True, "time": now_iso(), "active_job": state.active_job}


@app.get("/api/runtime", response_model=RuntimeInfo, dependencies=[Depends(require_auth)])
async def runtime() -> RuntimeInfo:
    torch_info: dict[str, Any] = {"available": False}
    flash_info: dict[str, Any] = {"available": False}
    cuda_info: dict[str, Any] = {
        "nvcc": shell_version(["nvcc", "--version"]),
        "nvidia_smi": shell_version(["nvidia-smi"]),
    }
    try:
        import torch

        torch_info = {
            "available": True,
            "version": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_version": torch.version.cuda,
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "device_capability": torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None,
        }
    except Exception as exc:  # noqa: BLE001
        torch_info = {"available": False, "error": f"{type(exc).__name__}: {exc}"}

    try:
        from vace.models.wan.modules import attention as wan_attention

        flash_info = {
            "available": bool(wan_attention.FLASH_ATTN_2_AVAILABLE or wan_attention.FLASH_ATTN_3_AVAILABLE),
            "flash_attn_2": bool(wan_attention.FLASH_ATTN_2_AVAILABLE),
            "flash_attn_3": bool(wan_attention.FLASH_ATTN_3_AVAILABLE),
        }
    except Exception as exc:  # noqa: BLE001
        flash_info = {"available": False, "error": f"{type(exc).__name__}: {exc}"}

    return RuntimeInfo(
        project_root=str(REPO_ROOT),
        data_root=str(DATA_ROOT),
        python=sys.executable,
        platform={"machine": platform.machine(), "system": platform.system(), "release": platform.release()},
        cuda=cuda_info,
        torch=torch_info,
        flash_attention=flash_info,
        models=model_checks(),
        packages=package_checks(),
        auth_enabled=auth_enabled(),
        active_job=state.active_job,
    )


@app.post("/api/uploads", response_model=UploadInfo, dependencies=[Depends(require_auth)])
async def upload_file(kind: str = Form("asset"), file: UploadFile = File(...)) -> UploadInfo:
    ensure_dirs()
    upload_id = uuid.uuid4().hex
    filename = safe_filename(file.filename or "upload.bin")
    destination = UPLOADS_DIR / f"{upload_id}_{filename}"
    size = 0
    with destination.open("wb") as out:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            out.write(chunk)

    info = UploadInfo(
        id=upload_id,
        filename=filename,
        content_type=file.content_type,
        kind=kind,
        path=str(destination),
        size=size,
        url=f"/api/uploads/{upload_id}",
        created_at=now_iso(),
    )
    state.uploads[upload_id] = info
    write_uploads()
    return info


@app.get("/api/uploads/{upload_id}", dependencies=[Depends(require_auth)])
async def get_upload(upload_id: str) -> FileResponse:
    info = state.uploads.get(upload_id)
    if not info:
        raise HTTPException(status_code=404, detail="Upload not found")
    path = Path(info.path)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Upload file missing")
    return FileResponse(path, media_type=info.content_type, filename=info.filename)


def list_video_devices() -> list[dict[str, str]]:
    devices: list[dict[str, str]] = []
    for path in sorted(glob.glob("/dev/video*")):
        name = path
        sys_name = Path(f"/sys/class/video4linux/{Path(path).name}/name")
        if sys_name.exists():
            try:
                name = sys_name.read_text().strip() or path
            except OSError:
                pass
        devices.append({"path": path, "name": name})
    return devices


def list_audio_devices() -> list[dict[str, str]]:
    try:
        text = subprocess.check_output(["arecord", "-l"], text=True, stderr=subprocess.DEVNULL, timeout=5)
    except Exception:  # noqa: BLE001
        return [{"id": "default", "name": "ALSA default"}]
    devices: list[dict[str, str]] = []
    for line in text.splitlines():
        match = re.match(r"card (\d+):\s*\w+\s*\[(.+?)\].*device (\d+):", line)
        if match:
            card, card_name, dev = match.group(1), match.group(2).strip(), match.group(3)
            devices.append({"id": f"hw:{card},{dev}", "name": f"{card_name} (hw:{card},{dev})"})
    return devices or [{"id": "default", "name": "ALSA default"}]


_VIDEO_DEVICE_RE = re.compile(r"^/dev/video\d+$")
_AUDIO_DEVICE_RE = re.compile(r"^(default|plug)?(hw:\d+,\d+|default)$")


class CaptureRequest(BaseModel):
    video_device: str = "/dev/video0"
    audio_device: str | None = "default"
    duration: int | None = None  # seconds; None records until /api/capture/stop
    fps: int = 25
    size: str = "1280x720"

    @field_validator("video_device")
    @classmethod
    def _validate_video(cls, value: str) -> str:
        if not _VIDEO_DEVICE_RE.match(value):
            raise ValueError("video_device must look like /dev/videoN")
        return value

    @field_validator("audio_device")
    @classmethod
    def _validate_audio(cls, value: str | None) -> str | None:
        if value and not _AUDIO_DEVICE_RE.match(value):
            raise ValueError("audio_device must be 'default' or 'hw:C,D'")
        return value

    @field_validator("size")
    @classmethod
    def _validate_size(cls, value: str) -> str:
        if not re.match(r"^\d{2,5}x\d{2,5}$", value):
            raise ValueError("size must be WIDTHxHEIGHT")
        return value


@app.get("/api/devices", dependencies=[Depends(require_auth)])
async def devices() -> dict[str, Any]:
    """Enumerate webcams + microphones attached to the Thor box (server-side capture)."""
    return {"video": list_video_devices(), "audio": list_audio_devices()}


@app.post("/api/capture/start", dependencies=[Depends(require_auth)])
async def capture_start(req: CaptureRequest) -> dict[str, Any]:
    """Begin recording from a webcam + mic on the Thor box into an upload."""
    if state.capture.get("process") is not None:
        raise HTTPException(status_code=409, detail="A capture is already running")
    if not Path(req.video_device).exists():
        raise HTTPException(status_code=400, detail=f"Video device not found: {req.video_device}")
    ensure_dirs()
    upload_id = uuid.uuid4().hex
    dest = UPLOADS_DIR / f"{upload_id}_capture.mp4"
    command = [
        "ffmpeg", "-y", "-v", "error",
        "-f", "v4l2", "-framerate", str(req.fps), "-video_size", req.size, "-i", req.video_device,
    ]
    if req.audio_device:
        command += ["-f", "alsa", "-i", req.audio_device]
    if req.duration:
        command += ["-t", str(int(req.duration))]
    command += ["-pix_fmt", "yuv420p", str(dest)]
    try:
        process = await asyncio.create_subprocess_exec(
            *command, stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE,
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Failed to start capture: {exc}") from exc
    state.capture = {"process": process, "dest": dest, "upload_id": upload_id, "started": now_iso()}
    return {"capturing": True, "upload_id": upload_id, "started": state.capture["started"]}


@app.get("/api/capture/status", dependencies=[Depends(require_auth)])
async def capture_status() -> dict[str, Any]:
    process = state.capture.get("process")
    running = process is not None and process.returncode is None
    return {"capturing": running, "started": state.capture.get("started")}


@app.post("/api/capture/stop", response_model=UploadInfo, dependencies=[Depends(require_auth)])
async def capture_stop() -> UploadInfo:
    """Stop the running capture, finalize the file, and register it as an upload."""
    capture = state.capture
    process = capture.get("process")
    if process is None:
        raise HTTPException(status_code=409, detail="No capture is running")
    try:
        process.send_signal(signal.SIGINT)  # let ffmpeg flush/finalize the mp4 moov atom
    except ProcessLookupError:
        pass
    try:
        await asyncio.wait_for(process.wait(), timeout=10)
    except asyncio.TimeoutError:
        process.kill()
        await process.wait()
    dest: Path = capture["dest"]
    upload_id: str = capture["upload_id"]
    state.capture = {}
    if not dest.exists() or dest.stat().st_size == 0:
        raise HTTPException(status_code=500, detail="Capture produced no output file")
    info = UploadInfo(
        id=upload_id, filename=dest.name, content_type="video/mp4", kind="capture",
        path=str(dest), size=dest.stat().st_size, url=f"/api/uploads/{upload_id}", created_at=now_iso(),
    )
    state.uploads[upload_id] = info
    write_uploads()
    return info


@app.post("/api/jobs", response_model=JobInfo, dependencies=[Depends(require_auth)])
async def create_job(payload: JobRequest) -> JobInfo:
    paths = validate_job_request(payload)
    job_id = uuid.uuid4().hex[:12]
    command = build_command(job_id, payload, paths)
    info = JobInfo(id=job_id, status="queued", created_at=now_iso(), updated_at=now_iso(), request=payload, command=command)
    job = JobState(info)
    state.jobs[job_id] = job
    write_job(job)
    asyncio.create_task(run_job(job))
    return job.info


@app.get("/api/jobs", response_model=list[JobInfo], dependencies=[Depends(require_auth)])
async def list_jobs() -> list[JobInfo]:
    jobs = sorted((job.info for job in state.jobs.values()), key=lambda item: item.created_at, reverse=True)
    for job in jobs:
        job.artifacts = list_artifacts(job.id)
    return jobs


@app.get("/api/jobs/{job_id}", response_model=JobInfo, dependencies=[Depends(require_auth)])
async def get_job(job_id: str) -> JobInfo:
    job = state.jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    job.info.artifacts = list_artifacts(job_id)
    return job.info


@app.post("/api/jobs/{job_id}/cancel", response_model=JobInfo, dependencies=[Depends(require_auth)])
async def cancel_job(job_id: str) -> JobInfo:
    job = state.jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.info.status in TERMINAL_STATUSES:
        return job.info
    job.cancel_requested = True
    if job.process and job.process.returncode is None:
        job.process.terminate()
    if job.info.status == "queued":
        job.set_status("cancelled")
    return job.info


@app.get("/api/jobs/{job_id}/events", dependencies=[Depends(require_auth)])
async def job_events(job_id: str) -> StreamingResponse:
    if job_id not in state.jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    async def stream() -> Any:
        cursor = 0
        while True:
            job = state.jobs[job_id]
            while cursor < len(job.events):
                event = job.events[cursor]
                cursor += 1
                yield f"event: {event.get('type', 'message')}\ndata: {json.dumps(event)}\n\n"
            if job.info.status in TERMINAL_STATUSES:
                yield f"event: done\ndata: {json.dumps(job.info.model_dump())}\n\n"
                break
            await asyncio.sleep(0.5)

    return StreamingResponse(stream(), media_type="text/event-stream")


@app.get("/api/artifacts/{job_id}/{name}", dependencies=[Depends(require_auth)])
async def get_artifact(job_id: str, name: str) -> FileResponse:
    if "/" in name or "\\" in name:
        raise HTTPException(status_code=400, detail="Invalid artifact name")
    path = JOBS_DIR / job_id / "artifacts" / name
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail="Artifact not found")
    return FileResponse(path, filename=name)


frontend_dist = REPO_ROOT / "studio/frontend/dist"
if frontend_dist.exists():
    app.mount("/", StaticFiles(directory=frontend_dist, html=True), name="frontend")
else:
    @app.get("/")
    async def root() -> JSONResponse:
        return JSONResponse({"ok": True, "message": "Hallo4 Studio API is running. Start the Vite frontend from studio/frontend."})
