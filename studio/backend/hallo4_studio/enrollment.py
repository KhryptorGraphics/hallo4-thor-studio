"""Voice-enrollment API for the Phase 2 live mirror (offline 'training' job).

Frontend contract (studio/frontend/src/LiveMirror.tsx):

    POST /api/voice/enroll        body {"audio": <upload id>, "name": <string>}
        -> {"id", "status": "queued"|"running", "progress"?, "error"?, "model_id"?}
    GET  /api/voice/enroll/{id}   -> same EnrollStatus shape; terminal statuses
        are "succeeded"/"failed"; "model_id" is set on success.
    GET  /api/voice/models        -> [{"id", "name"?, "status"?, "created_at"?}]

The job resolves the upload, transcodes it to 16k mono wav (ffmpeg, so any
webm/opus/mp3/m4a the browser hands us is readable), and runs
``RVCTrainer.enroll`` off the event loop, writing an artifact under
``DATA_ROOT/voice_models/<id>/``. The model-id, enrollment-job-id and dir name
are the same value end-to-end (the live session loads the model by that id).

Reuses Phase 1's job pattern: an in-memory registry persisted best-effort to
``voice_models/registry.json`` so GET works after a restart; it also self-heals
by re-registering any model dir found on disk.

No auth here by design — mount with the app's dependency::

    from hallo4_studio.enrollment import enroll_router
    app.include_router(enroll_router, dependencies=[Depends(require_auth)])

Smoke (app running on :8000)::

    UP=$(curl -sF kind=voice -F file=@assets/01.wav :8000/api/uploads | jq -r .id)
    ID=$(curl -s -XPOST :8000/api/voice/enroll -H 'content-type: application/json' \
            -d "{\"audio\":\"$UP\",\"name\":\"Demo\"}" | jq -r .id)
    curl -s :8000/api/voice/enroll/$ID    # poll until status=succeeded, model_id set
    curl -s :8000/api/voice/models        # [{"id","name","status","created_at"}]
"""
from __future__ import annotations

import asyncio
import json
import logging
import uuid
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from .app import DATA_ROOT, now_iso, resolve_path, to_wav16k
from .rvc_engine import RVCTrainer

logger = logging.getLogger(__name__)

enroll_router = APIRouter()

VOICE_MODELS_DIR = DATA_ROOT / "voice_models"
REGISTRY_PATH = VOICE_MODELS_DIR / "registry.json"

# In-memory enrollment-job registry; id == model-id == artifact dir name.
# ponytail: plain dict + a small json. Single host, jobs are rare; no DB needed.
_jobs: dict[str, dict[str, Any]] = {}
_loaded = False


class EnrollRequest(BaseModel):
    audio: str          # upload id (or path) of the target speaker's clip
    name: str = ""      # human label for the enrolled voice


def _ensure_loaded() -> None:
    """Load the persisted registry once, then reconcile with on-disk models."""
    global _loaded
    if _loaded:
        return
    VOICE_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    if REGISTRY_PATH.is_file():
        try:
            for jid, entry in json.loads(REGISTRY_PATH.read_text(encoding="utf-8")).items():
                _jobs[jid] = entry
        except Exception:  # noqa: BLE001
            logger.warning("Could not read voice registry %s; starting empty.", REGISTRY_PATH)
    # Self-heal: register any model dir on disk missing from the registry (so a
    # lost registry still lists usable voices), and demote jobs a crash left mid-run.
    for model_json in VOICE_MODELS_DIR.glob("*/model.json"):
        mid = model_json.parent.name
        if mid not in _jobs:
            try:
                meta = json.loads(model_json.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                meta = {}
            _jobs[mid] = {
                "id": mid, "name": meta.get("name"), "status": "succeeded",
                "created_at": meta.get("created_at") or now_iso(),
                "model_id": mid, "progress": None, "error": None,
            }
    for entry in _jobs.values():
        if entry.get("status") in ("queued", "running"):
            entry["status"] = "failed"
            entry["error"] = "Backend restarted before enrollment completed."
    _loaded = True
    _save()


def _save() -> None:
    try:
        VOICE_MODELS_DIR.mkdir(parents=True, exist_ok=True)
        REGISTRY_PATH.write_text(json.dumps(_jobs, indent=2), encoding="utf-8")
    except Exception as exc:  # noqa: BLE001 — persistence is best-effort
        logger.warning("Could not persist voice registry: %s", exc)


def _status(entry: dict[str, Any]) -> dict[str, Any]:
    """Project a registry entry to the frontend's EnrollStatus shape."""
    return {
        "id": entry["id"],
        "status": entry["status"],
        "progress": entry.get("progress"),
        "error": entry.get("error"),
        "model_id": entry.get("model_id"),
    }


async def _run_enroll(job_id: str, audio_path: str) -> None:
    entry = _jobs[job_id]
    entry["status"] = "running"
    entry["progress"] = "starting"
    _save()
    model_dir = VOICE_MODELS_DIR / job_id

    def on_log(msg: str) -> None:
        entry["progress"] = msg  # live in-memory progress; persisted at transitions

    def _train() -> str:
        model_dir.mkdir(parents=True, exist_ok=True)
        # Transcode any container to 16k mono wav so soundfile can read it.
        ref = to_wav16k(Path(audio_path), model_dir / "reference.wav")
        return RVCTrainer.enroll(str(ref), str(model_dir), on_log=on_log)

    try:
        await asyncio.to_thread(_train)
        entry.update(status="succeeded", model_id=job_id, progress="done", error=None)
        logger.info("Voice enrollment %s succeeded -> %s", job_id, model_dir)
    except Exception as exc:  # noqa: BLE001
        entry.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        logger.warning("Voice enrollment %s failed: %s", job_id, exc)
    _save()


@enroll_router.post("/api/voice/enroll")
async def enroll(req: EnrollRequest) -> dict[str, Any]:
    _ensure_loaded()
    src = resolve_path(req.audio)  # raises 400 if the upload id / path is missing
    if src is None:
        raise HTTPException(status_code=400, detail="audio is required")
    job_id = uuid.uuid4().hex[:12]
    _jobs[job_id] = {
        "id": job_id, "name": req.name.strip() or None, "status": "queued",
        "created_at": now_iso(), "model_id": None, "progress": None, "error": None,
    }
    _save()
    asyncio.create_task(_run_enroll(job_id, str(src)))
    return _status(_jobs[job_id])


@enroll_router.get("/api/voice/enroll/{job_id}")
async def enroll_status(job_id: str) -> dict[str, Any]:
    _ensure_loaded()
    entry = _jobs.get(job_id)
    if entry is None:
        raise HTTPException(status_code=404, detail="Enrollment job not found")
    return _status(entry)


@enroll_router.get("/api/voice/models")
async def list_models() -> list[dict[str, Any]]:
    _ensure_loaded()
    models = [
        {
            "id": entry["id"],
            "name": entry.get("name"),
            "status": entry.get("status"),
            "created_at": entry.get("created_at"),
        }
        for entry in _jobs.values()
        if entry.get("status") == "succeeded" and (VOICE_MODELS_DIR / entry["id"] / "model.json").is_file()
    ]
    models.sort(key=lambda m: m.get("created_at") or "", reverse=True)
    return models
