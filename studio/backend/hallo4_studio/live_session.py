"""Phase 2 live-mirror transport/session layer (aiortc over WebRTC).

Browser sends webcam video + mic audio over a WebRTC offer; this module stands up
an aiortc ``RTCPeerConnection`` per session, transforms each inbound media frame
through the injected engines, and returns the result on outbound tracks:

    inbound video frame --bgr24--> video_engine.drive(bgr, latest_audio) --> outbound
    inbound audio frame --16k mono--> rvc_engine.convert_chunk(pcm) ------> outbound

Engines are injected via ``set_engines(...)``; until then (and whenever the offer
asks for ``engine="passthrough"``) it runs standalone with the passthrough
engines below — the video one stamps a "SYNTHETIC" watermark so the path is
visibly exercised with no real models loaded.

Mount it from the studio app with::

    from hallo4_studio.live_session import live_router
    app.include_router(live_router)                       # or add dependencies=[Depends(require_auth)]

Inject real engines once at startup::

    from hallo4_studio.live_session import set_engines
    set_engines(my_live_video_engine, my_rvc_engine)

Self-test: ``conda run --no-capture-output -n hallo4-thor python scripts/phase2_live_session_test.py``.
"""
from __future__ import annotations

import asyncio
import uuid
from fractions import Fraction
from typing import Any, Callable, Optional

import cv2
import numpy as np
from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaRelay
from av import AudioFrame, AudioResampler, VideoFrame
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

live_router = APIRouter()

# Engine audio contract: 16 kHz mono signed-16 PCM in and out.
ENGINE_RATE = 16000
_AUDIO_TIME_BASE = Fraction(1, ENGINE_RATE)

WATERMARK_TEXT = "SYNTHETIC"
_WM_BG = (0, 0, 255)        # BGR red badge — absent from natural webcam feeds, survives VP8
_WM_FG = (255, 255, 255)    # white text


def draw_watermark(bgr: np.ndarray) -> np.ndarray:
    """Stamp a red 'SYNTHETIC' badge in the bottom-left corner (in place)."""
    h, w = bgr.shape[:2]
    x0, y0 = 4, max(0, h - 22)
    x1, y1 = min(w - 1, x0 + 132), h - 4
    cv2.rectangle(bgr, (x0, y0), (x1, y1), _WM_BG, thickness=-1)
    cv2.putText(bgr, WATERMARK_TEXT, (x0 + 5, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, _WM_FG, 1, cv2.LINE_AA)
    return bgr


# --------------------------------------------------------------------------- #
# Default engines — let the whole transport run with no real models loaded.    #
# --------------------------------------------------------------------------- #
class PassthroughVideoEngine:
    """Returns the driving frame unchanged except for the synthetic watermark."""

    def __init__(self) -> None:
        self._ready = True

    def warmup(self, target_image_path: Optional[str] = None) -> None:  # noqa: ARG002
        self._ready = True

    @property
    def ready(self) -> bool:
        return self._ready

    def drive(self, driving_bgr: np.ndarray, audio_pcm16_16k: Any = None) -> np.ndarray:  # noqa: ARG002
        return draw_watermark(driving_bgr)


class PassthroughRVCEngine:
    """Returns the input audio chunk unchanged."""

    def load(self, model_path: Optional[str] = None) -> None:  # noqa: ARG002
        pass

    def convert_chunk(self, pcm16_16k: np.ndarray) -> np.ndarray:
        return pcm16_16k


_PASSTHROUGH_VIDEO = PassthroughVideoEngine()
_PASSTHROUGH_RVC = PassthroughRVCEngine()
_video_engine: Any = _PASSTHROUGH_VIDEO
_rvc_engine: Any = _PASSTHROUGH_RVC


def set_engines(video_engine: Any, rvc_engine: Any) -> None:
    """Inject the real engines used by new sessions (call once at startup)."""
    global _video_engine, _rvc_engine
    _video_engine = video_engine
    _rvc_engine = rvc_engine


# --------------------------------------------------------------------------- #
# Media transform tracks                                                       #
# --------------------------------------------------------------------------- #
class _VideoTransformTrack(MediaStreamTrack):
    """Reads inbound webcam frames, runs the video engine, re-emits BGR frames."""

    kind = "video"

    def __init__(self, source: MediaStreamTrack, engine: Any, latest_audio: Callable[[], Any]) -> None:
        super().__init__()
        self.source = source
        self.engine = engine
        self.latest_audio = latest_audio

    async def recv(self) -> VideoFrame:
        frame = await self.source.recv()
        bgr = frame.to_ndarray(format="bgr24")
        out = self.engine.drive(bgr, self.latest_audio())
        new = VideoFrame.from_ndarray(np.ascontiguousarray(out, dtype=np.uint8), format="bgr24")
        new.pts = frame.pts            # preserve timing so A/V stays in sync
        new.time_base = frame.time_base
        return new


class _AudioTransformTrack(MediaStreamTrack):
    """Resamples inbound audio to 16k mono, runs RVC, re-emits 16k mono s16.

    The Opus encoder resamples 16k->48k itself; it only requires s16 mono/stereo
    frames with a valid sample_rate/pts (see aiortc/codecs/opus.py)."""

    kind = "audio"

    def __init__(self, source: MediaStreamTrack, engine: Any, session: "LiveSession") -> None:
        super().__init__()
        self.source = source
        self.engine = engine
        self.session = session
        self.resampler = AudioResampler(format="s16", layout="mono", rate=ENGINE_RATE)
        self._pts = 0

    def _to_pcm16(self, frame: AudioFrame) -> np.ndarray:
        chunks = [f.to_ndarray().reshape(-1) for f in self.resampler.resample(frame)]
        if not chunks:
            return np.empty(0, dtype=np.int16)
        return np.concatenate(chunks).astype(np.int16)

    async def recv(self) -> AudioFrame:
        pcm = np.empty(0, dtype=np.int16)
        while pcm.size == 0:                       # resampler may buffer; pull until it yields
            pcm = self._to_pcm16(await self.source.recv())
        self.session.latest_audio = pcm            # feed the video engine's lip-sync input
        out = np.asarray(self.engine.convert_chunk(pcm), dtype=np.int16).reshape(1, -1)
        new = AudioFrame.from_ndarray(np.ascontiguousarray(out), format="s16", layout="mono")
        new.sample_rate = ENGINE_RATE
        new.pts = self._pts
        new.time_base = _AUDIO_TIME_BASE
        self._pts += out.shape[1]
        return new


# --------------------------------------------------------------------------- #
# Session registry — one active session at a time (single GPU).                #
# --------------------------------------------------------------------------- #
class LiveSession:
    def __init__(self, sid: str, pc: RTCPeerConnection, engine: str, video_engine: Any, rvc_engine: Any) -> None:
        self.id = sid
        self.pc = pc
        self.engine = engine
        self.video_engine = video_engine
        self.rvc_engine = rvc_engine
        self.relay = MediaRelay()
        self.latest_audio: Any = None
        self.has_video = False
        self.has_audio = False

    def summary(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "engine": self.engine,
            "connection_state": self.pc.connectionState,
            "has_video": self.has_video,
            "has_audio": self.has_audio,
        }


_sessions: dict[str, LiveSession] = {}
_lock = asyncio.Lock()      # ponytail: one global lock; sessions are rare and serialized anyway


async def _cleanup(sid: str) -> None:
    session = _sessions.pop(sid, None)
    if session is None:
        return
    try:
        await session.pc.close()
    except Exception:  # noqa: BLE001 — closing a dead pc must never raise
        pass


class OfferRequest(BaseModel):
    sdp: str
    type: str
    target_image: Optional[str] = None
    target_voice: Optional[str] = None
    engine: str = "live"        # "live" -> injected engines; "passthrough" -> built-in defaults


@live_router.post("/api/live/offer")
async def live_offer(offer: OfferRequest) -> dict[str, str]:
    async with _lock:
        if _sessions:
            raise HTTPException(status_code=409, detail="A live session is already active")

        if offer.engine == "passthrough":
            video_engine, rvc_engine = _PASSTHROUGH_VIDEO, _PASSTHROUGH_RVC
        else:
            video_engine, rvc_engine = _video_engine, _rvc_engine

        # Engines may load models; run off the event loop. Failures -> 500.
        try:
            if offer.target_image:
                await asyncio.to_thread(video_engine.warmup, offer.target_image)
            if offer.target_voice:
                await asyncio.to_thread(rvc_engine.load, offer.target_voice)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=f"Engine warmup failed: {exc}") from exc

        sid = uuid.uuid4().hex[:12]
        pc = RTCPeerConnection()
        session = LiveSession(sid, pc, offer.engine, video_engine, rvc_engine)
        _sessions[sid] = session

        @pc.on("track")
        def on_track(track: MediaStreamTrack) -> None:
            if track.kind == "video":
                session.has_video = True
                pc.addTrack(_VideoTransformTrack(
                    session.relay.subscribe(track), video_engine, lambda: session.latest_audio))
            elif track.kind == "audio":
                session.has_audio = True
                pc.addTrack(_AudioTransformTrack(
                    session.relay.subscribe(track), rvc_engine, session))

        @pc.on("connectionstatechange")
        async def on_state() -> None:
            # "disconnected" is often a transient ICE blip that recovers; only the
            # terminal states free the session (and the single-GPU slot).
            if pc.connectionState in ("failed", "closed"):
                await _cleanup(sid)

        try:
            await pc.setRemoteDescription(RTCSessionDescription(sdp=offer.sdp, type=offer.type))
            await pc.setLocalDescription(await pc.createAnswer())  # non-trickle: waits for ICE gather
        except Exception as exc:  # noqa: BLE001
            await _cleanup(sid)
            raise HTTPException(status_code=400, detail=f"Negotiation failed: {exc}") from exc

        return {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}


@live_router.get("/api/live/sessions")
async def live_sessions() -> dict[str, Any]:
    return {"active": bool(_sessions), "sessions": [s.summary() for s in _sessions.values()]}


@live_router.post("/api/live/{session_id}/stop")
async def live_stop(session_id: str) -> dict[str, str]:
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    await _cleanup(session_id)
    return {"stopped": session_id}
