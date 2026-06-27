#!/usr/bin/env python
"""Phase 2 live-mirror session self-test (in-process, no browser/network).

Models scripts/phase2_aiortc_loopback.py but drives the real session layer:
pc1 plays a synthetic webcam (+ mic) and sends a WebRTC offer straight into the
``live_offer`` handler; the session transforms each frame through the passthrough
engines and streams it back; pc1 receives it. Asserts video frames flow at the
expected dims with the SYNTHETIC watermark present, audio frames flow, the
single-session guard returns 409, and stop cleans the registry.

Run: conda run --no-capture-output -n hallo4-thor python scripts/phase2_live_session_test.py
"""
from __future__ import annotations

import asyncio
import sys
import time
from fractions import Fraction
from pathlib import Path

import numpy as np
from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from av import AudioFrame, VideoFrame
from fastapi import HTTPException

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from studio.backend.hallo4_studio.live_session import (  # noqa: E402
    OfferRequest,
    live_offer,
    live_sessions,
    live_stop,
)

WIDTH, HEIGHT = 320, 240
N_VIDEO, N_AUDIO_MIN = 25, 5


class SyntheticWebcamTrack(VideoStreamTrack):
    """Gray frame with a moving white bar — a stand-in driving webcam feed."""

    def __init__(self) -> None:
        super().__init__()
        self.counter = 0

    async def recv(self) -> VideoFrame:
        pts, time_base = await self.next_timestamp()
        arr = np.full((HEIGHT, WIDTH, 3), 100, dtype=np.uint8)  # mid-gray (no red)
        arr[:, (self.counter * 8) % WIDTH] = (255, 255, 255)
        self.counter += 1
        frame = VideoFrame.from_ndarray(arr, format="bgr24")
        frame.pts, frame.time_base = pts, time_base
        return frame


class SyntheticMicTrack(MediaStreamTrack):
    """48 kHz mono s16 tone, paced in real time — a stand-in driving mic feed."""

    kind = "audio"

    def __init__(self) -> None:
        super().__init__()
        self.rate, self.samples = 48000, 960
        self._pts, self._start = 0, None

    async def recv(self) -> AudioFrame:
        if self._start is None:
            self._start = time.time()
        else:
            delay = self._start + (self._pts / self.rate) - time.time()
            if delay > 0:
                await asyncio.sleep(delay)
        t = np.arange(self._pts, self._pts + self.samples) / self.rate
        tone = (0.2 * np.sin(2 * np.pi * 220.0 * t) * 32767).astype(np.int16)
        frame = AudioFrame.from_ndarray(tone.reshape(1, -1), format="s16", layout="mono")
        frame.sample_rate = self.rate
        frame.pts, frame.time_base = self._pts, Fraction(1, self.rate)
        self._pts += self.samples
        return frame


def watermark_score(frame: VideoFrame) -> tuple[int, int, int]:
    """(width, height, red-dominant pixel count in the bottom-left badge region)."""
    img = frame.to_ndarray(format="bgr24")
    h, w = img.shape[:2]
    patch = img[max(0, h - 26):h, 0:min(w, 144)].astype(np.int16)
    b, g, r = patch[..., 0], patch[..., 1], patch[..., 2]
    mask = (r > 120) & (r > g + 40) & (r > b + 40)
    return w, h, int(mask.sum())


async def main() -> None:
    pc1 = RTCPeerConnection()
    video_q: asyncio.Queue = asyncio.Queue()
    audio_q: asyncio.Queue = asyncio.Queue()

    @pc1.on("track")
    def on_track(track: MediaStreamTrack) -> None:
        queue = video_q if track.kind == "video" else audio_q
        async def reader() -> None:
            try:
                while True:
                    frame = await track.recv()
                    if track.kind == "video":
                        await queue.put(watermark_score(frame))
                    else:
                        await queue.put(int(frame.samples))
            except Exception:
                pass
        asyncio.ensure_future(reader())

    pc1.addTrack(SyntheticWebcamTrack())
    pc1.addTrack(SyntheticMicTrack())

    # Offer -> server handler -> answer (non-trickle: localDescription already has ICE).
    await pc1.setLocalDescription(await pc1.createOffer())
    answer = await live_offer(OfferRequest(
        sdp=pc1.localDescription.sdp, type=pc1.localDescription.type, engine="passthrough"))

    # Session is registered immediately; a second offer must be rejected (single GPU).
    state = await live_sessions()
    assert state["active"] and len(state["sessions"]) == 1, state
    sid = state["sessions"][0]["id"]
    try:
        await live_offer(OfferRequest(sdp=pc1.localDescription.sdp, type="offer", engine="passthrough"))
        raise AssertionError("expected 409 on a second concurrent session")
    except HTTPException as exc:
        assert exc.status_code == 409, exc.status_code

    await pc1.setRemoteDescription(RTCSessionDescription(sdp=answer["sdp"], type=answer["type"]))

    dims, scores, start = [], [], None
    for _ in range(N_VIDEO):
        w, h, score = await asyncio.wait_for(video_q.get(), timeout=20)
        if start is None:
            start = time.perf_counter()
        dims.append((w, h))
        scores.append(score)
    elapsed = time.perf_counter() - start

    audio_frames = []
    try:
        while len(audio_frames) < N_AUDIO_MIN:
            audio_frames.append(await asyncio.wait_for(audio_q.get(), timeout=20))
    except asyncio.TimeoutError:
        pass

    # Stop and confirm the registry is clean.
    stopped = await live_stop(sid)
    assert stopped == {"stopped": sid}, stopped
    assert (await live_sessions())["active"] is False
    try:
        await live_stop(sid)
        raise AssertionError("expected 404 stopping an unknown session")
    except HTTPException as exc:
        assert exc.status_code == 404, exc.status_code
    await pc1.close()

    assert all(d == (WIDTH, HEIGHT) for d in dims), f"bad dims: {set(dims)}"
    watermarked = sum(1 for s in scores[-10:] if s > 200)
    assert watermarked >= 8, f"watermark missing: last-10 scores={scores[-10:]}"
    assert len(audio_frames) >= N_AUDIO_MIN and all(s > 0 for s in audio_frames), audio_frames

    fps = (len(dims) - 1) / elapsed if elapsed > 0 else float("inf")
    print(f"Relayed {len(dims)} video frames @ {WIDTH}x{HEIGHT} (~{fps:.1f} fps), "
          f"{len(audio_frames)} audio frames; watermark on {watermarked}/10 recent frames")
    print("SESSION_OK")


if __name__ == "__main__":
    asyncio.run(main())
