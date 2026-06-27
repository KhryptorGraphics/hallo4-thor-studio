#!/usr/bin/env python
"""Phase 2 (live-mirror) feasibility spike — WebRTC transport self-test.

Runs a full aiortc session in one process: pc1 sends synthetic video frames over
WebRTC (DTLS/SRTP + PyAV VP8 encode), pc2 receives and decodes them. No browser,
no network — just proves the transport stack works on Thor against av's bundled
ffmpeg. Measures encode+transport+decode throughput as a first latency signal.

Run: conda run -n hallo4-thor python scripts/phase2_aiortc_loopback.py
"""
from __future__ import annotations

import asyncio
import time

import numpy as np
from aiortc import RTCPeerConnection, VideoStreamTrack
from av import VideoFrame

WIDTH, HEIGHT, N_FRAMES = 320, 240, 30


class SyntheticTrack(VideoStreamTrack):
    """Emits a moving green bar so each frame is distinct (decode is exercised)."""

    def __init__(self) -> None:
        super().__init__()
        self.counter = 0

    async def recv(self) -> VideoFrame:
        pts, time_base = await self.next_timestamp()
        arr = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        arr[:, (self.counter * 8) % WIDTH] = (0, 255, 0)
        self.counter += 1
        frame = VideoFrame.from_ndarray(arr, format="rgb24")
        frame.pts, frame.time_base = pts, time_base
        return frame


async def main() -> None:
    pc1, pc2 = RTCPeerConnection(), RTCPeerConnection()
    received: asyncio.Queue = asyncio.Queue()

    @pc2.on("track")
    def on_track(track):  # noqa: ANN001
        async def reader() -> None:
            try:
                while True:
                    frame = await track.recv()
                    await received.put((frame.width, frame.height))
            except Exception:
                pass

        asyncio.ensure_future(reader())

    pc1.addTrack(SyntheticTrack())

    # Non-trickle signaling: setLocalDescription waits for ICE gather, so the SDP
    # already carries host candidates — enough for a same-host loopback.
    await pc1.setLocalDescription(await pc1.createOffer())
    await pc2.setRemoteDescription(pc1.localDescription)
    await pc2.setLocalDescription(await pc2.createAnswer())
    await pc1.setRemoteDescription(pc2.localDescription)

    dims = []
    start = None
    for _ in range(N_FRAMES):
        dim = await asyncio.wait_for(received.get(), timeout=20)
        if start is None:
            start = time.perf_counter()  # first decoded frame = connection established
        dims.append(dim)
    elapsed = time.perf_counter() - start

    await pc1.close()
    await pc2.close()

    assert all(d == (WIDTH, HEIGHT) for d in dims), f"bad frame dims: {set(dims)}"
    fps = (len(dims) - 1) / elapsed if elapsed > 0 else float("inf")
    print(f"Decoded {len(dims)} frames @ {WIDTH}x{HEIGHT}; "
          f"transport throughput ~{fps:.1f} fps over {elapsed*1000:.0f} ms")
    print("LOOPBACK_OK")


if __name__ == "__main__":
    asyncio.run(main())
