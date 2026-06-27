# -*- coding: utf-8 -*-
"""Zero-shot voice conversion for Hallo4 Studio.

Converts a *driving* speech clip (e.g. the user talking into a mic) into the
*timbre* of a target speaker given a short reference clip of that target. The
converted WAV is then used both to drive lip-sync and as the final muxed audio,
so the animated portrait speaks the user's words in the target's voice.

Backend: kNN-VC (https://github.com/bshall/knn-vc), pulled via ``torch.hub``.
It is any-to-any, zero-shot, and pure PyTorch — no new pip dependency and, in
particular, no ``onnxruntime`` (which on aarch64/Thor would clobber the
locally-built ``onnxruntime-gpu`` build). The backend is hidden behind a tiny
interface so OpenVoice v2 / seed-vc can be swapped in later.

Audio IO note: torchaudio 2.10 routes load/save through ``torchcodec`` (not
installed on Thor) and ignores the ``backend=`` kwarg, so torchaudio IO is
unusable here. We use ``soundfile`` + ``librosa`` (both work) for our own IO and
patch ``torchaudio.load``/``save`` with soundfile shims so the kNN-VC hub code's
internal ``torchaudio.load`` calls also work — no torchcodec install required.

# ponytail: torch.hub kNN-VC + soundfile shim chosen to avoid a fragile aarch64
# torchcodec/pip install. Upgrade path: install torchcodec, or add an
# OpenVoice/seed-vc backend, if timbre fidelity is too low.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000


def _patch_torchaudio_io() -> None:
    """Make ``torchaudio.load``/``save`` work without torchcodec, via soundfile.

    No-op when torchcodec is importable (native path already works). kNN-VC's
    hub code calls ``torchaudio.load`` internally; this keeps it working.
    """
    import torchaudio

    try:
        import torchcodec  # noqa: F401
        return
    except Exception:
        pass

    import soundfile as sf
    import torch

    def _load(filepath, frame_offset=0, num_frames=-1, normalize=True,
              channels_first=True, format=None, backend=None):
        data, sr = sf.read(str(filepath), dtype="float32", always_2d=True)  # (T, C)
        if frame_offset or (num_frames is not None and num_frames > 0):
            end = None if (num_frames is None or num_frames < 0) else frame_offset + num_frames
            data = data[frame_offset:end]
        # Downmix to mono — speech models (WavLM/kNN-VC) expect a single channel,
        # and mismatched per-channel handling otherwise corrupts feature lengths.
        data = data.mean(axis=1, keepdims=True)  # (T, 1)
        wav = torch.from_numpy(data.T).contiguous()  # (1, T)
        return (wav, sr) if channels_first else (wav.t(), sr)

    def _save(filepath, src, sample_rate, channels_first=True, format=None, backend=None, **_kw):
        arr = src.detach().cpu().numpy()
        if arr.ndim == 1:
            arr = arr[None]
        if channels_first:
            arr = arr.T  # soundfile wants (T, C)
        sf.write(str(filepath), arr, int(sample_rate))

    torchaudio.load = _load
    torchaudio.save = _save
    logger.info("Patched torchaudio.load/save to use soundfile (torchcodec absent).")


def _write_wav(path: str, wav, sr: int = SAMPLE_RATE) -> None:
    import numpy as np
    import soundfile as sf

    arr = wav.detach().cpu().numpy() if hasattr(wav, "detach") else np.asarray(wav)
    arr = np.squeeze(arr)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    sf.write(path, arr, int(sr))


class VoiceConverter:
    """Lazy-loaded zero-shot voice converter.

    Parameters
    ----------
    device: torch device string ("cuda", "cuda:0", "cpu").
    backend: only "knn-vc" is implemented today.
    topk: kNN neighbours averaged per frame (higher = smoother, less crisp).
    """

    def __init__(self, device: str = "cuda", backend: str = "knn-vc", topk: int = 4) -> None:
        self.device = device
        self.backend = backend
        self.topk = topk
        self._model = None

    def load(self) -> "VoiceConverter":
        if self._model is not None:
            return self
        if self.backend != "knn-vc":
            raise ValueError(f"Unsupported voice-conversion backend: {self.backend}")

        import torch

        device = self.device
        if device.startswith("cuda") and not torch.cuda.is_available():
            logger.warning("CUDA unavailable for voice conversion; falling back to CPU.")
            device = "cpu"

        _patch_torchaudio_io()
        logger.info("Loading kNN-VC voice-conversion model (torch.hub bshall/knn-vc)...")
        self._model = torch.hub.load(
            "bshall/knn-vc", "knn_vc",
            prematched=True, trust_repo=True, pretrained=True, device=device,
        )
        self.device = device
        logger.info("kNN-VC ready on %s.", device)
        return self

    def convert(self, src_wav: str, ref_wav, out_wav: str) -> str:
        """Convert ``src_wav`` into the timbre of ``ref_wav``; write 16 kHz mono ``out_wav``.

        ``ref_wav`` may be a path or list of paths (more reference audio improves
        the target timbre). Returns ``out_wav``.
        """
        if not os.path.exists(src_wav):
            raise FileNotFoundError(f"Source audio not found: {src_wav}")
        refs = list(ref_wav) if isinstance(ref_wav, (list, tuple)) else [ref_wav]
        for r in refs:
            if not os.path.exists(r):
                raise FileNotFoundError(f"Reference voice not found: {r}")

        self.load()

        import torch

        with torch.no_grad():
            query_seq = self._model.get_features(src_wav)
            matching_set = self._model.get_matching_set(refs)
            out = self._model.match(query_seq, matching_set, topk=self.topk)  # 1-D, 16 kHz

        _write_wav(out_wav, out, SAMPLE_RATE)
        logger.info("Voice-converted %s -> %s (%.2fs)", src_wav, out_wav, out.numel() / SAMPLE_RATE)
        return out_wav


def demo() -> None:
    """Self-check.

    Default: validate the interface and soundfile WAV round-trip (the non-model
    logic). Set HALLO4_VC_FULL_TEST=1 to also run a real conversion (downloads
    the kNN-VC + WavLM weights on first run).
    """
    import tempfile

    import soundfile as sf

    here = os.path.dirname(__file__)
    src = os.path.abspath(os.path.join(here, "..", "..", "..", "assets", "01.wav"))
    assert os.path.exists(src), f"missing test asset {src}"

    data, sr = sf.read(src, dtype="float32")
    assert data.size > 0 and sr > 0
    with tempfile.TemporaryDirectory() as d:
        rt = os.path.join(d, "roundtrip.wav")
        _write_wav(rt, data, sr)
        back, sr2 = sf.read(rt)
        assert os.path.getsize(rt) > 0 and sr2 == sr
    vc = VoiceConverter(device="cpu")
    assert vc.backend == "knn-vc" and vc._model is None
    print("OK: VoiceConverter interface + soundfile IO round-trip")

    if os.getenv("HALLO4_VC_FULL_TEST") == "1":
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "converted.wav")
            VoiceConverter(device=os.getenv("HALLO4_VC_DEVICE", "cuda")).convert(src, src, out)
            info = sf.info(out)
            assert info.samplerate == SAMPLE_RATE, info.samplerate
            assert info.frames > 0
            print(f"OK: full conversion -> {info.frames/SAMPLE_RATE:.2f}s @ {info.samplerate} Hz")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    demo()
