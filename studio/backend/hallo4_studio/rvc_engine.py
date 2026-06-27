# -*- coding: utf-8 -*-
"""Realtime voice conversion for the Phase 2 live mirror.

    int16 16k (user speech) --window--> WavLM features (REAL)
        --> kNN match against the ENROLLED target's features (REAL)
        --> HiFiGAN vocode (REAL) --overlap-add--> int16 16k, same length

REAL zero-shot timbre conversion, no per-voice training
-------------------------------------------------------
The backend is **kNN-VC** (WavLM-Large encoder + HiFiGAN vocoder, ``torch.hub
bshall/knn-vc``), reused as a *streaming* converter via the proven, env-safe
``vace/models/utils/voice_convert.VoiceConverter`` (pure PyTorch — no new pip
dep, no ``onnxruntime``/numpy clobber, soundfile IO shim instead of broken
torchaudio). Enrollment is zero-shot: the reference clip's WavLM features *are*
the model, so there is nothing to train — ``RVCTrainer.enroll`` validates the
clip and writes ``reference.wav`` + ``model.json``; ``RVCEngine.load`` builds a
resident converter and precomputes the target matching-set once.

Streaming design (see :class:`RVCEngine`): a ring buffer accumulates incoming
16k int16; once a ~0.4 s window fills it is converted and overlap-added (linear
crossfade) into an output buffer emitted at the input rate, so each
``convert_chunk`` returns int16 of the *same length* as its input (the live
session advances its audio pts by that length). Added latency ≈ one window
(~0.4 s; tunable via HALLO4_RVC_WIN — compute is only ~12 ms, so the window is
the latency lever). Before the first window fills, or with no/failed model,
audio passes through unchanged. ``convert_chunk`` never raises — a bad frame
must not kill the session.

TODO(rvc): a vendored *trained* RVC (per-voice ``net_g`` + RMVPE f0) could cut
latency / raise fidelity below zero-shot, but is install-fragile on
aarch64/py3.12 and would clobber the GPU onnxruntime / numpy pin. Left as a
future follow-on; kNN-VC is the shipping path.

Audio IO uses ``soundfile``/``librosa`` (torchaudio load/save are broken on Thor
— no torchcodec; see vace/models/utils/voice_convert.py). Self-test::

    conda run --no-capture-output -n hallo4-thor python \
        studio/backend/hallo4_studio/rvc_engine.py
"""
from __future__ import annotations

import json
import logging
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000

# Streaming analysis window / hop for kNN-VC. Added latency ≈ one window, so the
# window is the latency lever (compute is ~12 ms regardless — RTF ~0.03). Measured
# quality floor on Thor: out_rms tracks the source down to ~0.32 s, then collapses
# (<0.2 s windows vocode to near-silence — WavLM/HiFiGAN need a minimum context).
# So 0.40 s is the low-latency sweet spot (safely above the floor): ~0.4 s latency,
# ~2.5x better than the old 1.0 s, quality intact. Env-tunable for experimentation.
# ponytail: shrinking the window is the whole win here; a context-window /
# emit-newest-hop redesign could reach ~0.16 s but adds alignment/artifact risk.
_STREAM_WIN = int(os.getenv("HALLO4_RVC_WIN", "6400"))   # 0.40 s WavLM context
_STREAM_HOP = int(os.getenv("HALLO4_RVC_HOP", "3200"))   # 0.20 s (50% overlap)
# Windows quieter than this (RMS, ~-80 dBFS) emit silence instead of letting
# HiFiGAN hallucinate target breath/hiss into the gaps between words.
_SILENCE_RMS = 1e-4

REPO_ROOT = Path(__file__).resolve().parents[3]
# Mirror app.py's DATA_ROOT so a live session can hand load() a bare model-id.
# ponytail: a 2-line dup beats importing the whole FastAPI app into the engine.
_DATA_ROOT = Path(os.getenv("HALLO4_STUDIO_DATA", REPO_ROOT / "studio_data")).resolve()
_VOICE_MODELS_DIR = _DATA_ROOT / "voice_models"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class RVCEngine:
    """Per-chunk streaming voice converter (real zero-shot kNN-VC).

    Robust by design: ``convert_chunk`` never raises and always returns int16 of
    the input length — with no model loaded, while the first window fills, on a
    silent/tiny chunk, or if conversion fails it returns the input unchanged. The
    live session must not die on a bad frame, and its audio pts advances by the
    returned length, so same-length-out is a hard contract.
    """

    def __init__(self, device: str = "cuda") -> None:
        self.device = device
        self._vc = None                 # resident VoiceConverter (kNN-VC), shared across chunks
        self._matching_set = None       # precomputed TARGET features (resident); None => passthrough
        self._model_meta: Optional[dict] = None
        self._gain = 1.0
        self._ready = False
        self._engaged = False           # True once a real converted chunk has been emitted
        # streaming state (all float32, mono, 16 kHz)
        self._in_buf = np.zeros(0, dtype=np.float32)   # pending input not yet windowed
        self._out_buf = np.zeros(0, dtype=np.float32)  # converted audio ready to emit
        self._xfade_tail: Optional[np.ndarray] = None  # trailing samples of last window (for crossfade)

    @property
    def ready(self) -> bool:
        """True once a real kNN-VC voice model is loaded and converting.

        False keeps the session alive in passthrough (convert_chunk still returns
        the input); it just signals that timbre conversion is not engaged.
        """
        return self._ready

    # -- voice model -------------------------------------------------------- #
    @staticmethod
    def _resolve_model_json(model_path: str) -> Optional[Path]:
        """Accept a model.json path, a model dir, or a bare enrolled model-id."""
        if not model_path:
            return None
        p = Path(model_path)
        candidates = [
            p if p.name == "model.json" else None,
            p / "model.json",
            _VOICE_MODELS_DIR / model_path / "model.json",
        ]
        for c in candidates:
            if c is not None and c.is_file():
                return c
        return None

    def _reset_stream(self) -> None:
        self._in_buf = np.zeros(0, dtype=np.float32)
        self._out_buf = np.zeros(0, dtype=np.float32)
        self._xfade_tail = None

    def _passthrough(self) -> None:
        """Drop into passthrough: keep any resident converter, stop converting."""
        self._matching_set = None
        self._ready = False
        self._reset_stream()

    def load(self, model_path: str) -> None:
        """Load an enrolled voice model and arm real kNN-VC conversion.

        Resolves the model dir, reads ``reference.wav``, builds a resident
        VoiceConverter (WavLM + HiFiGAN) and **precomputes the target
        matching-set once**, then sets ``ready=True``. Tolerant by design: an
        unresolved/broken model, a missing reference, or a kNN-VC load failure
        logs a warning and stays in passthrough (``ready=False``) rather than
        raising and killing the session warmup.
        """
        meta_path = self._resolve_model_json(model_path)
        if meta_path is None:
            logger.warning("RVC voice model not found for %r; staying in passthrough.", model_path)
            self._model_meta = None
            self._passthrough()
            return
        try:
            self._model_meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to read RVC model.json %s (%s); passthrough.", meta_path, exc)
            self._model_meta = None
            self._passthrough()
            return

        self._gain = float(self._model_meta.get("gain", 1.0) or 1.0)
        ref = meta_path.parent / str(self._model_meta.get("reference", "reference.wav"))
        if not ref.is_file():
            logger.warning("RVC reference clip missing (%s); passthrough.", ref)
            self._passthrough()
            return

        # Build the resident converter and precompute the TARGET features ONCE.
        # Heavy (loads WavLM-Large + HiFiGAN) — done here at warmup, never per chunk.
        try:
            from vace.models.utils.voice_convert import VoiceConverter

            topk = int(self._model_meta.get("topk", 4) or 4)
            vc = VoiceConverter(device=self.device, topk=topk)
            matching_set = vc.precompute_target(str(ref))
            self._vc = vc
            self._matching_set = matching_set
            self.device = vc.device
            self._reset_stream()
            self._engaged = False
            self._ready = True
            logger.info(
                "RVC kNN-VC loaded from %s (target=%s, topk=%d, gain=%.3f) on %s.",
                meta_path.parent, ref.name, topk, self._gain, self.device,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "kNN-VC unavailable (%s: %s); RVC stays in passthrough.",
                type(exc).__name__, exc,
            )
            self._passthrough()

    # -- realtime conversion ------------------------------------------------ #
    def _convert_window(self, window: np.ndarray) -> np.ndarray:
        """Convert one full analysis window (float32 16k [-1,1]) to the target
        timbre. Levels the output to the source window's RMS for continuity (and
        to avoid clipping), then applies the optional per-voice gain knob."""
        s_rms = float(np.sqrt(np.mean(window ** 2)))
        if s_rms < _SILENCE_RMS:                       # silence in -> silence out
            return np.zeros(window.shape[0], dtype=np.float32)
        out = self._vc.convert_stream(window, self._matching_set)
        o_rms = float(np.sqrt(np.mean(out ** 2))) if out.size else 0.0
        if o_rms > 1e-6:                               # match perceived level to the source
            out = out * (s_rms / o_rms)
        if self._gain != 1.0:
            out = out * self._gain
        return np.clip(out, -1.0, 1.0).astype(np.float32)

    def _ola_add(self, w: np.ndarray) -> None:
        """Overlap-add a converted window onto ``_out_buf`` with a linear
        crossfade over the region it shares with the previous window. Commits
        ``_STREAM_HOP`` samples per window (matching the input hop, so the stream
        stays rate-synchronous) and holds the trailing overlap for the next join."""
        L = w.shape[0]
        if L == 0:
            return
        hop = min(_STREAM_HOP, L)
        if self._xfade_tail is None:                   # first window: no fade-in
            self._out_buf = np.concatenate([self._out_buf, w[:hop]])
            self._xfade_tail = w[hop:].copy()
            return
        tail = self._xfade_tail
        ov = min(tail.shape[0], L)                      # crossfade length = shared overlap
        if ov > 0:
            ramp = np.linspace(0.0, 1.0, ov, dtype=np.float32)
            blended = tail[:ov] * (1.0 - ramp) + w[:ov] * ramp
        else:
            blended = np.zeros(0, dtype=np.float32)
        body = w[ov:hop] if hop > ov else np.zeros(0, dtype=np.float32)
        self._out_buf = np.concatenate([self._out_buf, blended, body])
        self._xfade_tail = w[hop:].copy()

    def convert_chunk(self, pcm16_16k: np.ndarray) -> np.ndarray:
        """Convert one realtime chunk. int16 mono 16k in -> int16 same length out.

        Buffers input into ~1 s windows, kNN-VC converts each window against the
        enrolled target, overlap-adds the result, and emits the same number of
        samples as came in. Returns the input unchanged while priming (before the
        first window), with no model loaded, or on any failure. Never raises.
        """
        pcm = np.ascontiguousarray(np.asarray(pcm16_16k)).reshape(-1)
        n = pcm.shape[0]
        if n == 0:
            return np.zeros(0, dtype=np.int16)
        if pcm.dtype != np.int16:
            pcm = np.clip(np.rint(pcm.astype(np.float64)), -32768, 32767).astype(np.int16)

        # No model / kNN-VC not engaged -> passthrough (audio keeps flowing).
        if not self._ready or self._matching_set is None:
            return pcm

        try:
            self._in_buf = np.concatenate([self._in_buf, pcm.astype(np.float32) / 32768.0])
            while self._in_buf.shape[0] >= _STREAM_WIN:
                window = self._in_buf[:_STREAM_WIN].copy()
                self._ola_add(self._convert_window(window))
                self._in_buf = self._in_buf[_STREAM_HOP:]

            if self._out_buf.shape[0] >= n:
                out = self._out_buf[:n]
                self._out_buf = self._out_buf[n:]
                self._engaged = True
                return np.clip(out * 32768.0, -32768.0, 32767.0).astype(np.int16)
            # Priming: not enough converted audio yet -> input unchanged.
            # ponytail: a one-time hard cut at the priming->converted handoff;
            # crossfade it if the seam is audible (TODO, low priority).
            return pcm
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "RVC convert_chunk failed (%s: %s); passthrough for this chunk.",
                type(exc).__name__, exc,
            )
            return pcm


class RVCTrainer:
    """Offline voice enrollment (zero-shot — no training).

    kNN-VC needs no per-voice training: the reference clip's WavLM features *are*
    the model. ``enroll`` validates the clip and writes a self-contained artifact
    (``reference.wav`` + ``model.json``) that :class:`RVCEngine` loads to convert
    any speech into this target's timbre.
    """

    @staticmethod
    def enroll(audio_path: str, out_model_path: str, on_log: Optional[Callable[[str], None]] = None) -> str:
        """Validate ``audio_path`` and write a voice-model artifact dir.

        Writes ``out_model_path/model.json`` (describing the enrolled voice) plus
        a self-contained ``reference.wav`` copy of the clip. Returns the model
        directory path. ``on_log`` receives human-readable progress strings.
        """
        import soundfile as sf

        def log(msg: str) -> None:
            logger.info(msg)
            if on_log is not None:
                try:
                    on_log(msg)
                except Exception:  # noqa: BLE001 — logging must never fail the job
                    pass

        src = Path(audio_path)
        if not src.is_file():
            raise FileNotFoundError(f"Enrollment audio not found: {audio_path}")

        log(f"Validating enrollment audio {src.name} ...")
        info = sf.info(str(src))  # raises if libsndfile can't read it
        if info.frames <= 0 or info.samplerate <= 0:
            raise ValueError(f"Enrollment audio is empty or unreadable: {audio_path}")
        duration = info.frames / float(info.samplerate)
        log(f"Audio OK: {duration:.1f}s, {info.samplerate} Hz, {info.channels}ch.")
        if duration < 3.0:
            log("Note: clip is short; more clean reference speech improves the cloned timbre.")

        out = Path(out_model_path)
        out.mkdir(parents=True, exist_ok=True)
        ref = out / "reference.wav"
        if src.resolve() != ref.resolve():
            shutil.copyfile(src, ref)  # self-contained artifact (caller guarantees readable audio)
            log(f"Stored reference clip -> {ref.name}")

        # Zero-shot: nothing to train — the reference's WavLM features are the
        # model (RVCEngine precomputes them at load). TODO(rvc): a vendored
        # *trained* RVC (per-voice net_g + RMVPE f0) could lower latency / raise
        # fidelity, but is install-fragile on aarch64 and would clobber the GPU
        # onnxruntime / numpy pin — left as a future follow-on.
        meta = {
            "version": 2,
            "stub": False,
            "name": None,  # the enrollment job owns the human name (kept in its registry)
            "source_audio": str(src.resolve()),
            "reference": ref.name,
            "duration_sec": round(duration, 3),
            "sample_rate": int(info.samplerate),
            "channels": int(info.channels),
            "frames": int(info.frames),
            "encoder": "wavlm-large",
            "synthesizer": "knn-vc",
            "topk": 4,
            "gain": 1.0,
            "created_at": _now_iso(),
            "note": (
                "Real zero-shot voice conversion via kNN-VC (WavLM-Large + HiFiGAN, "
                "torch.hub bshall/knn-vc): the reference clip's WavLM features are "
                "the model, no per-voice training. A vendored trained-RVC (net_g + "
                "RMVPE f0) remains a documented future follow-on."
            ),
        }
        (out / "model.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        log("Enrollment complete (kNN-VC voice model written).")
        return str(out)


def demo() -> None:
    """Self-check: enroll 01.wav as the TARGET, stream a DIFFERENT source through
    convert_chunk, prove kNN-VC engaged, save wavs for A/B, confirm env intact."""
    import librosa
    import soundfile as sf

    src_clip = REPO_ROOT / "assets" / "01.wav"
    assert src_clip.is_file(), f"missing test asset {src_clip}"

    out_dir = _DATA_ROOT / "rvc_demo"
    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir = out_dir / "model"

    # 1) enroll the target voice (real zero-shot artifact) + check the metadata.
    out = RVCTrainer.enroll(str(src_clip), str(model_dir))
    meta = json.loads((Path(out) / "model.json").read_text(encoding="utf-8"))
    assert meta["stub"] is False, meta
    assert meta["synthesizer"] == "knn-vc", meta
    assert (Path(out) / "reference.wav").is_file()
    print(f"RVC_META stub={meta['stub']} synthesizer={meta['synthesizer']}")

    # 2) build a DIFFERENT source: 01.wav (first 6 s) pitch-shifted down so it
    #    sounds like another speaker; converting it should pull timbre back to 01.
    y, _sr = librosa.load(str(src_clip), sr=SAMPLE_RATE, mono=True)
    y = y[: SAMPLE_RATE * 6]
    y = librosa.effects.pitch_shift(y, sr=SAMPLE_RATE, n_steps=-4.0)
    src_pcm = np.clip(y * 32768.0, -32768, 32767).astype(np.int16)
    src_path = out_dir / "source_pitchshifted.wav"
    sf.write(str(src_path), src_pcm, SAMPLE_RATE)

    # 3) load + stream the source through convert_chunk in ~20 ms slices.
    eng = RVCEngine(device=os.getenv("HALLO4_RVC_DEVICE", "cuda"))
    eng.load(out)
    assert eng.ready, "engine should be ready after loading a kNN-VC model"

    slice_n = SAMPLE_RATE // 50            # 20 ms = 320 samples
    chunks_out = []
    for i in range(0, src_pcm.shape[0], slice_n):
        chunk = src_pcm[i:i + slice_n]
        y_out = eng.convert_chunk(chunk)
        assert y_out.dtype == np.int16, y_out.dtype
        assert y_out.shape[0] == chunk.shape[0], (y_out.shape, chunk.shape)
        chunks_out.append(y_out)
    converted = np.concatenate(chunks_out)
    assert converted.shape[0] == src_pcm.shape[0], (converted.shape, src_pcm.shape)
    conv_path = out_dir / "converted.wav"
    sf.write(str(conv_path), converted, SAMPLE_RATE)

    # robustness: tiny / empty / wrong-dtype chunks honour the contract.
    assert eng.convert_chunk(np.zeros(1, np.int16)).shape[0] == 1
    assert eng.convert_chunk(np.zeros(0, np.int16)).shape[0] == 0
    assert eng.convert_chunk(np.zeros(160, np.float32)).dtype == np.int16

    # proof kNN-VC engaged: the converted tail (past the ~1 s passthrough priming
    # head) must actually differ from the source — not an identity passthrough.
    seg = slice(SAMPLE_RATE, None)
    moved = float(np.mean(np.abs(converted[seg].astype(np.float32) - src_pcm[seg].astype(np.float32))))
    print(f"RVC_ENGAGED knn_vc={'YES' if eng._engaged else 'NO (passthrough)'} mean_abs_delta={moved:.1f}")
    print(f"RVC_AB source={src_path}")
    print(f"RVC_AB converted={conv_path}")
    assert eng._engaged, "kNN-VC did not engage (stayed passthrough)"
    assert moved > 1.0, "converted output is identical to source (passthrough?)"

    # 4) env integrity: numpy pin + GPU onnxruntime providers must be untouched.
    import onnxruntime
    print(f"ENV numpy={np.__version__}")
    print(f"ENV onnxruntime_providers={onnxruntime.get_available_providers()}")
    assert np.__version__ == "1.26.4", np.__version__

    print("RVC_OK kNN-VC streaming voice conversion engaged.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    demo()
