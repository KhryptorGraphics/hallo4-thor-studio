# -*- coding: utf-8 -*-
"""Realtime voice conversion (RVC) for the Phase 2 live mirror.

Structure of the real pipeline (target):

    int16 16k --normalize--> HuBERT/wav2vec2 content encoder (REAL)
              --> RMVPE f0 + enrolled speaker model --> net_g synthesizer (TODO)
              --> int16 16k waveform, same length

What is REAL here vs STUBBED
----------------------------
* **REAL** — the *content encoder*: the repo's ``wav2vec2-base-960h`` transformer
  (a ~94M HuBERT/ContentVec-class net) is loaded via ``transformers`` and run on
  every chunk, so ``convert_chunk`` exercises the dominant compute leg for
  cost-realism (~5 ms/chunk on Thor, see studio/PHASE2_LIVE_MIRROR.md).
* **STUBBED** — the *synthesizer* and *training*. Real RVC needs fairseq-HuBERT +
  a trained ``net_g`` + RMVPE f0, which are install-fragile on aarch64/py3.12 and
  would drag in CPU ``onnxruntime`` / bump numpy and break this env. Until that
  lands, the synthesizer is an identity passthrough (with an optional per-voice
  ``gain`` knob) so audio keeps flowing through the live session, and "training"
  just validates the clip and writes a metadata artifact. ``ready`` reports
  whether an enrolled (stub) voice model is loaded.

A working passthrough beats a broken env. Search for ``TODO(rvc)`` for the exact
points where the real f0/synthesizer/trainer slot in.

Audio IO uses ``soundfile`` (torchaudio load/save are broken on Thor — no
torchcodec; see vace/models/utils/voice_convert.py). Self-test::

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
# wav2vec2's conv stack has a ~400-sample (25 ms) receptive field; a shorter
# input can't produce even one feature frame, so we pad up to this for the
# encoder pass only (the returned PCM keeps its original length).
_MIN_ENCODER_SAMPLES = 400

REPO_ROOT = Path(__file__).resolve().parents[3]
# Mirror app.py's DATA_ROOT so a live session can hand load() a bare model-id.
# ponytail: a 2-line dup beats importing the whole FastAPI app into the engine.
_DATA_ROOT = Path(os.getenv("HALLO4_STUDIO_DATA", REPO_ROOT / "studio_data")).resolve()
_VOICE_MODELS_DIR = _DATA_ROOT / "voice_models"
_DEFAULT_WAV2VEC = REPO_ROOT / "pretrained_models" / "wav2vec2-base-960h"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class RVCEngine:
    """Per-chunk voice converter; content encoder REAL, synthesizer STUBBED.

    Robust by design: ``convert_chunk`` never raises and always returns int16 of
    the input length, even with no model loaded, tiny chunks, or a missing
    content encoder — the live session must not die on a bad frame.
    """

    def __init__(self, device: str = "cuda") -> None:
        self.device = device
        self._encoder = None              # transformers Wav2Vec2Model (lazy, shared)
        self._encoder_failed = False
        self._model_meta: Optional[dict] = None   # parsed model.json of the loaded voice
        self._gain = 1.0
        self._ready = False

    @property
    def ready(self) -> bool:
        """True once an enrolled voice model artifact is loaded.

        NB: a loaded *stub* model is "ready" — the session is wired and audio
        flows; conversion is passthrough until the real synthesizer lands.
        """
        return self._ready

    # -- content encoder (REAL) --------------------------------------------- #
    def _resolve_device(self) -> str:
        dev = self.device
        if dev.startswith("cuda"):
            try:
                import torch

                if not torch.cuda.is_available():
                    logger.warning("CUDA unavailable for RVC; using CPU.")
                    dev = "cpu"
            except Exception:  # noqa: BLE001
                dev = "cpu"
        self.device = dev
        return dev

    def _ensure_encoder(self):
        """Lazily load the shared wav2vec2 content encoder (best-effort).

        On any failure the engine degrades to pure passthrough rather than
        breaking the audio path (set HALLO4_RVC_ENCODER=0 to skip it entirely,
        e.g. for a CPU-only integration smoke).
        """
        if self._encoder is not None or self._encoder_failed:
            return self._encoder
        if os.getenv("HALLO4_RVC_ENCODER", "1") == "0":
            self._encoder_failed = True
            return None
        try:
            import torch
            from transformers import Wav2Vec2Model

            path = os.getenv("HALLO4_RVC_WAV2VEC", str(_DEFAULT_WAV2VEC))
            dev = self._resolve_device()
            logger.info("Loading RVC content encoder (wav2vec2) from %s on %s ...", path, dev)
            model = Wav2Vec2Model.from_pretrained(path, local_files_only=True)
            self._encoder = model.to(dev).eval()
            self._torch = torch
            logger.info("RVC content encoder ready on %s.", dev)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "RVC content encoder unavailable (%s: %s); convert_chunk will pass audio through.",
                type(exc).__name__, exc,
            )
            self._encoder_failed = True
            self._encoder = None
        return self._encoder

    def _encode(self, pcm16: np.ndarray) -> None:
        """Run the REAL content encoder for cost-realism. Output is discarded by
        the stub synthesizer (see TODO(rvc) below)."""
        enc = self._ensure_encoder()
        if enc is None or pcm16.size == 0:
            return
        try:
            torch = self._torch
            x = pcm16.astype(np.float32) / 32768.0
            if x.shape[0] < _MIN_ENCODER_SAMPLES:
                x = np.pad(x, (0, _MIN_ENCODER_SAMPLES - x.shape[0]))
            with torch.no_grad():
                feats = enc(input_values=torch.from_numpy(x)[None].to(self.device)).last_hidden_state
            # TODO(rvc): feed `feats` (+ RMVPE f0 + the enrolled speaker model)
            # into a trained net_g synthesizer to produce the converted waveform.
            # STUB: features computed (real cost) then dropped; convert_chunk
            # returns the gain-adjusted input instead.
            _ = feats
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "RVC encoder forward failed (%s: %s); passing audio through.",
                type(exc).__name__, exc,
            )
            self._encoder_failed = True

    # -- voice model (STUB artifact) ---------------------------------------- #
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

    def load(self, model_path: str) -> None:
        """Load an enrolled voice model (a stub artifact today).

        Tolerant by design: an unresolved/broken model logs a warning and leaves
        the engine in passthrough (ready=False) rather than raising and killing
        the session warmup.
        """
        meta_path = self._resolve_model_json(model_path)
        if meta_path is None:
            logger.warning("RVC voice model not found for %r; staying in passthrough.", model_path)
            self._model_meta = None
            self._ready = False
            return
        try:
            self._model_meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to read RVC model.json %s (%s); passthrough.", meta_path, exc)
            self._model_meta = None
            self._ready = False
            return
        self._gain = float(self._model_meta.get("gain", 1.0) or 1.0)
        # Warm the shared content encoder so the first session frame isn't slow.
        self._ensure_encoder()
        self._ready = True
        logger.info(
            "RVC voice model loaded from %s (stub=%s, gain=%.3f).",
            meta_path.parent, self._model_meta.get("stub", True), self._gain,
        )

    # -- realtime conversion ------------------------------------------------ #
    def convert_chunk(self, pcm16_16k: np.ndarray) -> np.ndarray:
        """Convert one realtime chunk. int16 mono 16k in -> int16 same length out.

        Runs the real content encoder (cost-realism), then the stub synthesizer
        (identity + optional per-voice gain). Never raises.
        """
        pcm = np.ascontiguousarray(np.asarray(pcm16_16k)).reshape(-1)
        n = pcm.shape[0]
        if n == 0:
            return np.zeros(0, dtype=np.int16)
        if pcm.dtype != np.int16:
            pcm = np.clip(np.rint(pcm.astype(np.float64)), -32768, 32767).astype(np.int16)

        self._encode(pcm)  # REAL encoder; output consumed by the (TODO) synthesizer

        # STUB synthesizer: identity passthrough + optional per-voice gain knob.
        if self._gain != 1.0:
            return np.clip(pcm.astype(np.float32) * self._gain, -32768, 32767).astype(np.int16)
        return pcm


class RVCTrainer:
    """Offline voice 'enrollment'. STUB: validates the clip and writes a metadata
    artifact; real per-voice net_g training is a documented TODO."""

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
            log("Note: clip is short; real RVC training wants minutes of clean speech.")

        out = Path(out_model_path)
        out.mkdir(parents=True, exist_ok=True)
        ref = out / "reference.wav"
        if src.resolve() != ref.resolve():
            shutil.copyfile(src, ref)  # self-contained artifact (caller guarantees readable audio)
            log(f"Stored reference clip -> {ref.name}")

        # TODO(rvc): real enrollment trains a per-voice net_g and caches a
        # HuBERT/RMVPE feature index from this audio on the GPU (grab the app's
        # gpu_lock). STUB: validate + persist metadata only.
        meta = {
            "version": 1,
            "stub": True,
            "name": None,  # the enrollment job owns the human name (kept in its registry)
            "source_audio": str(src.resolve()),
            "reference": ref.name,
            "duration_sec": round(duration, 3),
            "sample_rate": int(info.samplerate),
            "channels": int(info.channels),
            "frames": int(info.frames),
            "encoder": "wav2vec2-base-960h",
            "synthesizer": "stub-passthrough",
            "gain": 1.0,
            "created_at": _now_iso(),
            "todo": (
                "Real RVC training not implemented: fairseq-HuBERT + a trained net_g "
                "+ RMVPE f0 are install-fragile on aarch64/py3.12 and would clobber "
                "the GPU onnxruntime / numpy pin."
            ),
        }
        (out / "model.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        log("Enrollment complete (stub model written).")
        return str(out)


def demo() -> None:
    """Self-check: enroll the repo's sample clip, load it, convert a chunk."""
    import tempfile

    src = REPO_ROOT / "assets" / "01.wav"
    assert src.is_file(), f"missing test asset {src}"

    with tempfile.TemporaryDirectory() as d:
        model_dir = Path(d) / "voice_model"
        out = RVCTrainer.enroll(str(src), str(model_dir))
        meta = json.loads((Path(out) / "model.json").read_text(encoding="utf-8"))
        assert meta["stub"] is True and meta["sample_rate"] > 0 and meta["duration_sec"] > 0, meta
        assert (Path(out) / "reference.wav").is_file()

        eng = RVCEngine(device=os.getenv("HALLO4_RVC_DEVICE", "cuda"))
        eng.load(out)
        assert eng.ready, "engine should be ready after loading a model"

        out_pcm = eng.convert_chunk(np.zeros(5120, dtype=np.int16))
        assert out_pcm.dtype == np.int16, out_pcm.dtype
        assert out_pcm.shape[0] == 5120, out_pcm.shape

        # robustness: tiny chunk, empty chunk, wrong dtype all honour the contract
        assert eng.convert_chunk(np.zeros(1, np.int16)).shape[0] == 1
        assert eng.convert_chunk(np.zeros(0, np.int16)).shape[0] == 0
        assert eng.convert_chunk(np.zeros(160, np.float32)).dtype == np.int16

    print(f"RVC_OK encoder={'on' if eng._encoder is not None else 'off(passthrough)'}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    demo()
