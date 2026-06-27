"""Phase 2 live-mirror video engine — LivePortrait reenactment + Wav2Lip lip-sync.

Single-pass, per-frame face reenactment for the live mirror: a source portrait is
encoded ONCE (warmup), then each driving webcam frame drives it (drive) and the
result is composited back into the driving frame's geometry, with an optional
Wav2Lip mouth refinement driven by the (converted) audio.

Pipeline (all fp16 except where noted):
    warmup:  insightface crop target -> 256² -> F (appearance feature volume)
                                              -> M (source implicit keypoints)
    drive:   insightface crop driver (every HALLO4_LIVE_DET_EVERY frames, cached
                                      box reused between detections) -> 256²
             M (driving keypoints; torch eager OR onnxruntime-TensorRT)
             relative motion: kp = kp_source + (kp_driving - kp_driving_initial)
             W split:  dense_motion_network (torch eager; see note) -> deformation +
                       occlusion; then the eager tail exactly as warping_network.py
                       (deform_input 5D grid_sample -> third -> fourth -> *occlusion)
             G (SPADE decoder -> 512², torch eager OR onnxruntime-TensorRT)
             paste 512² face back into the driving frame
             Wav2Lip mouth refine on the lower-face crop, driven by an audio mel
             return BGR uint8

Thor / sm110 findings honored:
  * torch.compile is dead on Thor — never used. Speedup path is the onnxruntime
    TensorRT EP only. Audio IO uses librosa/soundfile, never torchaudio.
  * SPADE on ORT-TRT: 31 -> ~10 ms (3x), builds a native *_sm110.engine. Big win.
  * Motion extractor M on ORT-TRT: small net, best-effort.
  * dense_motion on ORT-TRT: MEASURED REGRESSION on Thor. Its conv-heavy hourglass
    exports only at opset>=17... actually 20 (opset 17 rejects the 5D volumetric
    grid_sample inside create_deformed_feature). It runs (TRT EP active) but TRT
    can't take the 5D GridSample node (nbDims==4 only) so ORT partitions it to the
    CUDA EP, and steady-state is ~32.7 ms vs ~21.2 ms eager fp16. So the warp split
    keeps dense_motion EAGER by default; the TRT builder exists behind
    HALLO4_WARP_TRT=1 for hardware where 5D GridSample is TRT-native. The eager tail
    (deform_input/third/fourth/occ) is ~0.3 ms — irrelevant either way.

Interface contract (consumed by live_session.py):
    LiveVideoEngine(use_trt=False, device="cuda")
    .warmup(target_image_path: str) -> None
    .drive(driving_bgr: np.ndarray, audio_pcm16_16k) -> np.ndarray   # BGR uint8 HxWx3
    .ready -> bool

Self-test: ``conda run --no-capture-output -n hallo4-thor python \
    studio/backend/hallo4_studio/live_video_engine.py``  (HALLO4_TEST_TRT=1 for TRT).
"""
from __future__ import annotations

import logging
import os
import os.path as osp
import sys
import threading
import time
from copy import deepcopy
from typing import Any, Optional

import cv2
import numpy as np

log = logging.getLogger("hallo4.live_video_engine")
if not log.handlers:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

_SIBLING_LP = "/home/kp/thordrive2/repos/LivePortrait"
_HERE = osp.dirname(osp.abspath(__file__))
if _HERE not in sys.path:                 # make the vendored Wav2Lip importable as `vendor.*`
    sys.path.insert(0, _HERE)             # works for both __main__ and package import


def _liveportrait_dir() -> str:
    return os.environ.get("HALLO4_LIVEPORTRAIT_DIR", _SIBLING_LP)


def _ensure_lp_on_path() -> str:
    """Make the sibling LivePortrait repo importable; return its path."""
    lp = _liveportrait_dir()
    if lp not in sys.path:
        sys.path.insert(0, lp)
    return lp


def _studio_data_dir() -> str:
    d = os.environ.get("HALLO4_STUDIO_DATA") or osp.join(
        osp.dirname(osp.abspath(__file__)), "..", "..", "studio_data")
    d = osp.join(osp.abspath(d), "live_trt")
    os.makedirs(d, exist_ok=True)
    return d


# ---------------------------------------------------------------------------- #
# Wav2Lip mel — replicates Wav2Lip/audio.py melspectrogram() with their hparams #
# (16kHz, num_mels 80, n_fft 800, hop 200, win 800, fmin 55, fmax 7600, preemph #
# 0.97, ref_level_db 20, symmetric norm max_abs 4). librosa, NOT torchaudio.    #
# ---------------------------------------------------------------------------- #
_MEL_BASIS: Optional[np.ndarray] = None


def _mel_basis_80() -> np.ndarray:
    global _MEL_BASIS
    if _MEL_BASIS is None:
        import librosa
        # keyword args: librosa>=0.10 made mel() keyword-only (Wav2Lip's positional call breaks).
        _MEL_BASIS = librosa.filters.mel(
            sr=16000, n_fft=800, n_mels=80, fmin=55, fmax=7600).astype(np.float32)
    return _MEL_BASIS


def _wav2lip_mel(wav_float: np.ndarray) -> np.ndarray:
    """16k float wav -> (80, T) float32 mel, matching Wav2Lip/audio.py melspectrogram()."""
    import librosa
    from scipy import signal
    wav = signal.lfilter([1, -0.97], [1], wav_float)                  # preemphasis
    D = np.abs(librosa.stft(y=wav, n_fft=800, hop_length=200, win_length=800)).astype(np.float32)
    S = np.dot(_mel_basis_80(), D)                                    # linear -> mel
    min_level = np.exp(-100.0 / 20.0 * np.log(10))                    # min_level_db = -100
    S = 20.0 * np.log10(np.maximum(min_level, S)) - 20.0             # _amp_to_db - ref_level_db
    S = np.clip(8.0 * ((S + 100.0) / 100.0) - 4.0, -4.0, 4.0)        # symmetric normalize
    return S.astype(np.float32)


class LiveVideoEngine:
    """LivePortrait per-frame reenactment + Wav2Lip lip-sync. Eager fp16 by default;
    SPADE/motion via onnxruntime-TensorRT when ``use_trt=True`` (best-effort, eager fallback)."""

    # ponytail: bbox expansion factor in lieu of LivePortrait's full landmark affine
    # cropper. Tune if the crop clips foreheads/chins. Upgrade path: src/utils/cropper.py.
    CROP_SCALE = 1.6
    _AUDIO_BUF_MAX = 16000        # keep the last ~1 s of 16k PCM in the ring buffer
    _AUDIO_MIN = 3200             # need >= ~0.2 s before a mel window is meaningful

    def __init__(self, use_trt: bool = False, device: str = "cuda") -> None:
        self.use_trt = use_trt
        self.device = device
        self._ready = False
        self.real_weights = False        # True iff all 4 base checkpoints loaded
        self.trt_active = False          # True iff SPADE is running on ORT-TRT

        # Lazily initialised in warmup() (heavy imports/model loads).
        self._lp = _ensure_lp_on_path()
        self._torch = None
        self._face = None                # insightface FaceAnalysis
        self._F = self._M = self._W = self._G = None
        self._cam = None                 # camera helpers module
        self._spade_sess = None          # ORT session (TRT) or None
        self._motion_sess = None         # ORT session for M, or None (eager)
        self._motion_keys = None
        self._dm_sess = None             # ORT session for dense_motion, or None (eager)

        # Per-session source state (set in warmup).
        self._feature_3d = None          # (1,32,16,64,64) fp16
        self._kp_source = None           # (1,21,3) fp32
        # Relative-motion anchor (set on first driven frame).
        self._kp_driving_initial = None
        self._mask512 = None             # cached feather alpha
        self._drive_failed_logged = False

        # Detection skipping (insightface ~16-26 ms/frame): detect every N frames,
        # reuse the cached box (and re-crop the *current* frame) in between.
        self._det_every = max(1, int(os.environ.get("HALLO4_LIVE_DET_EVERY", "3")))
        self._frame_idx = 0
        self._cached_box = None

        # Wav2Lip lip-sync.
        repo_root = osp.abspath(osp.join(_HERE, "..", "..", ".."))
        self._wav2lip_ckpt = os.environ.get(
            "HALLO4_WAV2LIP_CKPT",
            osp.join(repo_root, "pretrained_models", "wav2lip", "checkpoints", "wav2lip_gan.pth"))
        self._wav2lip_enabled = os.environ.get(
            "HALLO4_WAV2LIP", "1" if osp.exists(self._wav2lip_ckpt) else "0") == "1"
        self._wav2lip = None             # lazily loaded torch net (fp16) or None
        self._wav2lip_sess = None        # optional ORT-TRT session
        self._wav2lip_load_failed = False
        self._lip_failed_logged = False
        self._audio_buf = np.zeros(0, np.float32)
        self._mouth_alpha_cache = (None, None)

        # Optional per-stage profiler (set to a dict-of-lists by demo()).
        self._prof = None

    # ------------------------------------------------------------------ #
    # Lazy model construction + weight loading                           #
    # ------------------------------------------------------------------ #
    def _build_nets(self) -> None:
        import torch
        import yaml
        from src.modules.appearance_feature_extractor import AppearanceFeatureExtractor
        from src.modules.motion_extractor import MotionExtractor
        from src.modules.warping_network import WarpingNetwork
        from src.modules.spade_generator import SPADEDecoder
        from src.utils import camera

        self._torch = torch
        self._cam = camera
        cfg = yaml.safe_load(open(osp.join(self._lp, "src/config/models.yaml")))["model_params"]
        dev = self.device

        self._F = AppearanceFeatureExtractor(**cfg["appearance_feature_extractor_params"]).to(dev).eval()
        self._M = MotionExtractor(**cfg["motion_extractor_params"]).to(dev).eval()
        self._W = WarpingNetwork(**cfg["warping_module_params"]).to(dev).eval()
        self._G = SPADEDecoder(**cfg["spade_generator_params"]).to(dev).eval()

        self.real_weights = self._load_weights()

        # fp16 for inference (matches bench_eager.py; ~16 fps on Thor).
        for m in (self._F, self._M, self._W, self._G):
            m.half()

    def _ckpt_paths(self) -> dict[str, str]:
        base = osp.join(self._lp, "pretrained_weights", "liveportrait", "base_models")
        return {
            "F": osp.join(base, "appearance_feature_extractor.pth"),
            "M": osp.join(base, "motion_extractor.pth"),
            "W": osp.join(base, "warping_module.pth"),
            "G": osp.join(base, "spade_generator.pth"),
        }

    def _load_weights(self) -> bool:
        """Load real LivePortrait weights, downloading from HF if missing. Returns
        True if all four base nets got real weights, else leaves random init + warns."""
        paths = self._ckpt_paths()
        if not all(osp.exists(p) for p in paths.values()):
            self._download_weights()
        if not all(osp.exists(p) for p in paths.values()):
            log.warning("LivePortrait weights NOT found at %s — running with RANDOM "
                        "weights. Pipeline runs but the output is NOT a faithful likeness. "
                        "Set HALLO4_LIVEPORTRAIT_DIR or allow the HF download to fix this.",
                        osp.dirname(paths["F"]))
            return False
        torch = self._torch
        try:
            for key, net in (("F", self._F), ("M", self._M), ("W", self._W), ("G", self._G)):
                # weights_only=False: trusted local checkpoints (some hold non-tensor scalars).
                sd = torch.load(paths[key], map_location="cpu", weights_only=False)
                net.load_state_dict(sd)
            log.info("Loaded real LivePortrait weights from %s", osp.dirname(paths["F"]))
            return True
        except Exception as exc:  # noqa: BLE001
            log.warning("Failed to load LivePortrait weights (%s) — falling back to RANDOM.", exc)
            return False

    def _download_weights(self) -> None:
        """snapshot_download the human base+retargeting models into <LP>/pretrained_weights.
        Runs in a thread with a timeout so a slow/absent network can't hang warmup."""
        timeout = float(os.environ.get("HALLO4_HF_TIMEOUT", "600"))
        dest = osp.join(self._lp, "pretrained_weights")

        def _dl() -> None:
            try:
                from huggingface_hub import snapshot_download
                log.info("Downloading KwaiVGI/LivePortrait base weights -> %s (timeout %ss)...",
                         dest, timeout)
                snapshot_download(
                    repo_id="KwaiVGI/LivePortrait",
                    local_dir=dest,
                    allow_patterns=["liveportrait/base_models/*.pth",
                                    "liveportrait/retargeting_models/*.pth"],
                )
                log.info("LivePortrait weight download complete.")
            except Exception as exc:  # noqa: BLE001
                log.warning("HF weight download failed: %s", exc)

        t = threading.Thread(target=_dl, daemon=True)
        t.start()
        t.join(timeout)
        if t.is_alive():
            log.warning("HF weight download exceeded %ss — continuing without it.", timeout)

    def _build_face(self) -> None:
        from insightface.app import FaceAnalysis
        # GPU detector; --no-deps insightface reuses the GPU onnxruntime build (2a finding).
        # detection-only: the full buffalo_l pack (recognition/genderage/landmarks) would
        # run all 5 onnx models per frame (~150ms); we only need the bbox.
        self._face = FaceAnalysis(name="buffalo_l", allowed_modules=["detection"],
                                  providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        self._face.prepare(ctx_id=0, det_size=(640, 640))

    # ------------------------------------------------------------------ #
    # Crop / keypoint helpers                                            #
    # ------------------------------------------------------------------ #
    def _crop256(self, bgr: np.ndarray, box) -> Optional[np.ndarray]:
        """Crop bgr at box and return a 256² RGB array (or None if the box is empty)."""
        x0, y0, x1, y1 = box
        crop = bgr[y0:y1, x0:x1]
        if crop.size == 0:
            return None
        return cv2.cvtColor(cv2.resize(crop, (256, 256)), cv2.COLOR_BGR2RGB)

    def _detect_box(self, bgr: np.ndarray):
        """Detect the largest face; return box=(x0,y0,x1,y1) or None. (the ~16-26 ms part)"""
        faces = self._face.get(bgr)
        if not faces:
            return None
        f = max(faces, key=lambda a: (a.bbox[2] - a.bbox[0]) * (a.bbox[3] - a.bbox[1]))
        x1, y1, x2, y2 = f.bbox
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        half = max(x2 - x1, y2 - y1) * self.CROP_SCALE / 2.0
        h, w = bgr.shape[:2]
        x0, y0 = int(max(0, cx - half)), int(max(0, cy - half))
        x1b, y1b = int(min(w, cx + half)), int(min(h, cy + half))
        if x1b - x0 < 16 or y1b - y0 < 16:
            return None
        return (x0, y0, x1b, y1b)

    def _detect_crop(self, bgr: np.ndarray):
        """Detect the largest face; return (rgb256, box) or (None, None). (full detect)"""
        box = self._detect_box(bgr)
        if box is None:
            return None, None
        return self._crop256(bgr, box), box

    def _prep(self, rgb256: np.ndarray):
        torch = self._torch
        x = rgb256.astype(np.float32) / 255.0
        x = torch.from_numpy(x).permute(2, 0, 1)[None].to(self.device).half()
        return x

    def _kp_transform(self, kp_info: dict):
        """Replicate LivePortraitWrapper.get_kp_info(refine)+transform_keypoint in fp32.
        Returns implicit 3D keypoints (1,21,3)."""
        torch, cam = self._torch, self._cam
        kp = kp_info["kp"].float()
        bs = kp.shape[0]
        kp = kp.reshape(bs, -1, 3)
        exp = kp_info["exp"].float().reshape(bs, -1, 3)
        t = kp_info["t"].float()
        scale = kp_info["scale"].float()
        pitch = cam.headpose_pred_to_degree(kp_info["pitch"].float())
        yaw = cam.headpose_pred_to_degree(kp_info["yaw"].float())
        roll = cam.headpose_pred_to_degree(kp_info["roll"].float())
        rot = cam.get_rotation_matrix(pitch, yaw, roll)          # (bs,3,3)
        kp_t = kp @ rot + exp                                    # Eqn.2: s*(R*x_c + exp)+t
        kp_t = kp_t * scale[..., None]
        kp_t[:, :, 0:2] += t[:, None, 0:2]
        return kp_t

    def _kp_from_crop(self, rgb256: np.ndarray):
        torch = self._torch
        if self._motion_sess is not None:                        # M on ORT-TRT
            x = self._prep(rgb256).float().cpu().numpy()
            outs = self._motion_sess.run(None, {"image": x})
            kp_info = {k: torch.from_numpy(v).to(self.device) for k, v in zip(self._motion_keys, outs)}
        else:
            with torch.no_grad():
                kp_info = self._M(self._prep(rgb256))
        return self._kp_transform(kp_info)

    # ------------------------------------------------------------------ #
    # Generic ORT-TensorRT session builder (used by warp/motion/wav2lip) #
    # ------------------------------------------------------------------ #
    def _make_trt_sess(self, tag: str, make_fp32, example_inputs: dict,
                       output_names, opsets=(17, 20)):
        """Export ``make_fp32()`` to ONNX (first opset that works) and wrap it in an
        onnxruntime TensorRT-EP session. Engine cache persists under studio_data.
        ``example_inputs``: ordered dict name->np.float32 array (also used to warm up).
        Returns (session, active_provider) or (None, None) on any failure (caller -> eager)."""
        torch = self._torch
        try:
            import onnxruntime as ort
        except Exception as exc:  # noqa: BLE001
            log.warning("onnxruntime import failed (%s) — %s stays eager.", exc, tag)
            return None, None
        data_dir = _studio_data_dir()
        onnx_path = osp.join(data_dir, f"{tag}.onnx")
        cache_dir = osp.join(data_dir, "trt_cache")
        os.makedirs(cache_dir, exist_ok=True)
        try:
            if not osp.exists(onnx_path):
                model = make_fp32()
                args = tuple(torch.from_numpy(v).to(self.device) for v in example_inputs.values())
                last_err = RuntimeError("no opset tried")
                for op in opsets:
                    try:
                        torch.onnx.export(model, args, onnx_path,
                                          input_names=list(example_inputs.keys()),
                                          output_names=list(output_names),
                                          opset_version=op, dynamo=False)
                        log.info("%s: ONNX export ok (opset %d)", tag, op)
                        last_err = None
                        break
                    except Exception as exc:  # noqa: BLE001
                        last_err = exc
                        if osp.exists(onnx_path):
                            os.remove(onnx_path)
                del model
                if last_err is not None:
                    raise last_err
            providers = [("TensorrtExecutionProvider",
                          {"trt_fp16_enable": True, "trt_engine_cache_enable": True,
                           "trt_engine_cache_path": cache_dir}),
                         "CUDAExecutionProvider"]
            sess = ort.InferenceSession(onnx_path, providers=providers)
            for _ in range(3):                                   # builds/loads the engine
                sess.run(None, example_inputs)
            active = sess.get_providers()[0]
            log.info("%s on onnxruntime (active provider=%s, cache=%s)", tag, active, cache_dir)
            return sess, active
        except Exception as exc:  # noqa: BLE001
            log.warning("%s TensorRT setup failed (%s) — eager fallback.", tag, repr(exc)[:200])
            return None, None

    def _build_trt_spade(self) -> None:
        """Best-effort: export the (real-weight) SPADE decoder to ONNX and run it via
        the onnxruntime TensorRT EP. Engine cache persists under studio_data."""
        torch = self._torch
        try:
            import onnxruntime as ort
            data_dir = _studio_data_dir()
            onnx_path = osp.join(data_dir, "spade_real.onnx")
            cache_dir = osp.join(data_dir, "trt_cache")
            os.makedirs(cache_dir, exist_ok=True)
            if not osp.exists(onnx_path):
                g32 = deepcopy(self._G).float().eval()           # fp32 graph for export
                x = torch.randn(1, 256, 64, 64, device=self.device)
                torch.onnx.export(g32, x, onnx_path, input_names=["x"], output_names=["y"],
                                  opset_version=17, dynamo=False)  # dynamo needs onnxscript
                del g32
            providers = [("TensorrtExecutionProvider",
                          {"trt_fp16_enable": True, "trt_engine_cache_enable": True,
                           "trt_engine_cache_path": cache_dir}),
                         "CUDAExecutionProvider"]
            sess = ort.InferenceSession(onnx_path, providers=providers)
            warm = np.random.randn(1, 256, 64, 64).astype(np.float32)
            for _ in range(3):                                   # builds/loads the engine
                sess.run(None, {"x": warm})
            self._spade_sess = sess
            self.trt_active = sess.get_providers()[0] == "TensorrtExecutionProvider"
            log.info("SPADE on onnxruntime-TRT (active provider=%s, cache=%s)",
                     sess.get_providers()[0], cache_dir)
        except Exception as exc:  # noqa: BLE001
            log.warning("SPADE TensorRT setup failed (%s) — using eager SPADE.", exc)
            self._spade_sess = None
            self.trt_active = False

    def _build_trt_motion(self) -> None:
        """Best-effort TRT for the motion extractor M (convnextv2_tiny; ~4 ms eager).
        Wrap to return its dict's tensors as a tuple, rebuild the dict in _kp_from_crop."""
        torch = self._torch
        keys = ["pitch", "yaw", "roll", "t", "exp", "scale", "kp"]

        class _MWrap(torch.nn.Module):
            def __init__(self, m): super().__init__(); self.m = m
            def forward(self, image):
                d = self.m(image)
                return tuple(d[k] for k in keys)

        def make():
            return _MWrap(deepcopy(self._M).float().eval()).to(self.device).eval()

        ex = {"image": np.random.randn(1, 3, 256, 256).astype(np.float32)}
        self._motion_sess, _ = self._make_trt_sess("motion_real", make, ex, keys, opsets=(17,))
        self._motion_keys = keys

    def _build_trt_warp(self) -> None:
        """Best-effort TRT for dense_motion_network (the conv-heavy part of W). Returns
        (deformation, occlusion_map); the eager tail (deform_input/third/fourth/occ) stays
        torch. NOTE: opset 17 can't export the 5D grid_sample inside create_deformed_feature
        (-> opset 20), and TRT can't take 5D GridSample so ORT runs it on the CUDA EP; on Thor
        this MEASURED ~32.7 ms vs ~21.2 ms eager — a regression. Default OFF; enable only on
        hardware with native 5D GridSample via HALLO4_WARP_TRT=1."""
        torch = self._torch

        class _DMWrap(torch.nn.Module):
            def __init__(self, dmn): super().__init__(); self.dmn = dmn
            def forward(self, feature, kp_driving, kp_source):
                d = self.dmn(feature=feature, kp_driving=kp_driving, kp_source=kp_source)
                occ = d.get("occlusion_map")
                if occ is None:                                  # keep ONNX outputs tensor-typed
                    occ = torch.ones(feature.shape[0], 1, feature.shape[3], feature.shape[4],
                                     device=feature.device, dtype=feature.dtype)
                return d["deformation"], occ

        def make():
            return _DMWrap(deepcopy(self._W.dense_motion_network).float().eval()).to(self.device).eval()

        ex = {
            "feature": np.random.randn(1, 32, 16, 64, 64).astype(np.float32),
            "kp_driving": np.random.randn(1, 21, 3).astype(np.float32),
            "kp_source": np.random.randn(1, 21, 3).astype(np.float32),
        }
        self._dm_sess, _ = self._make_trt_sess(
            "dense_motion_real", make, ex, ["deformation", "occlusion_map"], opsets=(17, 20))

    # ------------------------------------------------------------------ #
    # Warp (split) + SPADE                                               #
    # ------------------------------------------------------------------ #
    def _warp(self, kp):
        """Faithful WarpingNetwork.forward: dense_motion (TRT if built else eager) ->
        eager 5D-grid_sample tail. Returns (1,256,64,64) fp16 for SPADE."""
        torch = self._torch
        W = self._W
        if self._dm_sess is not None:                            # dense_motion on ORT-TRT
            outs = self._dm_sess.run(None, {
                "feature": self._feature_3d.float().cpu().numpy(),
                "kp_driving": kp.float().cpu().numpy(),
                "kp_source": self._kp_source.float().cpu().numpy()})
            deformation = torch.from_numpy(outs[0]).to(self.device).half()
            occlusion_map = torch.from_numpy(outs[1]).to(self.device).half()
        else:                                                    # eager fp16 (default)
            dm = W.dense_motion_network(feature=self._feature_3d,
                                        kp_driving=kp.half(), kp_source=self._kp_source.half())
            deformation = dm["deformation"]
            occlusion_map = dm.get("occlusion_map")
        # Eager tail, exactly as src/modules/warping_network.py forward():
        out = W.deform_input(self._feature_3d, deformation)      # Bx32x16x64x64 (5D grid_sample)
        bs, c, d, h, w = out.shape
        out = out.view(bs, c * d, h, w)                          # -> Bx512x64x64
        out = W.third(out)                                       # -> Bx256x64x64
        out = W.fourth(out)                                      # -> Bx256x64x64
        if W.flag_use_occlusion_map and (occlusion_map is not None):
            out = out * occlusion_map
        return out

    def _spade(self, warp_out):
        """warp_out: (1,256,64,64) torch fp16 -> 512² RGB uint8 (HxWx3)."""
        torch = self._torch
        if self._spade_sess is not None:
            y = self._spade_sess.run(None, {"x": warp_out.float().cpu().numpy()})[0]  # (1,3,512,512)
            img = np.clip(np.transpose(y[0], (1, 2, 0)) * 255.0, 0, 255).astype(np.uint8)
            return img
        # Eager: do clamp/scale/permute on the GPU, transfer only the 786KB uint8 frame.
        img = self._G(warp_out)[0].clamp(0, 1).mul(255).permute(1, 2, 0).to(torch.uint8)
        return img.cpu().numpy()

    def _feather(self, bw: int, bh: int) -> np.ndarray:
        """Soft elliptical alpha (bh,bw,1) float32. Blurred ONCE at 512² then resized —
        per-frame GaussianBlur with a huge sigma was ~30ms."""
        if self._mask512 is None:
            m = np.zeros((512, 512), np.uint8)
            cv2.ellipse(m, (256, 256), (236, 236), 0, 0, 360, 255, -1)
            self._mask512 = cv2.GaussianBlur(m, (0, 0), 30).astype(np.float32) / 255.0
        return cv2.resize(self._mask512, (bw, bh))[..., None]

    def _paste(self, frame_bgr: np.ndarray, rgb512: np.ndarray, box) -> np.ndarray:
        """Composite the generated face (RGB) into the frame at box with a soft ellipse."""
        x0, y0, x1, y1 = box
        bw, bh = x1 - x0, y1 - y0
        face = cv2.cvtColor(cv2.resize(rgb512, (bw, bh)), cv2.COLOR_RGB2BGR)
        alpha = self._feather(bw, bh)        # ponytail: ellipse feather; TODO real face mask
        out = frame_bgr.copy()
        roi = out[y0:y1, x0:x1].astype(np.float32)
        out[y0:y1, x0:x1] = (alpha * face + (1 - alpha) * roi).astype(np.uint8)
        return out

    # ------------------------------------------------------------------ #
    # Wav2Lip mouth refinement                                           #
    # ------------------------------------------------------------------ #
    def _push_audio(self, audio: Any) -> None:
        """Append a frame's worth of 16k PCM to the ring buffer (cap ~1 s)."""
        if audio is None:
            return
        a = audio
        if isinstance(a, (bytes, bytearray, memoryview)):
            a = np.frombuffer(bytes(a), dtype=np.int16)
        a = np.asarray(a).reshape(-1)
        if a.size == 0:
            return
        if a.dtype == np.int16:
            a = a.astype(np.float32) / 32768.0
        else:
            a = a.astype(np.float32)
        self._audio_buf = np.concatenate([self._audio_buf, a])[-self._AUDIO_BUF_MAX:]

    def _mel_now(self) -> Optional[np.ndarray]:
        """The ~16-wide mel window aligned to 'now', or None if not enough audio yet."""
        if self._audio_buf.size < self._AUDIO_MIN:
            return None
        mel = _wav2lip_mel(self._audio_buf)                      # (80, T)
        if mel.shape[1] < 16:
            return None
        return mel[:, -16:]                                      # (80,16)

    def _load_wav2lip(self) -> None:
        """Lazily load the vendored Wav2Lip generator (checkpoint is {'state_dict': {...}}
        with 'module.' prefixes; strip them; strict=False). Optional ORT-TRT."""
        torch = self._torch
        try:
            from vendor.wav2lip import Wav2Lip
            sd = torch.load(self._wav2lip_ckpt, map_location="cpu", weights_only=False)
            sd = sd.get("state_dict", sd)
            sd = {k.replace("module.", ""): v for k, v in sd.items()}
            net = Wav2Lip()
            net.load_state_dict(sd, strict=False)
            self._wav2lip = net.to(self.device).eval().half()
            log.info("Wav2Lip loaded from %s", self._wav2lip_ckpt)
            # ponytail: 36M conv net is ~3-5 ms eager fp16 — TRT (a 4th heavy engine build)
            # is opt-in via HALLO4_WAV2LIP_TRT=1, not worth the warmup cost by default.
            if self.use_trt and os.environ.get("HALLO4_WAV2LIP_TRT", "0") == "1":
                def make():
                    return deepcopy(self._wav2lip).float().eval().to(self.device)
                ex = {"audio": np.random.randn(1, 1, 80, 16).astype(np.float32),
                      "face": np.random.rand(1, 6, 96, 96).astype(np.float32)}
                self._wav2lip_sess, _ = self._make_trt_sess(
                    "wav2lip_real", make, ex, ["out"], opsets=(17, 20))
        except Exception as exc:  # noqa: BLE001
            log.warning("Wav2Lip load failed (%s) — lip refine disabled.", exc)
            self._wav2lip = None

    def _mouth_alpha(self, bw: int, bh: int) -> np.ndarray:
        """Vertical feather (bh,1,1): 0 over the upper face, ramps to ~0.9 over the mouth.
        ponytail: vertical-only ramp; upgrade to a landmark mouth mask if seams show."""
        if self._mouth_alpha_cache[0] != (bw, bh):
            rows = np.arange(bh, dtype=np.float32)
            ramp = np.clip((rows - 0.5 * bh) / (0.18 * bh + 1.0), 0.0, 1.0)
            a = (ramp * 0.9).reshape(bh, 1, 1)
            self._mouth_alpha_cache = ((bw, bh), a)
        return self._mouth_alpha_cache[1]

    def _wav2lip_run(self, mel: np.ndarray, face6: np.ndarray) -> np.ndarray:
        """mel (80,16), face6 (1,6,96,96) fp32 in [0,1] -> (96,96,3) BGR uint8."""
        torch = self._torch
        mel_in = mel[None, None]                                 # (1,1,80,16)
        if self._wav2lip_sess is not None:
            y = self._wav2lip_sess.run(None, {"audio": mel_in.astype(np.float32),
                                              "face": face6.astype(np.float32)})[0]
            return np.clip(np.transpose(y[0], (1, 2, 0)) * 255.0, 0, 255).astype(np.uint8)
        a = torch.from_numpy(mel_in).to(self.device).half()
        f = torch.from_numpy(face6).to(self.device).half()
        with torch.no_grad():
            y = self._wav2lip(a, f)                              # (1,3,96,96) sigmoid, BGR order
        return y[0].clamp(0, 1).mul(255).permute(1, 2, 0).to(torch.uint8).cpu().numpy()

    def _wav2lip_apply(self, frame_bgr: np.ndarray, mel: np.ndarray, box) -> np.ndarray:
        """Run Wav2Lip on the face box and composite the refined mouth back."""
        x0, y0, x1, y1 = box
        crop = frame_bgr[y0:y1, x0:x1]
        if crop.size == 0:
            return frame_bgr
        bh, bw = y1 - y0, x1 - x0
        face = cv2.resize(crop, (96, 96)).astype(np.float32) / 255.0     # BGR (Wav2Lip is BGR)
        masked = face.copy()
        masked[48:] = 0.0                                                # lower half -> 0
        face6 = np.concatenate([masked, face], axis=2).transpose(2, 0, 1)[None]  # (1,6,96,96)
        gen = cv2.resize(self._wav2lip_run(mel, face6), (bw, bh)).astype(np.float32)
        alpha = self._mouth_alpha(bw, bh)                               # (bh,1,1)
        out = frame_bgr.copy()
        roi = out[y0:y1, x0:x1].astype(np.float32)
        out[y0:y1, x0:x1] = (alpha * gen + (1.0 - alpha) * roi).astype(np.uint8)
        return out

    def _lip_refine(self, frame_bgr: np.ndarray, audio_pcm16_16k: Any) -> np.ndarray:
        """Wav2Lip mouth refinement driven by the (converted) audio. Best-effort: any
        failure (or too little audio) returns the frame unchanged. Never crashes a frame."""
        if not self._wav2lip_enabled:
            return frame_bgr
        try:
            self._push_audio(audio_pcm16_16k)
            mel = self._mel_now()
            if mel is None or self._cached_box is None:
                return frame_bgr
            if self._wav2lip is None:
                if self._wav2lip_load_failed:
                    return frame_bgr
                self._load_wav2lip()
                if self._wav2lip is None:
                    self._wav2lip_load_failed = True
                    return frame_bgr
            return self._wav2lip_apply(frame_bgr, mel, self._cached_box)
        except Exception as exc:  # noqa: BLE001
            if not self._lip_failed_logged:
                log.warning("lip refine failed (%s) — passthrough.", exc)
                self._lip_failed_logged = True
            return frame_bgr

    # ------------------------------------------------------------------ #
    # Public API                                                         #
    # ------------------------------------------------------------------ #
    def warmup(self, target_image_path: str) -> None:
        """Encode the source portrait ONCE: appearance feature volume + source keypoints."""
        if self._F is None:
            self._build_nets()
        if self._face is None:
            self._build_face()
        if self.use_trt:
            if self._spade_sess is None:
                self._build_trt_spade()
            if self._motion_sess is None:
                self._build_trt_motion()
            # dense_motion TRT is a measured regression on Thor (5D GridSample -> CUDA EP);
            # opt-in only. See _build_trt_warp docstring.
            if self._dm_sess is None and os.environ.get("HALLO4_WARP_TRT", "0") == "1":
                self._build_trt_warp()

        bgr = cv2.imread(target_image_path)
        if bgr is None:
            raise FileNotFoundError(f"target image not readable: {target_image_path}")
        rgb256, _ = self._detect_crop(bgr)
        if rgb256 is None:
            raise RuntimeError(f"no face detected in target image: {target_image_path}")

        torch = self._torch
        with torch.no_grad():
            src = self._prep(rgb256)
            self._feature_3d = self._F(src)                      # (1,32,16,64,64) fp16
            self._kp_source = self._kp_from_crop(rgb256)         # (1,21,3) fp32
        self._kp_driving_initial = None
        self._frame_idx = 0
        self._cached_box = None
        self._ready = True
        log.info("warmup done (real_weights=%s, trt=%s, motion_trt=%s, warp_trt=%s, wav2lip=%s)",
                 self.real_weights, self.trt_active, self._motion_sess is not None,
                 self._dm_sess is not None, self._wav2lip_enabled)

    def drive(self, driving_bgr: np.ndarray, audio_pcm16_16k: Any = None) -> np.ndarray:
        """Per driving frame -> output BGR uint8 HxWx3. Unknown/failed frames pass through."""
        if not self._ready:
            return driving_bgr
        torch = self._torch
        prof = self._prof

        def _now() -> float:
            if prof is None:
                return 0.0
            torch.cuda.synchronize()
            return time.perf_counter()

        try:
            # --- detect/crop (skip insightface between detections) ---------------
            t0 = _now()
            need_detect = (self._cached_box is None) or (self._frame_idx % self._det_every == 0)
            if need_detect:
                rgb256, box = self._detect_crop(driving_bgr)
                if rgb256 is None:                                # detection missed this frame
                    box = self._cached_box                        # reuse last box if we have one
                    rgb256 = self._crop256(driving_bgr, box) if box is not None else None
                else:
                    self._cached_box = box
            else:
                box = self._cached_box
                rgb256 = self._crop256(driving_bgr, box)
            self._frame_idx += 1
            if rgb256 is None:
                return driving_bgr                                # no face anywhere -> unchanged
            t1 = _now()

            with torch.no_grad():
                kp_driving = self._kp_from_crop(rgb256)
                if self._kp_driving_initial is None:
                    self._kp_driving_initial = kp_driving          # relative-motion anchor
                # Relative motion: keep source identity/pose, apply driver's *changes*.
                # ponytail: additive delta (LivePortrait flag_relative_motion); skipped
                # stitching/retargeting nets — add from retargeting_models if neck drifts.
                kp = self._kp_source + (kp_driving - self._kp_driving_initial)
                t2 = _now()
                warp_out = self._warp(kp)
                t3 = _now()
                rgb512 = self._spade(warp_out)
                t4 = _now()
            out = self._paste(driving_bgr, rgb512, box)
            t5 = _now()
            out = self._lip_refine(out, audio_pcm16_16k)
            t6 = _now()

            if prof is not None:
                prof["detect"].append((t1 - t0) * 1000)
                prof["motion"].append((t2 - t1) * 1000)
                prof["warp"].append((t3 - t2) * 1000)
                prof["spade"].append((t4 - t3) * 1000)
                prof["paste"].append((t5 - t4) * 1000)
                prof["lip"].append((t6 - t5) * 1000)
            return out
        except Exception as exc:  # noqa: BLE001 — a bad frame must never kill the session
            if not self._drive_failed_logged:
                log.warning("drive() failed on a frame (%s) — passing it through.", exc)
                self._drive_failed_logged = True
            return driving_bgr

    @property
    def ready(self) -> bool:
        return self._ready


def demo() -> None:
    """Self-test: warmup on assets/01.png, drive frames, print per-stage ms + fps, and a
    Wav2Lip silent-vs-noise mouth-diff check. HALLO4_TEST_TRT=1 exercises the TRT path."""
    from collections import defaultdict

    repo_root = osp.abspath(osp.join(osp.dirname(osp.abspath(__file__)), "..", "..", ".."))
    asset = osp.join(repo_root, "assets", "01.png")
    use_trt = os.environ.get("HALLO4_TEST_TRT", "0") == "1"

    eng = LiveVideoEngine(use_trt=use_trt)
    eng.warmup(asset)
    assert eng.ready, "engine not ready after warmup"

    driving = cv2.imread(asset)
    assert driving is not None, f"cannot read {asset}"

    for _ in range(5):                                           # warm CUDA + set rel-motion anchor
        eng.drive(driving)
    n = 30
    t0 = time.perf_counter()
    for _ in range(n):
        out = eng.drive(driving)
    dt = (time.perf_counter() - t0) / n

    assert isinstance(out, np.ndarray), "output is not an ndarray"
    assert out.dtype == np.uint8, f"output dtype {out.dtype} != uint8"
    assert out.ndim == 3 and out.shape[2] == 3, f"output shape {out.shape} not HxWx3"

    print(f"output shape={out.shape} dtype={out.dtype}")
    print(f"real_weights={eng.real_weights}  trt_active(SPADE)={eng.trt_active}  "
          f"motion_trt={eng._motion_sess is not None}  warp_trt={eng._dm_sess is not None}")
    print(f"per-frame {dt*1000:.1f} ms -> {1.0/dt:.1f} fps  (det_every={eng._det_every}, "
          f"{'TRT' if eng.trt_active else 'eager'})")

    # ---- per-stage breakdown -------------------------------------------------
    eng._prof = defaultdict(list)
    for _ in range(n):
        eng.drive(driving)
    if eng._prof.get("warp"):
        det = eng._prof["detect"]
        print("per-stage (ms): "
              f"detect mean={np.mean(det):.1f} (peak={np.max(det):.1f}, ~1/{eng._det_every} frames)  "
              f"motion={np.median(eng._prof['motion']):.1f}  "
              f"warp(dense_motion+tail)={np.median(eng._prof['warp']):.1f}  "
              f"spade={np.median(eng._prof['spade']):.1f}  "
              f"paste={np.median(eng._prof['paste']):.1f}  "
              f"lip={np.median(eng._prof['lip']):.1f}")
    eng._prof = None

    # ---- Wav2Lip: non-silent vs silent must change the mouth pixels ----------
    if eng._wav2lip_enabled and eng._cached_box is not None:
        x0, y0, x1, y1 = eng._cached_box
        ly0 = y0 + (y1 - y0) // 2                                # lower-face region only
        eng._audio_buf = np.zeros(0, np.float32)
        out_s = eng.drive(driving, np.zeros(8000, np.int16))    # silence
        eng._audio_buf = np.zeros(0, np.float32)
        rng = np.random.default_rng(0)
        noisy = np.clip(rng.standard_normal(8000) * 6000, -32767, 32767).astype(np.int16)
        out_n = eng.drive(driving, noisy)                       # non-silent
        a = out_s[ly0:y1, x0:x1].astype(np.float32)
        b = out_n[ly0:y1, x0:x1].astype(np.float32)
        d = float(np.mean(np.abs(a - b)))
        cv2.imwrite("/tmp/wav2lip_silent.png", out_s[ly0:y1, x0:x1])
        cv2.imwrite("/tmp/wav2lip_noise.png", out_n[ly0:y1, x0:x1])
        print(f"Wav2Lip status: loaded={eng._wav2lip is not None}  trt={eng._wav2lip_sess is not None}")
        print(f"Wav2Lip mouth-region mean|Δ| silent-vs-noise: {d:.2f} uint8 levels "
              f"(crops dumped to /tmp/wav2lip_{{silent,noise}}.png)")
        assert eng._wav2lip is not None, "Wav2Lip enabled but failed to load"
        assert d > 1.0, f"Wav2Lip did not change mouth pixels (Δ={d:.2f})"
        print("WAV2LIP_OK")
    else:
        print(f"Wav2Lip disabled (enabled={eng._wav2lip_enabled}, ckpt={eng._wav2lip_ckpt})")

    # ---- env sanity (must stay intact) ---------------------------------------
    try:
        import onnxruntime as ort
        print(f"numpy={np.__version__}  ort_providers={ort.get_available_providers()}")
    except Exception as exc:  # noqa: BLE001
        print(f"numpy={np.__version__}  (onnxruntime import failed: {exc})")
    print("LIVE_VIDEO_ENGINE_OK")


if __name__ == "__main__":
    demo()
