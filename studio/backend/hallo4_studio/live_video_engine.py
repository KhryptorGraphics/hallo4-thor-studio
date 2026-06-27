"""Phase 2 live-mirror video engine — LivePortrait reenactment (+ Wav2Lip hook).

Single-pass, per-frame face reenactment for the live mirror: a source portrait is
encoded ONCE (warmup), then each driving webcam frame drives it (drive) and the
result is composited back into the driving frame's geometry.

Pipeline (all fp16 except where noted):
    warmup:  insightface crop target -> 256² -> F (appearance feature volume)
                                              -> M (source implicit keypoints)
    drive:   insightface crop driver -> 256² -> M (driving keypoints)
             relative motion: kp = kp_source + (kp_driving - kp_driving_initial)
             W (warping, torch eager — 5D grid_sample does NOT onnx-export)
             G (SPADE decoder -> 512², torch eager OR onnxruntime-TensorRT)
             paste 512² face back into the driving frame, return BGR uint8

Why this shape (Thor / sm110 findings honored):
  * torch.compile is dead on Thor — never used.
  * Speedup path is the onnxruntime TensorRT EP, and only for SPADE (the warping
    net's volumetric grid_sample can't be exported), so use_trt swaps ONLY G.
  * Audio IO would use soundfile/librosa, never torchaudio.

Interface contract (consumed by live_session.py):
    LiveVideoEngine(use_trt=False, device="cuda")
    .warmup(target_image_path: str) -> None
    .drive(driving_bgr: np.ndarray, audio_pcm16_16k) -> np.ndarray   # BGR uint8 HxWx3
    .ready -> bool

Self-test: ``conda run --no-capture-output -n hallo4-thor python \
    studio/backend/hallo4_studio/live_video_engine.py``
"""
from __future__ import annotations

import logging
import os
import os.path as osp
import sys
import threading
from copy import deepcopy
from typing import Any, Optional

import cv2
import numpy as np

log = logging.getLogger("hallo4.live_video_engine")
if not log.handlers:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

_SIBLING_LP = "/home/kp/thordrive2/repos/LivePortrait"


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


class LiveVideoEngine:
    """LivePortrait per-frame reenactment engine. Eager fp16 by default; SPADE via
    onnxruntime-TensorRT when ``use_trt=True`` (best-effort, falls back to eager)."""

    # ponytail: bbox expansion factor in lieu of LivePortrait's full landmark affine
    # cropper. Tune if the crop clips foreheads/chins. Upgrade path: src/utils/cropper.py.
    CROP_SCALE = 1.6

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

        # Per-session source state (set in warmup).
        self._feature_3d = None          # (1,32,16,64,64) fp16
        self._kp_source = None           # (1,21,3) fp32
        # Relative-motion anchor (set on first driven frame).
        self._kp_driving_initial = None
        self._mask512 = None             # cached feather alpha
        self._drive_failed_logged = False

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
    def _detect_crop(self, bgr: np.ndarray):
        """Detect the largest face; return (rgb256, box=(x0,y0,x1,y1)) or (None, None)."""
        faces = self._face.get(bgr)
        if not faces:
            return None, None
        f = max(faces, key=lambda a: (a.bbox[2] - a.bbox[0]) * (a.bbox[3] - a.bbox[1]))
        x1, y1, x2, y2 = f.bbox
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        half = max(x2 - x1, y2 - y1) * self.CROP_SCALE / 2.0
        h, w = bgr.shape[:2]
        x0, y0 = int(max(0, cx - half)), int(max(0, cy - half))
        x1b, y1b = int(min(w, cx + half)), int(min(h, cy + half))
        if x1b - x0 < 16 or y1b - y0 < 16:
            return None, None
        crop = bgr[y0:y1b, x0:x1b]
        rgb256 = cv2.cvtColor(cv2.resize(crop, (256, 256)), cv2.COLOR_BGR2RGB)
        return rgb256, (x0, y0, x1b, y1b)

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
        with torch.no_grad():
            kp_info = self._M(self._prep(rgb256))
        return self._kp_transform(kp_info)

    # ------------------------------------------------------------------ #
    # SPADE — eager or ORT-TensorRT                                      #
    # ------------------------------------------------------------------ #
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
    # Wav2Lip hook (optional)                                            #
    # ------------------------------------------------------------------ #
    def _lip_refine(self, frame_bgr: np.ndarray, audio_pcm16_16k: Any) -> np.ndarray:
        """HOOK for Wav2Lip-style mouth refinement driven by the (converted) audio.

        TODO(2c): integrate Wav2Lip. Needs a mel from the 16k PCM (librosa, NOT
        torchaudio — broken here) + the Wav2Lip checkpoint + an ORT-TRT or eager
        mouth generator on the crop region. Left as identity passthrough so the
        per-frame wiring exists; enabling it must not add new pip installs.
        """
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
        if self.use_trt and self._spade_sess is None:
            self._build_trt_spade()

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
        self._ready = True
        log.info("warmup done (real_weights=%s, trt=%s)", self.real_weights, self.trt_active)

    def drive(self, driving_bgr: np.ndarray, audio_pcm16_16k: Any = None) -> np.ndarray:
        """Per driving frame -> output BGR uint8 HxWx3. Unknown/failed frames pass through."""
        if not self._ready:
            return driving_bgr
        try:
            rgb256, box = self._detect_crop(driving_bgr)
            if rgb256 is None:
                return driving_bgr                                # no face -> unchanged
            torch = self._torch
            with torch.no_grad():
                kp_driving = self._kp_from_crop(rgb256)
                if self._kp_driving_initial is None:
                    self._kp_driving_initial = kp_driving          # relative-motion anchor
                # Relative motion: keep source identity/pose, apply driver's *changes*.
                # ponytail: additive delta (LivePortrait flag_relative_motion); skipped
                # stitching/retargeting nets — add from retargeting_models if neck drifts.
                kp = self._kp_source + (kp_driving - self._kp_driving_initial)
                warp_out = self._W(self._feature_3d,
                                   kp_driving=kp.half(),
                                   kp_source=self._kp_source.half())["out"]
                rgb512 = self._spade(warp_out)
            out = self._paste(driving_bgr, rgb512, box)
            return self._lip_refine(out, audio_pcm16_16k)
        except Exception as exc:  # noqa: BLE001 — a bad frame must never kill the session
            if not self._drive_failed_logged:
                log.warning("drive() failed on a frame (%s) — passing it through.", exc)
                self._drive_failed_logged = True
            return driving_bgr

    @property
    def ready(self) -> bool:
        return self._ready


def demo() -> None:
    """Self-test: warmup on assets/01.png, drive ~20 frames, assert valid output, print fps."""
    import time

    repo_root = osp.abspath(osp.join(osp.dirname(osp.abspath(__file__)), "..", "..", ".."))
    asset = osp.join(repo_root, "assets", "01.png")
    use_trt = os.environ.get("HALLO4_TEST_TRT", "0") == "1"

    eng = LiveVideoEngine(use_trt=use_trt)
    eng.warmup(asset)
    assert eng.ready, "engine not ready after warmup"

    driving = cv2.imread(asset)
    assert driving is not None, f"cannot read {asset}"

    for _ in range(3):                                            # warmup the CUDA path
        eng.drive(driving)
    n = 20
    t0 = time.perf_counter()
    for _ in range(n):
        out = eng.drive(driving)
    dt = (time.perf_counter() - t0) / n

    assert isinstance(out, np.ndarray), "output is not an ndarray"
    assert out.dtype == np.uint8, f"output dtype {out.dtype} != uint8"
    assert out.ndim == 3 and out.shape[2] == 3, f"output shape {out.shape} not HxWx3"

    print(f"output shape={out.shape} dtype={out.dtype}")
    print(f"real_weights={eng.real_weights}  trt_active={eng.trt_active}")
    print(f"per-frame {dt*1000:.1f} ms -> {1.0/dt:.1f} fps "
          f"({'TRT' if eng.trt_active else 'eager'} SPADE)")
    print("LIVE_VIDEO_ENGINE_OK")


if __name__ == "__main__":
    demo()
