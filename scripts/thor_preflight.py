#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import shutil
import site
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class Check:
    name: str
    status: str
    detail: str


def run_text(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:  # noqa: BLE001
        return f"{type(exc).__name__}: {exc}"


def add(checks: list[Check], name: str, ok: bool, detail: str) -> None:
    checks.append(Check(name=name, status="ok" if ok else "fail", detail=detail))


def warn(checks: list[Check], name: str, detail: str) -> None:
    checks.append(Check(name=name, status="warn", detail=detail))


def find_x86_wheels() -> list[str]:
    offenders: list[str] = []
    roots: list[str] = []
    try:
        roots.extend(site.getsitepackages())
    except Exception:
        pass

    for root in roots:
        root_path = Path(root)
        if not root_path.exists() or not str(root_path).startswith(sys.prefix):
            continue
        for wheel in root_path.glob("*.dist-info/WHEEL"):
            try:
                text = wheel.read_text(errors="ignore")
            except OSError:
                continue
            if "x86_64" in text or "linux_x86_64" in text:
                offenders.append(str(wheel))
    return offenders


def import_check(name: str) -> tuple[bool, str]:
    try:
        module = importlib.import_module(name)
    except Exception as exc:  # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}"
    version = getattr(module, "__version__", "imported")
    return True, str(version)


def check_torch(checks: list[Check]) -> None:
    ok, detail = import_check("torch")
    add(checks, "import:torch", ok, detail)
    if not ok:
        return

    import torch

    add(checks, "torch:version", torch.__version__ == "2.10.0+cu130", torch.__version__)
    add(checks, "torch:cuda_available", bool(torch.cuda.is_available()), str(torch.cuda.is_available()))
    if torch.cuda.is_available():
        capability = torch.cuda.get_device_capability(0)
        add(checks, "torch:device_capability", capability == (11, 0), str(capability))
        try:
            x = torch.randn((64, 64), device="cuda", dtype=torch.float16)
            y = x @ x
            torch.cuda.synchronize()
            add(checks, "torch:cuda_tensor_op", y.is_cuda, f"sum={float(y.float().sum().item()):.4f}")
        except Exception as exc:  # noqa: BLE001
            add(checks, "torch:cuda_tensor_op", False, f"{type(exc).__name__}: {exc}")

        try:
            q = torch.randn((1, 2, 16, 64), device="cuda", dtype=torch.float16)
            out = torch.nn.functional.scaled_dot_product_attention(q, q, q)
            torch.cuda.synchronize()
            add(checks, "attention:sdpa", out.is_cuda and out.shape == q.shape, str(tuple(out.shape)))
        except Exception as exc:  # noqa: BLE001
            add(checks, "attention:sdpa", False, f"{type(exc).__name__}: {exc}")

        try:
            from vace.models.wan.modules import attention as wan_attention

            q = torch.randn((1, 16, 2, 64), device="cuda", dtype=torch.float16)
            out = wan_attention.attention(q, q, q, dtype=torch.float16)
            torch.cuda.synchronize()
            add(checks, "attention:hallo4", out.is_cuda and out.shape == q.shape, str(tuple(out.shape)))
            flash_available = bool(wan_attention.FLASH_ATTN_2_AVAILABLE or wan_attention.FLASH_ATTN_3_AVAILABLE)
            add(checks, "attention:flash_available", flash_available, str(flash_available))
        except Exception as exc:  # noqa: BLE001
            add(checks, "attention:hallo4", False, f"{type(exc).__name__}: {exc}")


def create_checkpoint_alias(checks: list[Check]) -> None:
    ckpt = PROJECT_ROOT / "pretrained_models/hallo4/model_weight.ckpt"
    pth = PROJECT_ROOT / "pretrained_models/hallo4/model_weight.pth"
    if ckpt.exists():
        add(checks, "checkpoint:alias", True, str(ckpt.relative_to(PROJECT_ROOT)))
        return
    if not pth.exists():
        add(checks, "checkpoint:alias", False, "missing model_weight.pth and model_weight.ckpt")
        return
    try:
        ckpt.symlink_to(pth.name)
        add(checks, "checkpoint:alias", True, f"created {ckpt.relative_to(PROJECT_ROOT)} -> {pth.name}")
    except OSError:
        shutil.copy2(pth, ckpt)
        add(checks, "checkpoint:alias", True, f"copied {pth.relative_to(PROJECT_ROOT)} to .ckpt alias")


def check_models(checks: list[Check], create_alias: bool) -> None:
    if create_alias:
        create_checkpoint_alias(checks)

    required = {
        "model:hallo4_checkpoint": [
            PROJECT_ROOT / "pretrained_models/hallo4/model_weight.ckpt",
            PROJECT_ROOT / "pretrained_models/hallo4/model_weight.pth",
        ],
        "model:wan_vae": [PROJECT_ROOT / "pretrained_models/Wan2.1_Encoders/Wan2.1_VAE.pth"],
        "model:wan_t5": [PROJECT_ROOT / "pretrained_models/Wan2.1_Encoders/models_t5_umt5-xxl-enc-bf16.pth"],
        "model:audio_separator": [PROJECT_ROOT / "pretrained_models/audio_separator/Kim_Vocal_2.onnx"],
        "model:wav2vec_config": [
            PROJECT_ROOT / "pretrained_models/wav2vec/wav2vec2-base-960h/config.json",
            PROJECT_ROOT / "pretrained_models/wav2vec2-base-960h/config.json",
        ],
    }
    for name, candidates in required.items():
        found = next((p for p in candidates if p.exists()), None)
        add(checks, name, found is not None, str(found.relative_to(PROJECT_ROOT)) if found else "missing")


def check_imports(checks: list[Check], skip_imports: Iterable[str]) -> None:
    modules = [
        "torchvision",
        "decord",
        "audio_separator",
        "onnxruntime",
        "transformers",
        "diffusers",
        "vace.models.wan",
    ]
    skip = set(skip_imports)
    for name in modules:
        if name in skip:
            warn(checks, f"import:{name}", "skipped")
            continue
        ok, detail = import_check(name)
        add(checks, f"import:{name}", ok, detail)

    if "onnxruntime" not in skip:
        try:
            import onnxruntime as ort

            providers = ort.get_available_providers()
            add(checks, "onnxruntime:providers", any("CUDA" in p or "Tensorrt" in p for p in providers), ", ".join(providers))
        except Exception as exc:  # noqa: BLE001
            add(checks, "onnxruntime:providers", False, f"{type(exc).__name__}: {exc}")


def check_decord_sample(checks: list[Check]) -> None:
    sample = PROJECT_ROOT / "assets/01.mp4"
    if not sample.exists():
        warn(checks, "decord:sample_decode", "assets/01.mp4 missing")
        return

    backend = os.getenv("HALLO4_VIDEO_BACKEND", "auto").strip().lower()
    prefer_decord = backend == "decord" or (backend == "auto" and platform.machine() != "aarch64")
    if not prefer_decord:
        add(checks, "video_decode:backend", True, "imageio")
        try:
            import imageio.v2 as imageio

            reader = imageio.get_reader(sample, "ffmpeg")
            try:
                frame = reader.get_data(0)
                count = reader.count_frames()
            finally:
                reader.close()
            add(checks, "video_decode:imageio", count > 0, f"frames={count} shape={tuple(frame.shape)}")
        except Exception as exc:  # noqa: BLE001
            add(checks, "video_decode:imageio", False, f"{type(exc).__name__}: {exc}")
        return

    try:
        import decord

        reader = decord.VideoReader(str(sample))
        frame = reader[0]
        add(checks, "decord:sample_decode", len(reader) > 0, f"frames={len(reader)} shape={tuple(frame.shape)}")
    except Exception as exc:  # noqa: BLE001
        warn(checks, "decord:sample_decode", f"{type(exc).__name__}: {exc}")
        try:
            import imageio.v2 as imageio

            reader = imageio.get_reader(sample, "ffmpeg")
            try:
                frame = reader.get_data(0)
                count = reader.count_frames()
            finally:
                reader.close()
            add(checks, "video_decode:imageio_fallback", count > 0, f"frames={count} shape={tuple(frame.shape)}")
        except Exception as fallback_exc:  # noqa: BLE001
            add(checks, "video_decode:imageio_fallback", False, f"{type(fallback_exc).__name__}: {fallback_exc}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Hallo4 Thor / CUDA 13 / SM110 runtime compatibility.")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of a table.")
    parser.add_argument("--skip-models", action="store_true", help="Skip pretrained model file checks.")
    parser.add_argument("--skip-imports", nargs="*", default=[], help="Module import names to skip.")
    parser.add_argument("--create-checkpoint-alias", action="store_true", help="Create model_weight.ckpt alias when only .pth exists.")
    args = parser.parse_args()

    checks: list[Check] = []
    add(checks, "system:arch", platform.machine() == "aarch64", platform.machine())
    add(checks, "system:cuda_nvcc", "release 13." in run_text(["nvcc", "--version"]), run_text(["nvcc", "--version"]).splitlines()[-1])
    add(checks, "system:nvidia_smi", "NVIDIA Thor" in run_text(["nvidia-smi"]), run_text(["nvidia-smi"]).splitlines()[0])
    nv_tegra = Path("/etc/nv_tegra_release")
    add(checks, "system:l4t_r39", nv_tegra.exists() and "R39" in nv_tegra.read_text(errors="ignore"), nv_tegra.read_text(errors="ignore").splitlines()[0] if nv_tegra.exists() else "missing")
    offenders = find_x86_wheels()
    add(checks, "packages:no_x86_64_wheels", not offenders, "; ".join(offenders[:5]) if offenders else "none")

    check_torch(checks)
    check_imports(checks, args.skip_imports)
    if "decord" not in set(args.skip_imports):
        check_decord_sample(checks)
    if not args.skip_models:
        check_models(checks, args.create_checkpoint_alias)

    if args.json:
        print(json.dumps([asdict(check) for check in checks], indent=2))
    else:
        width = max(len(check.name) for check in checks)
        for check in checks:
            print(f"{check.status.upper():5} {check.name:<{width}}  {check.detail}")

    return 1 if any(check.status == "fail" for check in checks) else 0


if __name__ == "__main__":
    raise SystemExit(main())
