from __future__ import annotations

import ctypes
import os
import site
from pathlib import Path
from typing import Any


def prepare_cuda_runtime() -> None:
    """Expose NVIDIA pip wheel CUDA/cuDNN libraries to ONNX Runtime."""
    for lib_dir in _nvidia_library_dirs():
        _prepend_ld_library_path(lib_dir)

    for path in _nvidia_library_paths():
        try:
            ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
        except OSError:
            continue


def assert_cuda_provider_available() -> None:
    try:
        import onnxruntime as ort
    except Exception as exc:
        raise RuntimeError(
            "GPU embeddings are enabled, but onnxruntime is not importable. "
            "Install the GPU FastEmbed package with `pip install fastembed-gpu`."
        ) from exc

    available = set(ort.get_available_providers())
    if "CUDAExecutionProvider" not in available:
        raise RuntimeError(
            "GPU embeddings are enabled, but ONNX Runtime does not expose "
            f"CUDAExecutionProvider. Available providers: {sorted(available)}. "
            "Install GPU dependencies with `pip uninstall -y fastembed onnxruntime` "
            "then `pip install fastembed-gpu` in the same Python environment."
        )


def active_onnx_providers(model: Any) -> list[str]:
    candidates = [
        ("model", "model"),
        ("model", "_model"),
        ("_model", "model"),
        ("_model", "_model"),
    ]
    for path in candidates:
        current = model
        for attr in path:
            current = getattr(current, attr, None)
            if current is None:
                break
        if current is not None and hasattr(current, "get_providers"):
            return list(current.get_providers())
    return []


def _prepend_ld_library_path(lib_dir: Path) -> None:
    current = os.environ.get("LD_LIBRARY_PATH")
    lib_dir_text = str(lib_dir)
    if not current:
        os.environ["LD_LIBRARY_PATH"] = lib_dir_text
    elif lib_dir_text not in current.split(":"):
        os.environ["LD_LIBRARY_PATH"] = f"{lib_dir_text}:{current}"


def _nvidia_library_dirs() -> list[Path]:
    lib_dirs = []
    for root in _nvidia_package_roots():
        lib_dirs.extend(
            [
                root / "cuda_nvrtc" / "lib",
                root / "cublas" / "lib",
                root / "cudnn" / "lib",
            ]
        )
    return [path for path in lib_dirs if path.exists()]


def _nvidia_library_paths() -> list[Path]:
    paths = []
    patterns_by_dir = {
        "cuda_nvrtc/lib": ["libnvrtc-builtins.so*", "libnvrtc.so*"],
        "cublas/lib": ["libcublasLt.so*", "libcublas.so*"],
        "cudnn/lib": ["libcudnn*.so*"],
    }
    for package_root in _nvidia_package_roots():
        for relative_dir, patterns in patterns_by_dir.items():
            lib_dir = package_root / relative_dir
            if not lib_dir.exists():
                continue
            for pattern in patterns:
                paths.extend(sorted(lib_dir.glob(pattern)))
    return paths


def _nvidia_package_roots() -> list[Path]:
    return [Path(path) / "nvidia" for path in site.getsitepackages()]
