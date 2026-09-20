"""Make pip-installed NVIDIA CUDA DLLs (cuBLAS, cuDNN) visible to CTranslate2.

Must be imported before ctranslate2 / faster_whisper.
"""

import os
import sys
from pathlib import Path


def add_nvidia_dll_dirs() -> None:
    if getattr(sys, "frozen", False):
        # PyInstaller build: torch's bundled CUDA DLLs (torch/lib carries a
        # full cuBLAS/cuDNN/cuFFT set) serve ctranslate2 and onnxruntime too,
        # so the nvidia wheels are not bundled separately.
        root = Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
        torch_lib = root / "torch" / "lib"
        if torch_lib.is_dir():
            os.add_dll_directory(str(torch_lib))
            os.environ["PATH"] = str(torch_lib) + os.pathsep + os.environ.get("PATH", "")
        base = root / "nvidia"
    else:
        base = Path(sys.prefix) / "Lib" / "site-packages" / "nvidia"
    if not base.is_dir():
        return
    for sub in base.iterdir():
        bin_dir = sub / "bin"
        if bin_dir.is_dir():
            os.add_dll_directory(str(bin_dir))
            os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ.get("PATH", "")


add_nvidia_dll_dirs()
