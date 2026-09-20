# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the standalone TransViber TransScriber build.

Build from the project root with:
    .venv\\Scripts\\python.exe -m PyInstaller packaging\\TransViber.spec --noconfirm

Produces dist/TransViber TransScriber/ containing the windowed app exe, a
console "TransViber Debug.exe" for troubleshooting, and everything else in
_internal/ — including the llama.cpp server, the NVIDIA CUDA DLLs from the
pip wheels (found by app/cuda_setup.py), and the preloaded Whisper model
(packaging/models/, found by app/runtime.py).
"""

import sysconfig
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, copy_metadata

ROOT = Path(SPECPATH).resolve().parent  # spec lives in packaging/
APP = ROOT / "app"

datas = [
    (str(APP / "assets"), "assets"),
    (str(ROOT / "packaging" / "models"), "models"),
]

# Runtime data files of the ASR packages (silero VAD, mel filterbanks, …).
for pkg in ("faster_whisper", "onnx_asr", "qwen_asr"):
    datas += collect_data_files(pkg)

# onnx_asr reads its own version from the installed-package metadata.
datas += copy_metadata("onnx-asr")

# nagisa (Japanese tokenizer used by qwen-asr's forced aligner) is py2-era
# code: it sys.path.append()s its own package dir and does bare
# `import prepro` / `import model`, which static analysis can't see. Ship the
# whole package as plain files so that keeps working.
site = Path(sysconfig.get_paths()["purelib"])
for f in (site / "nagisa").rglob("*"):
    if f.is_file() and f.suffix != ".pyc":
        datas.append((str(f), str(f.parent.relative_to(site))))

# llama.cpp: the server binary and its DLLs only — the other bundled tools
# (llama-cli, benchmarks, …) are never used. Added at COLLECT level (not
# Analysis datas) so PyInstaller doesn't reclassify the DLLs as binaries and
# duplicate their CUDA dependencies into the bundle root.
llama_tree = [
    (f"llama-cpp/{f.name}", str(f), "DATA")
    for f in (ROOT / "llama-cpp").iterdir()
    if f.name == "llama-server.exe" or f.suffix.lower() == ".dll"
]

# The NVIDIA pip wheels are NOT bundled: torch/lib ships a complete
# cuBLAS/cuDNN/cuFFT/NVRTC set that ctranslate2 and onnxruntime load via
# cuda_setup.add_nvidia_dll_dirs() in frozen mode.

a = Analysis(
    [str(APP / "main.py")],
    pathex=[str(APP)],
    binaries=[],
    datas=datas,
    hiddenimports=[],
    excludes=[
        # Pulled in as dependencies of dev/serving extras, unused by the app.
        "gradio", "gradio_client", "fastapi", "starlette", "uvicorn",
        "flask", "werkzeug", "bitsandbytes", "IPython", "tkinter",
        "matplotlib", "pytest",
    ],
    noarchive=False,
)

pyz = PYZ(a.pure)

common = dict(
    exclude_binaries=True,
    icon=str(APP / "assets" / "icon.ico"),
    upx=False,
)

exe = EXE(pyz, a.scripts, [], name="TransViber TransScriber",
          console=False, **common)
exe_debug = EXE(pyz, a.scripts, [], name="TransViber Debug",
                console=True, **common)

coll = COLLECT(exe, exe_debug, a.binaries, a.datas, llama_tree,
               name="TransViber TransScriber", upx=False)
