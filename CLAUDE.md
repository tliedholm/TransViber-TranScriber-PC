# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

**TransViber TransScriber** — a Windows PySide6 desktop app that transcribes
multi-track recordings (one audio file per speaker, or one multitrack
container) and merges them into a single conversation ordered by who speaks
when. All inference is local on an NVIDIA GPU (CUDA), with automatic CPU
fallback. There is a sibling **Mac edition** the PC version was ported from
and to; `PC-CHANGES-FOR-MAC.md` is a hand-off document of changes to
re-apply to the Mac code, not something to run here.

The whole app is under `app/`; there is no test suite, no linter config, and
no `requirements.txt` — the `.venv` is the source of truth for dependencies.

## Running & building

Everything runs through the in-repo virtualenv `.venv` (Python 3.14). Paths
below are from the project root (`multitrack-transcriber/`).

```powershell
# Run from source WITH logs (preferred while developing)
.venv\Scripts\python.exe app\main.py

# Run windowed, no console (what the .bat launcher does)
.venv\Scripts\pythonw.exe app\main.py

# Headless smoke test — transcribe a file with one or more models,
# GPU first then CPU fallback, printing segments. This is the only
# automated verification the repo has.
.venv\Scripts\python.exe app\main.py --selftest audio.wav large-v3-turbo parakeet-tdt-0.6b-v2
# add --notes as the first arg after --selftest to also exercise the LLM:
.venv\Scripts\python.exe app\main.py --selftest --notes audio.wav
```

Use `--selftest` to verify transcription/notes changes end-to-end; prefer it
over launching the GUI.

### Standalone build (PyInstaller onedir → installer + portable zip)

```powershell
# Full release (PyInstaller, then Inno Setup installer + portable zip)
powershell -NoProfile -ExecutionPolicy Bypass -File packaging\build_release.ps1
# Rebuild only installer/zip from an existing dist:
powershell ... -File packaging\build_release.ps1 -SkipPyInstaller

# PyInstaller alone:
.venv\Scripts\python.exe -m PyInstaller packaging\TransViber.spec --noconfirm
```

The app tree is ~7 GB (bundled CUDA DLLs, llama.cpp, a preloaded model), so
the installer ships as `Setup.exe` + numbered `.bin` slices (a single Windows
setup exe caps at ~4.2 GB). Requires Inno Setup 6; the script looks for
`ISCC.exe` under `%LOCALAPPDATA%\Programs\Inno Setup 6` and
`%ProgramFiles(x86)%\Inno Setup 6`. `installer.iss` carries the `AppVersion`
string — bump it when shipping a new build.

## Architecture

### Data flow

`audio.py` (probe/decode) → `transcriber.py` (ASR, background thread) →
per-track `Segment`s → `merger.py` (sort all tracks by start time, coalesce
same-speaker runs) → `exporters.py` (md/txt/srt/vtt/json) and/or `llm.py`
(AI notes). `main_window.py` orchestrates all of it; `models.py` holds the
shared `Track`/`Segment`/`TrackStatus` dataclasses.

### Window layout

A `QTabWidget` with two tabs, both built in `_build_ui`:

- **Transcribe** — control bar (files, model + ☆ star, language, device,
  elapsed label, Transcribe/Export) over a horizontal splitter: the
  full-height **Vocabulary hints** panel on the left (shown only when
  `engine_for(model)` is whisper or qwen), track table + live transcript on
  the right.
- **Notes** — a vertical splitter: on top, the **AI instructions** panel
  beside the notes toolbar + editor; underneath, the full-width read-only
  transcript mirror. The toolbar's "Show: Instructions / Transcript"
  checkboxes toggle those two panels (instructions still apply when
  hidden), and AI notes are *appended* to the editor under a dated
  `## AI Notes — …` heading, never replacing what's there.

### Threading & signals

Every long-running job is a `QThread` subclass communicating with the UI via
Qt `Signal`s (never touch widgets off the main thread):
`TranscribeWorker` (`transcriber.py`), `NotesWorker` (`llm.py`), and
`StatsPoller` (`stats.py`, the GPU/VRAM/CPU/RAM status readout via NVML +
psutil). `MainWindow` connects to their signals and updates widgets.

### The core merge idea

Each track is transcribed independently with VAD (silence while others talk
is skipped, not hallucinated). Segments are tagged with the track's speaker,
then **all** segments across tracks are sorted by start time. This assumes
all tracks start at the same instant (as Craig/OBS/REAPER exports do).

`SPLIT_GAP_SECONDS` (1.0s) is load-bearing: Whisper/Qwen can lump utterances
separated by silence into one segment, which would sort a speaker's later
remark at the time of their earlier one and break the interleaving. Segments
are split at word-level silence gaps (`split_on_word_gaps`) to prevent this.

### Three ASR engines behind one interface

`engine_for(model_name)` in `transcriber.py` routes each model to one of:
- **whisper** — faster-whisper / CTranslate2 (default; the `.transcribe`
  path with `word_timestamps`, `hotwords`, VAD).
- **onnx** — onnx-asr / ONNX Runtime for NVIDIA Parakeet & Canary
  (`_transcribe_track_onnx`). Ignores the Language dropdown (auto-detect).
- **qwen** — Qwen3-ASR via the `qwen-asr` package + a forced aligner for
  word timestamps (`_transcribe_track_qwen`). **Critical constraint:** Qwen
  must never be fed silence — it hallucinates repeated text into gaps and
  the aligner smears it across the silence, corrupting timestamps. The code
  cuts audio into VAD speech regions, detects language once from the
  clearest sample and locks it for the whole track, then drops segments
  written in a script the locked language can't use (`_foreign_script`).

Adding a model usually means adding an alias to `WHISPER_ALIASES` /
`ONNX_ALIASES` / `QWEN_REPOS`, and (for the UI) to `MODEL_GROUPS` in
`main_window.py`. The model combo box is editable, so any CTranslate2 repo id
or onnx-asr name also works when typed.

### AI Notes (local LLM)

`llm.py` runs notes generation through the **official llama.cpp CUDA server**
(`llama-cpp/llama-server.exe`), not an in-process library. The app spawns and
manages the server process: `_load()` starts it for the chosen GGUF,
`free_llm()` kills it. **The ASR models and the notes LLM never hold VRAM at
the same time** — loading the LLM calls `free_asr_models()` first, and the
server is killed after each notes run (killing the process is the only
reliable way to reclaim its C-level VRAM). It is also assigned to a Windows
job object with `KILL_ON_JOB_CLOSE` (`_kill_with_parent`) so it still dies
if the app crashes instead of closing — keep that, or a crash strands
several GB of VRAM. Long transcripts are handled map-reduce style: notes per
part, then a combining pass. **Every prompt is sized by real tokens**: parts
are cut at line boundaries using the ratio measured from this transcript and
then verified against the server's `/tokenize` (`_pack`), and the combining
pass reduces in rounds when the per-part notes don't fit one prompt.
`CHARS_PER_TOKEN` is only a starting guess and a fallback for when
`/tokenize` is unreachable — don't reintroduce it as the sizing rule; an
oversized prompt is truncated by llama-server silently.

Models are GGUFs listed in
`LLM_REPOS`: Q4_K_M by default, with a `-q8` name suffix for the Q8_0 build
of the same model (offered for the small/medium tiers only — a Q8 12–14B
won't fit 16 GB with the context). The box accepts `repo_id` or
`repo_id::filename.gguf`.

### CUDA runtime & source-vs-frozen paths

There is **no CUDA Toolkit install** — the CUDA runtime DLLs (cuBLAS, cuDNN)
come from pip wheels. `cuda_setup.py` adds their directories to the DLL
search path **and must be imported before ctranslate2/faster_whisper load**
(`transcriber.py` imports it first, `# noqa: F401`). In the frozen build the
DLLs come from torch's bundled `torch/lib` set instead of the nvidia wheels.

`runtime.py` centralizes every path that differs between source and the
PyInstaller build (`IS_FROZEN`): `bundle_dir()` (read-only bundled data),
`writable_dir()` (per-user LOCALAPPDATA folder for logs when the install dir
is read-only), `asset_path()` (files from `app/assets`, which the spec
bundles at the build root), and `bundled_model_dir()` (the one Whisper model
preloaded into the build). When adding files the app reads or writes, route
the path through `runtime.py` rather than hardcoding the project tree.

### Persistence

User settings persist via Qt `QSettings` (registry under the org/app names
set in `main.py`): model/notes-model/language choices, "starred" default
models, vocabulary hints, AI instructions, autosaved notes text, the Notes
tab's two pane-visibility toggles (`notes_show_instructions`,
`notes_show_transcript`), and dark mode. See
`_load_settings`/`_save_settings` in `main_window.py`.

## Conventions

- Heavy imports (torch, faster_whisper, onnx_asr, av, requests, numpy) are
  imported **lazily inside functions**, not at module top, to keep startup
  fast and let `cuda_setup` run first. Follow this pattern.
- `models.py` `AUDIO_EXTENSIONS` is the accepted-format allowlist for
  drag-drop and the file dialog.
- A multitrack file expands into per-speaker `Track`s in `tracks_for_file`:
  multiple audio streams → one track per stream; >2 channels → one per
  channel; stereo stays one speaker unless split. Sub-tracks
  (`Track.is_subtrack`) are decoded on demand by `decode_track_audio`.
