# TransViber TransScriber

A Windows desktop app that transcribes multi-track recordings (one audio file
per speaker) and merges them into a single conversation ordered by who speaks
when. Transcription runs locally on your NVIDIA GPU — Whisper via
[faster-whisper](https://github.com/SYSTRAN/faster-whisper), NVIDIA
Parakeet/Canary via onnx-asr, and Qwen3-ASR via the `qwen-asr` package —
completely free and unlimited, with no API, account, or quota. If the GPU
fails (e.g. runs out of VRAM), the app automatically falls back to the CPU
and keeps going. A second tab writes AI notes on the finished transcript,
also locally.

## Usage

1. Double-click **`TransViber TransScriber.bat`** (or run
   `.venv\Scripts\python.exe app\main.py` from a terminal to see logs).
   The batch file is the normal launcher for this source copy; no rebuild
   is needed after changes to `app/`. The window opens before the background
   GPU check finishes, without loading the AI engines. If opening still
   takes unusually long, `app.log` beside the batch file records the click
   time, Python startup, and the time the window becomes ready.
2. Drag & drop audio into the window (or **Add Files…**). FLAC, M4A/AAC/MP4,
   MP3, WAV, OGG, Opus, MKA/MKV and more are supported. Either one file per
   speaker, or a single multitrack file:
   - a container with **multiple audio streams** (OBS/REAPER/Craig-style
     recordings) is expanded into one speaker track per stream, and
   - a **multichannel** file (more than 2 channels) is expanded into one
     speaker track per channel.
   - Stereo files are treated as one speaker; if the two channels are really
     two people, right-click the track and choose *Split into one speaker per
     channel*.
3. Double-click a **Speaker** cell to rename speakers (names default to the
   filename). Renaming works before or after transcription.
4. Optional: put unusual names and terms in the **Vocabulary hints** panel
   down the left side (e.g. D&D character/place names — one per line or
   comma-separated: `Zephyrine`, `Thaldrin`, `Waterdeep`). Whisper is biased
   toward those spellings; Qwen3-ASR takes them as decoder context. The panel
   only appears for Whisper and Qwen models (Parakeet/Canary don't support
   hints) and its contents are remembered between sessions.
5. Pick a model and language, then click **Transcribe**. The first use of a
   model downloads it (to `%USERPROFILE%\.cache\huggingface`); the status bar
   shows live download percentages. After that it loads from disk. The merged
   transcript builds live in the lower pane. The **☆ button** beside either
   model dropdown pins the selected model as the startup default (★); unstar
   it and the app goes back to remembering whichever model you used last.
   A loaded transcription model stays cached for repeat runs; changing the
   model/device, starting AI Notes, or closing the app releases it.
6. **Dark mode**: toggle the checkbox in the status bar (bottom right). It
   follows Windows' theme until you set it manually; then it's remembered.
7. While transcribing, an **elapsed timer** runs next to the Transcribe
   button (freezes at "took 4:12" when done), and the status bar shows live
   **GPU / VRAM / CPU / RAM** utilization (whole-system numbers via NVML,
   refreshed every 2 s).
8. Click **Export…** and choose a format — Markdown, plain text, SRT, VTT,
   JSON, or "All formats" to write all five at once.
9. The **Notes** tab has an **AI instructions** panel and the notes editor
   side by side, with the merged transcript (read-only, live-updating)
   spanning the full width underneath. The **Show: Instructions /
   Transcript** checkboxes hide either panel so the editor can take the whole
   width — hidden instructions are still applied — and both toggles are
   remembered. Notes autosave a few seconds after you stop typing and persist
   between sessions — as a plain file at
   `%LOCALAPPDATA%\TransViber TransScriber\notes.md`, so they can be
   recovered by hand if the app ever misbehaves. **Export Notes…** writes a
   copy to a .md/.txt file wherever you like.
10. **AI Notes** (Notes tab) has a local LLM write structured Markdown notes
    on the transcript, streamed into the editor under a
    `## AI Notes — 2026-08-05 14:30` heading (appended, so earlier notes and
    anything you typed are kept): summary, key points,
    quotes, follow-ups, and a per-person full-session summary at the end.
    Notes inference runs through the official **llama.cpp CUDA server**
    (`llama-cpp\llama-server.exe`, managed automatically by the app) with
    **GGUF** models — measured 182 tok/s on `qwen3-4b`, 76 tok/s on
    `gemma-3-12b` and ~180 tok/s on `gpt-oss-20b`, on the RTX 5070 Ti.

    The dropdown is **one entry per model**, each at the best quant that
    comfortably fits 16 GB: `qwen3-4b` (default) and `qwen3-14b`,
    `mistral-nemo-12b`, `llama-3.1-8b`, `llama-3.2-3b`, `phi-4-mini`. If you
    want a smaller build of one of them — less VRAM, quicker download — add
    a **`-q4` suffix** (`qwen3-14b-q4`, `mistral-nemo-12b-q4`, and so on).
    You can also type `repo_id` (its Q4_K_M file) or
    `repo_id::filename.gguf` for any GGUF on Hugging Face, including split
    multi-part ones. The optional instructions field ADDS to the
    standard format (e.g. "focus on plot events and NPC names"). During
    generation the Notes toolbar shows a progress bar (`43% · 1,204 tok`)
    and a live timer with token count and rate, and the button turns into
    **Cancel AI Notes**; when the run ends the status bar keeps the final
    numbers (`AI notes done in 1:42 · 1,204 tok · 76.3 tok/s avg`) until
    something else changes it.
    Long transcripts are processed in parts and merged. The button is
    locked while transcribing — the ASR models and the LLM take turns on
    the GPU, never sharing VRAM (the llama-server process is stopped after
    each notes run and on app close). If a model turns out not to fit, the
    app retries it with part of the model on the CPU rather than failing,
    and says so in the status bar — expect it to be a great deal slower.
    The status bar also reports what the run actually cost: the largest
    prompt it sent, any thinking tokens the model spent that you never see,
    and whether the notes were cut off at the length limit.

    The dropdown groups those into Max Detail, Standard Speed and Quick
    Review tiers; the transcription dropdown uses equivalent tiers. A fourth
    group, **Untested — Benchmark Candidates**, holds newer models added so
    they can be compared against the ones above: `gemma-4-12b`,
    `qwen3.5-9b`, `qwen3.5-4b`, `gpt-oss-20b`, `lfm2.5-8b-a1b`,
    `qwen3.8-27b`, `ministral-3-8b` and `granite-4.2-3b`. They are there to
    be measured, not because they are known to be better on this kind of
    transcript — only `gpt-oss-20b` has actually been run here. `gpt-oss-20b`
    and `qwen3.8-27b` are tight on 16 GB, and `qwen3.8-27b` may need a newer
    llama.cpp than the bundled one.

    Four of those supersede something: `gemma-4-12b` → `gemma-3-12b`,
    `qwen3.5-9b` → `qwen3-8b`, `ministral-3-8b` → `ministral-8b`,
    `granite-4.2-3b` → `granite-3.3-2b`. The older four have been taken out
    of the dropdown but are **still there by name** — type `gemma-3-12b`,
    `qwen3-8b`, `ministral-8b` or `granite-3.3-2b` to get one back, which is
    what to do if a replacement turns out not to load. `ministral-3-14b` and
    `granite-4.2-8b` are reachable the same way. `qwen3-4b` is the one
    exception kept in the dropdown despite having a successor: it is the
    startup default, and `qwen3.5-4b` has to beat it first.

    Licensing is not uniform: Google Gemma 3 and 4 are under the Gemma
    terms, `ministral-8b` under the Mistral Research License (personal use
    only — its replacement `ministral-3-8b` is Apache-2.0), and
    `lfm2.5-8b-a1b` under the **LFM Open License v1.0**, which restricts
    commercial use above $10M revenue. The rest are Apache-2.0 or MIT.
    Qwen3 8B/14B and the Qwen3.5/3.8 models are hybrid-thinking and are run
    with thinking switched off — the `qwen3-4b` default is an Instruct build
    with no thinking mode to begin with. `gpt-oss-20b` reasons rather than
    thinking aloud, so it gets a low reasoning effort and a larger answer
    allowance to compensate.

## Models

All models below are free (no accounts, no quotas) and run locally. They
were chosen as the strongest freely licensed options that run on Windows
with the word/segment timestamps the speaker-merge needs. The accuracy
notes come from the models' own published benchmarks — none of it has been
measured on recordings like yours, and sparse multi-track speaker audio is
a different workload from the continuous speech those benchmarks use.

| Model | VRAM (approx) | Notes |
|---|---|---|
| `qwen3-asr-1.7b` | ~5 GB | **Qwen3-ASR** (Apache-2.0; 30 languages plus 22 Chinese dialects) — LLM-based, strong on names/jargon/punctuation; supports vocabulary hints; `qwen3-asr-0.6b` is the smaller/faster variant. Runs via the official `qwen-asr` package on CUDA PyTorch, with word timestamps from `Qwen/Qwen3-ForcedAligner-0.6B`, which covers 11 languages. |
| `large-v3-turbo` | ~2 GB | Whisper's best all-rounder; supports vocabulary hints |
| `parakeet-tdt-0.6b-v2` | ~3 GB | The fastest model here, and top of the published English WER leaderboards (CC-BY-4.0) |
| `parakeet-tdt-0.6b-v3` | ~3 GB | Parakeet, 25 European languages (CC-BY-4.0) |
| `canary-1b-v2` | ~5 GB | Leads the published multilingual benchmarks; 25 European languages (CC-BY-4.0, commercial OK) |
| `large-v3` | ~3.5 GB | Most accurate Whisper; ~100 languages |
| `large-v2` | ~3.5 GB | Older; some prefer it (fewer hallucinations on some audio) |
| `distil-large-v3.5-ct2` | ~1.5 GB | Newer English-only Distil-Whisper; fast CTranslate2 build, MIT licensed |
| `distil-large-v3` | ~1.5 GB | English-only, very fast |
| `medium` / `medium.en` | ~1.5 GB | Smallest listed; type smaller names (small/base/tiny) manually if ever needed |

Two models score higher on the English Open ASR leaderboard but are not in
the app, because neither can be reached through the three backends it has.
NVIDIA Canary-Qwen-2.5B needs the NeMo toolkit (no reliable Windows/ONNX
path, weak timestamp support). Cohere Transcribe (03-2026) has supported
plain Transformers since June 2026, including long-audio chunking — so the
older "vLLM only" objection no longer holds — but it would still need a new
audio backend here and its timestamps would have to be checked against what
the speaker-merge needs. Adding either is real work, not a new name in the
model box.

The model box is editable — you can also type any CTranslate2 model repo ID
from Hugging Face (e.g. `distil-whisper/distil-large-v3.5-ct2`), or other
onnx-asr model names. Whisper models run via
faster-whisper/CTranslate2; NVIDIA models run via onnx-asr/ONNX Runtime, both
on the GPU with the shared CUDA 12 runtime. The Language dropdown applies to
Whisper **and** Qwen models (Qwen is given the language by name); only
Parakeet/Canary ignore it and auto-detect. Qwen3-ASR has no Norwegian or
Ukrainian, so those are greyed out while a Qwen model is selected, and
Transcribe asks for another choice if one of them was already picked. Note that the routing goes by
name: anything starting with `qwen` goes to the Qwen backend, names like
`parakeet…`/`canary…`/`nemo-…` go to onnx-asr, and everything else is
treated as a Whisper/CTranslate2 repo. Your RTX 5070 Ti (16 GB) handles any
of these easily.

Note for future package updates: `onnxruntime-gpu` must come from Microsoft's
CUDA 12 index (`--index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/`);
the default PyPI build targets CUDA 13 and won't find the pip-installed
CUDA 12 DLLs. Don't reinstall plain `onnxruntime` alongside it: both install
into the same `onnxruntime` folder, and the CPU one wins. See
[Setting up from source](#setting-up-from-source) for the install order.

## How the merge works

Each track is transcribed independently with voice-activity detection (so
silence while other people talk is skipped, not hallucinated). Every segment
is tagged with its track's speaker and timestamped; the app then sorts all
segments from all tracks by start time. For readable output (Markdown/text),
consecutive segments from the same speaker are joined into one paragraph;
SRT/VTT keep the fine-grained timing.

Multitrack recorders and apps like Craig/Audacity/REAPER start every track
at the same moment, so there is nothing to set. If someone recorded
separately and joined late, set their track's **Offset**: when their file
started, measured from the start of the session. Double-click the cell and
type seconds (`150`), minutes:seconds (`2:30`) or hours:minutes:seconds
(`1:02:30`); a comma works as the decimal point. Their lines move to the
right place straight away, in the transcript and in every export, without
transcribing again. Leave it at `0:00` if the file already contains the
waiting time as silence at the start, and type `0` to reset it.

**Align…** can work the offsets out for you from sound the recordings
share: people in the same room, one voice picked up by another mic, or a
recording of the whole session. Separately recorded headsets usually share
no sound with each other, so for those, add a full-session recording (a
call or room mix) with **Add session reference…** above the track list.
It is used for timing only: it is never transcribed, its start counts as
0:00, and the individual speaker files are still what gets transcribed.
Align reads the files (no AI model, no GPU) and shows a review window:
strong matches are ticked, weak matches and guesses are not, and nothing
changes until you press **Apply**. When there is no shared sound to go on,
the only suggestion is a labelled guess that every recorder stopped at the
same moment; check it before ticking it.

## Setting up from source

The repository holds only the source. The Python environment, the llama.cpp
server and the model weights are left out (see `.gitignore`), so a fresh
clone needs them recreated before the app runs. You need Windows, an NVIDIA
GPU with a current driver, and Python 3.14.

1. Create the venv and install the pinned packages. torch and ONNX Runtime
   go in first, each from its own index, because neither CUDA 12 build is on
   PyPI: PyPI's torch is the CPU build, and PyPI's `onnxruntime-gpu` is built
   for CUDA 13 from 1.27 on (and has no 1.27.1 at all). The lock then goes in
   with `--no-deps`, since it already lists every package; without that,
   faster-whisper pulls in the CPU `onnxruntime`, which overwrites the CUDA
   build's files.

   ```bat
   py -3.14 -m venv .venv
   .venv\Scripts\python.exe -m pip install --upgrade pip
   .venv\Scripts\python.exe -m pip install torch==2.11.0+cu128 ^
       --index-url https://download.pytorch.org/whl/cu128
   .venv\Scripts\python.exe -m pip install onnxruntime-gpu==1.27.1 --no-deps ^
       --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/
   .venv\Scripts\python.exe -m pip install --no-deps -r requirements.lock.txt
   ```

   To check the result, this should print `1.27.1 12.8`:

   ```bat
   .venv\Scripts\python.exe -c "import onnxruntime as o; from onnxruntime.capi import build_and_package_info as b; print(o.__version__, b.cuda_version)"
   ```

   `pip check` will report that faster-whisper requires `onnxruntime`; that
   one is expected, since `onnxruntime-gpu` provides the same module. If the
   check prints anything else or fails, or Parakeet only runs on the CPU,
   remove both builds and install the right one again:

   ```bat
   .venv\Scripts\python.exe -m pip uninstall -y onnxruntime onnxruntime-gpu
   .venv\Scripts\python.exe -m pip install onnxruntime-gpu==1.27.1 --no-deps ^
       --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/
   ```

2. For AI Notes, download the llama.cpp Windows CUDA x64 build (release
   b10034 is the one this app was verified against) and its matching CUDA
   runtime zip from the
   [llama.cpp releases page](https://github.com/ggml-org/llama.cpp/releases),
   then unzip both into a `llama-cpp\` folder at the project root so that
   `llama-cpp\llama-server.exe` exists. Transcription works without it.

3. Double-click **`TransViber TransScriber.bat`**. Transcription and notes
   models download on first use.

## Setup notes (already done on this machine)

- Python 3.14 venv in `.venv` with: `PySide6`, `faster-whisper`, `onnx-asr` +
  `onnxruntime-gpu` (Parakeet/Canary), `qwen-asr` + `torch` (Qwen3-ASR),
  `av`, `nvidia-cublas-cu12`, `nvidia-cudnn-cu12` (CUDA runtime DLLs come
  from pip — no CUDA Toolkit install needed; only an NVIDIA driver).
- `app/cuda_setup.py` adds those DLL directories at startup so CTranslate2
  finds them.
- AI Notes needs no Python LLM package: `llama-cpp/llama-server.exe` (the
  official llama.cpp CUDA build, b10034) is in the project folder and the
  app starts/stops it as needed.
- `requirements.lock.txt` pins every package in that venv, with the order to
  install them in (torch, then onnxruntime-gpu, each from its own index —
  neither CUDA 12 build is on PyPI) and the hash of the exact onnxruntime-gpu
  wheel. It is a record of what this build was verified against; regenerate
  it with `pip freeze` after changing the venv.
- `tests/` holds the offline regression tests — no GPU, no downloads, a few
  seconds:
  `.venv\Scripts\python.exe -m unittest discover -s tests -t .`
  They cover the merge and timestamp logic, the export formats, notes-model
  resolution and prompt sizing, and the transcript view. Anything involving
  a real model still needs the headless smoke test:
  `.venv\Scripts\python.exe app\main.py --selftest audio.wav`, or
  `--selftest --notes=gemma-4-12b audio.wav` to put one notes model through
  a run without clicking through the GUI.

## Standalone build (for sharing)

Everything needed to build a copy that runs without Python lives in
`packaging/`:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File packaging\build_release.ps1
```

That runs PyInstaller (`packaging\TransViber.spec`, onedir), compiles the
Inno Setup installer (`packaging\installer.iss`, needs Inno Setup 6) and
writes two shareable zips to `packaging\dist\`:
"TransViber TransScriber Setup.zip" and "… Portable.zip". Add
`-SkipPyInstaller` to rebuild only the installer/zips from an existing
`packaging\dist` tree. The build bundles the CUDA DLLs, the llama.cpp
server, and one preloaded Whisper model (`packaging\models\`), so the app
tree is ~7 GB and the installer ships as `Setup.exe` plus numbered `.bin`
slices. `packaging\README-FIRST.txt` is the plain-language readme that goes
in the app folder for whoever receives it.

Rebuild whenever `app/` changes — the zips are a snapshot, not a link to the
source.
