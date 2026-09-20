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
   between sessions; **Export Notes…** writes them to a .md/.txt file.
10. **AI Notes** (Notes tab) has a local LLM write structured Markdown notes
    on the transcript, streamed into the editor under a
    `## AI Notes — 2026-08-05 14:30` heading (appended, so earlier notes and
    anything you typed are kept): summary, key points,
    quotes, follow-ups, and a per-person full-session summary at the end.
    Notes inference runs through the official **llama.cpp CUDA server**
    (`llama-cpp\llama-server.exe`, managed automatically by the app) with
    **GGUF** models — measured 182 tok/s on `qwen3-4b` and 76 tok/s
    on `gemma-3-12b` on the RTX 5070 Ti. Models: `qwen3-4b` (default),
    `qwen3-8b`, `qwen3-14b`, `gemma-3-12b`, `llama-3.1-8b`, `llama-3.2-3b`,
    `mistral-nemo-12b`, `ministral-8b`, `phi-4-mini`, `granite-3.3-2b` —
    all Q4_K_M, and most of the small and medium ones also have a **`-q8`
    variant** (`qwen3-4b-q8`, `qwen3-8b-q8`, `llama-3.1-8b-q8`,
    `llama-3.2-3b-q8`, `ministral-8b-q8`, `phi-4-mini-q8`,
    `granite-3.3-2b-q8`): the 8-bit build of the same model — better quality
    at roughly double the download and VRAM. The 12–14B models stay Q4-only
    (a Q8 of those wouldn't fit 16 GB alongside the context). You can also
    type `repo_id` (first Q4_K_M file) or `repo_id::filename.gguf` for any
    GGUF on Hugging Face. The optional instructions field ADDS to the
    standard format (e.g. "focus on plot events and NPC names"). During
    generation the Notes toolbar shows a progress bar (`43% · 1,204 tok`)
    and a live timer with token count and rate, and the button turns into
    **Cancel AI Notes**; when the run ends the status bar keeps the final
    numbers (`AI notes done in 1:42 · 1,204 tok · 76.3 tok/s avg`) until
    something else changes it.
    Long transcripts are processed in parts and merged. The button is
    locked while transcribing — the ASR models and the LLM take turns on
    the GPU, never sharing VRAM (the llama-server process is stopped after
    each notes run and on app close).

    The dropdown groups every notes model into Max Detail, Standard Speed,
    and Quick Review tiers; the transcription dropdown uses equivalent
    tiers. Google Gemma 3 is distributed under the Gemma terms;
    `ministral-8b` uses the Mistral Research License (free for personal
    use); Qwen3 8B/14B are hybrid-thinking models run with thinking
    disabled.

## Models

All models below are free (no accounts, no quotas) and run locally. As of
mid-2026 this is the strongest freely licensed lineup that runs on Windows
with the word/segment timestamps the speaker-merge needs.

| Model | VRAM (approx) | Notes |
|---|---|---|
| `qwen3-asr-1.7b` | ~5 GB | **Qwen3-ASR** (Apache-2.0, 52 languages) — LLM-based, best with names/jargon/punctuation; supports vocabulary hints; `qwen3-asr-0.6b` is the smaller/faster variant. Runs via the official `qwen-asr` package on CUDA PyTorch, with word timestamps from `Qwen/Qwen3-ForcedAligner-0.6B`. |
| `large-v3-turbo` | ~2 GB | Whisper's best all-rounder; supports vocabulary hints |
| `parakeet-tdt-0.6b-v2` | ~3 GB | **Best English accuracy** of any local model here, and the fastest (CC-BY-4.0) |
| `parakeet-tdt-0.6b-v3` | ~3 GB | Parakeet, 25 European languages (CC-BY-4.0) |
| `canary-1b-v2` | ~5 GB | **Best multilingual accuracy**, 25 European languages (CC-BY-4.0, commercial OK) |
| `large-v3` | ~3.5 GB | Most accurate Whisper; ~100 languages |
| `large-v2` | ~3.5 GB | Older; some prefer it (fewer hallucinations on some audio) |
| `distil-large-v3.5-ct2` | ~1.5 GB | Newer English-only Distil-Whisper; fast CTranslate2 build, MIT licensed |
| `distil-large-v3` | ~1.5 GB | English-only, very fast |
| `medium` / `medium.en` | ~1.5 GB | Smallest listed; type smaller names (small/base/tiny) manually if ever needed |

Two models score higher on the English Open ASR leaderboard but are not in
the app for practical reasons: NVIDIA Canary-Qwen-2.5B requires the NeMo
toolkit (no reliable Windows/ONNX path, weak timestamp support) and Cohere
Transcribe (03-2026) only ships a vLLM runtime (Linux-oriented, timestamps
unclear). If either becomes available in onnx-asr or CTranslate2, it can be
typed straight into the model box or added to the list.

The model box is editable — you can also type any CTranslate2 model repo ID
from Hugging Face (e.g. `distil-whisper/distil-large-v3.5-ct2`), or other
onnx-asr model names. Whisper models run via
faster-whisper/CTranslate2; NVIDIA models run via onnx-asr/ONNX Runtime, both
on the GPU with the shared CUDA 12 runtime. The Language dropdown only
affects Whisper models — Parakeet/Canary auto-detect. Your RTX 5070 Ti
(16 GB) handles any of these easily.

Note for future package updates: `onnxruntime-gpu` must come from Microsoft's
CUDA 12 index (`--index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/`);
the default PyPI build targets CUDA 13 and won't find the pip-installed
CUDA 12 DLLs. Don't reinstall plain `onnxruntime` alongside it.

## How the merge works

Each track is transcribed independently with voice-activity detection (so
silence while other people talk is skipped, not hallucinated). Every segment
is tagged with its track's speaker and timestamped; the app then sorts all
segments from all tracks by start time. For readable output (Markdown/text),
consecutive segments from the same speaker are joined into one paragraph;
SRT/VTT keep the fine-grained timing.

This assumes all tracks start at the same moment (as multitrack recorders and
apps like Craig/Audacity/REAPER produce). If a track is offset, align it in an
editor first.

## Setup notes (already done on this machine)

- Python 3.14 venv in `.venv` with: `PySide6`, `faster-whisper`, `onnx-asr` +
  `onnxruntime-gpu` (Parakeet/Canary), `qwen-asr` + `torch` (Qwen3-ASR),
  `av`, `nvidia-cublas-cu12`, `nvidia-cudnn-cu12` (CUDA runtime DLLs come
  from pip — no CUDA Toolkit install needed; only an NVIDIA driver).
- `app/cuda_setup.py` adds those DLL directories at startup so CTranslate2
  finds them.
- AI Notes needs no Python LLM package: `llama-cpp/llama-server.exe` (the
  official llama.cpp CUDA build) is in the project folder and the app
  starts/stops it as needed.

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
