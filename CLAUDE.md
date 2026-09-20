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

The whole app is under `app/`, the offline tests under `tests/`. There is no
linter config. `requirements.lock.txt` pins every package in the `.venv` as
a record of what the app and the standalone build were last verified
against — the `.venv` is still the working source of truth, and the lock
file is regenerated from it, not the other way round.

## Running & building

Everything runs through the in-repo virtualenv `.venv` (Python 3.14). Paths
below are from the project root (`multitrack-transcriber/`).

```powershell
# Offline tests — seconds, no GPU, no downloads, no window
.venv\Scripts\python.exe -m unittest discover -s tests -t .
.venv\Scripts\python.exe -m unittest tests.test_llm -v        # one module

# Run from source WITH logs (preferred while developing)
.venv\Scripts\python.exe app\main.py

# Run windowed, no console (what the .bat launcher does)
.venv\Scripts\pythonw.exe app\main.py

# Headless smoke test — transcribe a file with one or more models,
# GPU first then CPU fallback, printing segments. The only verification
# that touches a GPU and a real model.
.venv\Scripts\python.exe app\main.py --selftest audio.wav large-v3-turbo parakeet-tdt-0.6b-v2
# --notes (first arg after --selftest) also exercises the LLM; --notes=<model>
# picks which one, which is how a notes model gets compared repeatably:
.venv\Scripts\python.exe app\main.py --selftest --notes audio.wav
.venv\Scripts\python.exe app\main.py --selftest --notes=gemma-4-12b audio.wav
```

The unit tests cover the pure logic — merging, coalescing, word-gap
splitting, export formats, notes-model resolution and prompt sizing, and the
incremental transcript render. They do not touch a GPU or a model, so
`--selftest` is still what verifies transcription and notes end-to-end;
prefer it over launching the GUI.

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

### Transcript rendering

`_TranscriptPane` (`main_window.py`) keeps each read-only transcript view in
step by **appending**, not by rebuilding: a four-hour session would
otherwise re-lay-out thousands of paragraphs every 400 ms while still
transcribing. Two cases are not appends and are handled rather than hidden —
the last block keeps growing while its speaker talks, so it is redrawn every
refresh; and a segment from another track can sort in ahead of drawn blocks,
which forces a full rebuild. A pane that is hidden (the other tab, or the
mirror toggled off) is skipped and caught up by `_refresh_visible_panes`.
`tests/test_render.py` pins the output to be identical to a full redraw.

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

Models are GGUFs listed in `LLM_REPOS` as `NotesModel` records: repo, exact
filename, an optional pinned revision, and the generation policy for those
weights. The box accepts `repo_id` or `repo_id::filename.gguf`.

**`LLM_REPOS` is deliberately larger than `LLM_GROUPS`.** The dropdown is one
entry per model, at the best quant that comfortably fits; the catalog also
holds a `-q4` smaller build of each of those, the models they superseded, and
the reserve queue. Those stay resolvable by their short alias and out of the
dropdown — the dropdown was what got too long (34 entries, a third of them a
second quant of the line above), not the catalog, and a superseded model is
the fallback if the challenger that replaced it turns out not to load. Note
that the naming convention **inverted** when the duplicates were cut: a bare
name is now whatever quant that entry ships, and `-q4` is the smaller
fallback. `tests/test_llm.py` pins both halves — no two dropdown entries may
share a repo, and every `-q4` must point at its base entry's repo.

Five things about that module are easy to undo by accident:

- **Policy follows the resolved model, not the typed name.**
  `_apply_identity_policy` matches `repo/filename` against
  `_IDENTITY_POLICIES` and fills in whatever the catalog entry left unset,
  so `unsloth/Qwen3-14B-GGUF::Qwen3-14B-Q5_K_M.gguf` gets thinking switched
  off exactly like the `qwen3-14b` alias does. Matching on the alias alone
  is what let a typed path silently run a thinking model against a
  non-thinking one.
- **Quant choice is per model.** The old "large models must stay Q4" rule
  was wrong: at the default 23,000-token context with Q8_0 K/V caches, a 12B
  at Q6_K budgets ~12–13 GiB. Only Q8 of a 12B is genuinely tight — which is
  why `gemma-4-12b` is Q6 and `qwen3-14b` is Q5, not the largest quant each
  repo offers. "Best quant" means the best one that still leaves margin.
- **The context window is per model, and only ever from measurement.**
  `DEFAULT_CONTEXT_SIZE` (23,000) is what fits the *tightest* catalog entry,
  so raising it globally would bill every model for the room only some have.
  `NotesModel.context` opts one entry into more; `_input_tokens` and
  `_part_tokens` derive the prompt and part budgets from it, so a bigger
  window means fewer, larger map-reduce parts and fewer lossy merges — a
  measured 7 parts down to 2 on the same transcript. What it costs depends
  on how many layers hold **full** attention, not on parameter count: going
  23,000 → 49,152 cost LFM2.5 and Gemma 4 under 400 MiB, while a
  fully-dense-attention model of the same size would pay several times that.
  Set it **per entry, never in `_IDENTITY_POLICIES`** — headroom belongs to
  the quant, and Q6 and Q8 of one model are not the same model here. Measure
  with a full-window prefill before setting it; a model that no longer fits
  does not fail, it drops onto the CPU-offload ladder and crawls.
- **`qwen3-4b` stays although `qwen3.5-4b` supersedes it.** It is
  `DEFAULT_LLM` and the notes model with the longest working history here. A
  default has to be something known to load; swapping it for a model nobody
  has run is the one change most likely to break a fresh install.
- **A reasoning model's hidden trace is discarded but still charged.**
  `REASONING_EXTRA_TOKENS` buys gpt-oss a bigger completion allowance and
  takes it back out of the prompt budget, so the window — and the KV cache
  the server allocates for it — never moves. That stays true whatever
  `NotesModel.context` says: the window is the model's, and the completion
  cap is carved out of it, never added to it.
- **Not every reasoning model can be talked out of it.** LFM2.5's chat
  template has no thinking switch — `preserve_thinking` only governs
  history — so `--reasoning off` and an `enable_thinking` kwarg are both
  silent no-ops (byte-identical output at temperature 0). Only
  `--reasoning-budget 0` works, and it is a *launch* flag, which is what
  `NotesModel.server_args` and `_launch_attempts` exist for. The server then
  ends the trace with an empty `<think></think>` that arrives as ordinary
  content, which `_LeadingThinkFilter` strips. **Check a new model's
  template before assuming a kwarg will reach it**: the cost of guessing was
  notes truncated before the per-person summary, with 1,572 of 2,500 tokens
  spent on a trace nobody ever saw.
- **`-ngl 999` is a request, not a guarantee.** `_fallback_ladder` retries a
  model that ran out of VRAM, shedding experts (MoE) or layers (dense);
  `_runtime_report` scrapes the server log afterwards so a model quietly
  running half on the CPU is visible instead of merely slow.

Split GGUFs (`…-00001-of-00003.gguf`) download as a whole set —
`hf_hub_download` fetches one file per call, and a third of a model does not
load.

Adding a notes model is a `NotesModel` entry in `LLM_REPOS` plus a name in
`LLM_GROUPS`. Verify the exact filename against the repo's file list rather
than guessing it, and pin `revision` if the model is there to be compared
with something. Only add a `template_kwargs`/`reasoning_effort` field if the
model's family isn't already covered by `_IDENTITY_POLICIES` — the pattern
is what a typed path gets, so teaching the pattern is better than teaching
the one entry. A new family with a thinking switch belongs in the pattern
table, tested in `tests/test_llm.py`; one whose template has *no* switch
needs `server_args` instead. Read the template out of the GGUF rather than
trusting the family's reputation — the metadata header parses without
loading any tensors, and `llama.cpp/llama-server --help` lists the reasoning
flags that are left when the template offers nothing. Leave `context` unset
unless you have loaded that exact file at that exact window and watched a
full-window prefill succeed with room to spare.

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
preloaded into the build). It also owns the user's own content:
`user_data_dir()` and `notes_path()` are LOCALAPPDATA in *both* modes —
unlike `writable_dir()`, which is the project tree when running from source
— and `write_atomic()` is how anything there gets replaced. When adding
files the app reads or writes, route the path through `runtime.py` rather
than hardcoding the project tree.

### Persistence

User settings persist via Qt `QSettings` (registry under the org/app names
set in `main.py`): model/notes-model/language choices, "starred" default
models, vocabulary hints, AI instructions, the Notes tab's two
pane-visibility toggles (`notes_show_instructions`, `notes_show_transcript`),
and dark mode. See `_load_settings`/`_save_settings` in `main_window.py`.

**The notes text itself is not in the registry** — it is a document that
grows without bound, and a registry value is unrecoverable by hand when
something goes wrong. It lives in `runtime.notes_path()`
(`%LOCALAPPDATA%\TransViber TransScriber\notes.md`, in source runs too, so
it never lands in the project tree) and is written through
`runtime.write_atomic`: a temp file plus `os.replace`, because the autosave
fires a few seconds after every keystroke and an interrupted in-place write
would destroy the thing the autosave exists to protect. `_read_notes_text`
migrates an old registry value across once.

### Tests

`tests/` runs offline in seconds: no GPU, no model download, no window.
`tests/__init__.py` puts `app/` on `sys.path` (the app modules import each
other by bare name), so every test module starts with `import tests`.

What is pinned, and why it is worth keeping pinned:

- `test_merger` / `test_transcriber` — the sort order, the coalescing rules
  and `split_on_word_gaps`. These are the timestamp and interleaving logic
  everything downstream assumes; `test_merger` also pins the
  `MAX_COALESCE_CHARS` invariant that `llm.py`'s line-boundary splitting
  depends on, which is otherwise only a comment reaching across two modules.
- `test_exporters` — timestamp formats, VTT escaping, and `target_paths`
  (the "Session 2.5" case, where `with_suffix` would write a file the user
  never named).
- `test_llm` — catalog integrity, model resolution, and the identity policy,
  including that a typed `repo::file.gguf` gets the same policy as its
  alias. Prompt sizing is exercised through a `NotesWorker` subclass whose
  `_try_count_tokens` is a ratio, so the real packing and splitting run
  without a server.
- `test_generation` — a fake llama-server (`http.server` + SSE) that records
  the request and replays a scripted stream. It is the only way to check the
  payload fields (thinking switches, reasoning effort) and the stream fields
  (`reasoning_content`, `finish_reason`, the usage-only final event) without
  a 10 GB model. Extend this rather than mocking `requests`.
- `test_render` — the incremental transcript render, asserted equal to a
  full redraw. Qt runs on the offscreen platform; the module sets
  `QT_QPA_PLATFORM` before importing `QtWidgets`, so leave that import order
  alone.

Nothing here touches a GPU, a model or the network, and it should stay that
way — `--selftest` is where real inference gets verified.

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
