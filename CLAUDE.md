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

- **Transcribe** — control bar (files, Align…, model + ☆ star, language,
  device, elapsed label, Transcribe/Export) over a horizontal splitter: the
  full-height **Vocabulary hints** panel on the left (shown only when
  `engine_for(model)` is whisper or qwen), and on the right the optional
  session-reference row, the track table and the live transcript.
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
`TranscribeWorker` (`transcriber.py`), `NotesWorker` (`llm.py`),
`AlignWorker` (`align.py`), and `StatsPoller` (`stats.py`, the
GPU/VRAM/CPU/RAM status readout via NVML + psutil). `MainWindow` connects
to their signals and updates widgets. A one-shot answer can instead stay on
the worker for the `finished` handler to read after `wait()`: `AlignWorker`
leaves `result`, `error`, `cancelled` and `fingerprints` there.

`DeviceProbe` checks CUDA through the driver API in a background thread
after the window opens. Do not import CTranslate2/PyTorch to detect the GPU
at startup: CTranslate2's package import loads PyTorch too. All inference
engines and NumPy stay unloaded until needed. The source `.bat` and
`pythonw` launches also log click/startup timings to `app.log`.

Shutdown is asynchronous: `closeEvent` disables the UI, requests
cancellation, and retries via a timer until every worker has stopped.
Never accept a close while a QThread is running. A job counts as stopped
once its finish handler has run (the window drops its reference there),
not when `isRunning()` turns false: the last notes chunks can still be
queued behind a finished thread. Only then are the notes saved for the
last time, and a failed save keeps the window open behind one prompt
(Retry / Save Elsewhere / Return to App / Close Without Saving); the
stats poller is left running until that is settled, so returning to the
app needs nothing restarted. The finish handlers reset their controls
even while closing, and skip only status text and dialogs. Stats polling has an
interruptible wait; notes startup checks cancellation before/after downloads
and while waiting for the server. Cancel AI Notes kills a stalled stream.
The ASR model stays cached between runs and is released when its model/device
changes, notes starts, or the app closes; cleanup must not import PyTorch
merely to check whether it has anything to release.

### The core merge idea

Each track is transcribed independently with VAD (silence while others talk
is skipped, not hallucinated). Segments are tagged with the track's speaker,
then **all** segments across tracks are sorted by start time.

Tracks don't have to start at the same instant. Segment times inside a
track stay relative to **its own file**; `Track.offset` is when that file
began on the session clock (someone who joined late, or recorded on a
separate device), and `merge_tracks` adds it to *copies* of the segments.
So an offset edit never needs a re-transcription, can't be applied twice,
and reaches the live view, AI notes and every export through that one
function. Keep it there: don't bake an offset into `track.segments`. An
offset is the file's start, not the first word — leading silence in a file
is already correct timing. Offsets are never negative and never inferred
from file length. With every offset at zero (Craig/OBS/REAPER exports, which
pad late joiners), output is identical to before offsets existed.
`OffsetSource` records how each was set (default, typed, matched, guessed);
a typed value, including a typed 0, is the user's and is protected.
`align.py` holds `parse_offset`/`format_offset` for the track table's
editable **Offset** column, and Align (below) only ever proposes values for
it.

`SPLIT_GAP_SECONDS` (1.0s) is load-bearing: Whisper/Qwen can lump utterances
separated by silence into one segment, which would sort a speaker's later
remark at the time of their earlier one and break the interleaving. Segments
are split at word-level silence gaps (`split_on_word_gaps`) to prevent this.

### Decoding on the file's clock

Every engine and Align decode through `audio.py` (`decode_track_audio`,
`iter_stream_power`), and both follow the file's own timestamps instead of
laying decoded audio end to end. Two clocks, kept apart: the **source
clock**, one 0:00 for every stream in a file (`container.start_time`; a
file that declares none, like WAV, is measured from its earliest packet
timestamp), and the **session clock**, which `Track.offset` maps the file
onto in `merge_tracks`. A stream that began after the others starts with
that much silence, a gap in the timestamps (a paused recorder, a lost
packet) stays a gap, and a file whose clock starts at 10 s starts at 0:00
with no padding. Laid end to end, which is what faster-whisper's own decoder
does (so Whisper is handed the decoded array, never the path), a late
stream lands at 0:00 and everything after a gap moves earlier, and no single
offset can repair that: each stream and each stretch would need its own.
Align summed a file's streams from sample 0 too, and a file whose second
stream began a minute late matched strongly, every part agreeing, a minute
out.

The rules, in `_timed_blocks` and `_placed`:

- A frame within `TIMING_TOLERANCE` (20 ms) of where the audio before it
  ended is rounding (Matroska keeps milliseconds) and follows on. Beyond
  it, the resampler is flushed and started again, so it never blends audio
  across a gap, and the new run is placed by its exact timestamp. The
  per-frame check is in floats: Fractions on every frame cost Align a
  tenth more time reading a file.
- Later is silence. Earlier, including before 0:00, plays on from where the
  audio had got to; every later timestamp is read that much later, as a
  clock that restarted (two recordings joined end to end); and the jump is
  noted. Audio is never dropped, so a damaged clock costs timing, not words.
- No timestamp: the frame follows on, and a run that began without one ends
  at the first frame that has one.
- A packet that won't decode is skipped and counted, its place left silent
  by the timestamps after it; only a stream where nothing decodes fails.
  faster-whisper's decoder stopped at the first one, cutting the file short.
- Silence is only made up to `MAX_TIMELINE_SECONDS` (24 h) into a file;
  past that, `AudioFileError`, before anything is allocated for it. Align
  gets a long gap as zero power, `SILENCE_CHUNK_SECONDS` at a time, so it
  stays cancellable.
- `AudioFileError` is the file's own problem: `TranscribeWorker.run` fails
  that track without the GPU-to-CPU retry, which would read it no better
  and leave every later track on the CPU.
- What decoding ran into (`Timing`: jumps back, damaged packets) reaches the
  track as `Track.note`, shown as "Done ⚠" with the note as its tooltip.

WMA's decoder stamps its first frame 86 ms in; that is followed like any
other timestamp. `decode_track_audio` writes into one array sized from the
stream's declared duration rather than gathering chunks and joining them:
a four-hour track at 16 kHz is nearly a gigabyte, and the join held two.
On an hour of 48 kHz AAC that halved peak memory and cut decoding time by
about 40%; Align's read took the same time as before.

### Track alignment

**Align…** suggests offsets from sound the recordings share, and applies
only what the user ticks in its review window (`AlignReviewDialog`). Nothing
is ever applied automatically; a typed value is the undo.

- **Evidence.** `AlignWorker` decodes each stream once through
  `audio.iter_stream_power` at 8 kHz, on the file's clock (a file is one
  unit: all its rows, their streams' energies summed where each began),
  keeping only a 100-per-second *onset
  fingerprint*: the rises in log frame energy, less a 1 s moving average.
  Power is summed over every channel, not taken from a mono downmix, which
  gives an LFE-labelled channel (the 4th of 5.1) no weight — in a multitrack
  file that channel can be someone's microphone. `find_offset` slides one
  fingerprint along another with FFT normalised cross-correlation. The plan
  specified loudness itself; rises replaced it because an isolated speaker
  file and a mix share that speaker's syllable onsets while other voices
  fill the mix, and loudness matched with its peak barely clear of its own
  neighbourhood.
- **Reuse.** Decoding is most of Align's time (about 20 s per 4-hour
  file), so the window keeps each fingerprint from the last run, keyed by
  path and streams and stamped with the file's modified time and size. A
  file unchanged since is not decoded again, so pressing Align a second
  time costs only the matching. Only files loaded when Align is pressed
  keep theirs, which bounds it at about 11.5 MB per 4-hour file. Within one
  `find_offset`, `_Transforms` works out the reference's FFT once per size:
  the whole file, each part and each stretch all slide along the same
  reference, and that took the reference's FFTs from 12 to 2 per match.
- **Trust.** The label comes from peak ÷ best score more than 0.5 s away:
  strong ≥ `MATCH_RATIO` (ticked), weak ≥ `WEAK_RATIO` (unticked), else no
  match. Lags are only scored where the files overlap by
  `MIN_OVERLAP_SECONDS` (120 s, or half the shorter file), and a strong
  label needs `STRONG_MIN_OVERLAP_SECONDS` (240 s, or 90% of the shorter
  file). The plan said 30 s; at 30–70 s of overlap unrelated files threw up
  an occasional lone "strong" peak. Each file also needs
  `MIN_ACTIVE_SECONDS` of sound inside the overlap: two gated headsets with a
  few words each scored strong on chance coincidences alone. A second counts
  as sound above both a tenth of the busiest seconds and `SOUND_ENERGY`, so
  a file of hiss or digital silence has none, however loud its "busiest"
  seconds. A strong match is then searched for again in parts of the
  overlap (`_part_verdict`): the first and last third, sixth, twelfth and
  80 s, each part on its own. Shrinking parts towards each end is what
  reaches a cut near either end, where the bigger part straddling it mostly
  agrees; one within about 40 s of an end is still missed (surveyed: 60 s or
  more all caught, 30 s or less missed). The fractions only reach 80 s at
  240 s of overlap, but the two 80 s parts are checked from 160 s, where
  they no longer overlap (`_part_sizes`): before that, a three-minute file
  with a cut in it passed as strong unchecked. Under 160 s of overlap — a
  strong match there needs a file under about three minutes — nothing checks
  for an edit; that limit is pinned in `test_align`. Surveyed on 400 short
  synthetic sessions (170–270 s files), the 80 s parts downgraded no
  unedited match; of files with a 1–30 s cut, the share still strong
  without a session reference fell from about half to a third against a mix
  and a fifth in a shared room. A part is only scored at lags where
  nearly all of it lies inside the reference, since a sliver of overlap at
  the edge throws up chance peaks, and only once it holds
  `MIN_ACTIVE_SECONDS` of sound itself. It downgrades the match when its
  clear peak (≥ `WEAK_RATIO`) lands more than 0.1 s away, or when it has two
  real peaks, neither clear of the other but both clear of everything else,
  with the whole-file lag one of them: that is a part straddling a cut. A
  part whose only standout is a chance peak, with the whole-file lag no
  better than the rest, contradicts nothing. Against a session reference a
  strong match also needs two separate stretches of the file's *active*
  audio (20 s of it each, so silence never counts) to agree through the
  same `_part_verdict`; a stretch that disagrees downgrades it whatever the
  others say. `suggest_offsets` chains only through strong matches; a weak
  one is kept as that file's suggestion, never replaced by the "stopped
  together" guess and never used to place another file. Guesses start
  unticked, and so does any file with a typed value on any of its rows.
- **Session reference.** `MainWindow.session_reference` is a
  `SessionReference`, deliberately not a `Track`: never transcribed,
  exported, counted or sent to AI Notes. Its start is session 0:00 and
  never moves; a file matched before it gets a reason and no checkbox.
  Without one, the longest speaker file is the reference, anchored on its
  current offset. If the reference or a strong match would land before
  0:00, every row moves later by the same amount; weak rows never drive
  that shift, and one still negative afterwards can't be applied.
  `AlignmentResult.measured` keeps the rows relative to the reference so
  `rebase()` can re-anchor without re-matching: the window does that when
  the reference file's offset was typed while Align ran. When the clock
  moves (a shift, or a file that starts ticked would move from an offset it
  already has) and a file with an offset of its own (typed, or from an
  earlier Align) would start unticked, nothing starts ticked, since applying
  the others alone would leave that file behind on the old clock. An
  automatic reference that doesn't move keeps its value, but turns MATCHED
  from DEFAULT once others are applied against it, so the next run counts
  it as placed. The same canonical path can't hold both roles (checked in
  both import routes).
- **Busy state.** `_tracks_locked()` covers transcription, alignment and
  the review window: Add/Remove/Clear/split are disabled while any of them
  runs, and the reference controls from Align until the review closes.
  Offset edits, Transcribe and AI Notes stay available during Align. The
  review waits for any other modal prompt to close first (Apply under an
  open Export dialog would change what it is about to write), commits an
  Offset still being typed before reading "Now", and Export merges again
  right before each write.
- **Calibration status.** The thresholds are provisional. On synthetic
  energy sessions true matches scored about 3.3–24 and 1,200 unrelated
  pairs at most 1.89 (some weak, none strong). Over about 12,000 unrelated
  pairs the ratio alone reached 2.3, always at about 120 s of overlap: it is
  the 240 s strong-overlap rule, not `MATCH_RATIO`, that keeps those weak,
  so recalibrate the two together. On espeak-synthesised
  four-speaker speech (compressed mix, late joiners, an early leaver),
  isolated files against the mix scored 3.7–12.4 with every part agreeing,
  same-room bleed 7.4–8.2 and unrelated headset files 1.08–1.25. None of
  that is a real recording: calibrate on real sessions, mix-versus-isolated
  kept separate from shared-room, before trusting them.

### Three ASR engines behind one interface

`engine_for(model_name)` in `transcriber.py` routes each model to one of:
- **whisper** — faster-whisper / CTranslate2 (default; the `.transcribe`
  path with `word_timestamps`, `hotwords`, VAD), given the array from
  `decode_track_audio` like the others, never the file's path.
- **onnx** — onnx-asr / ONNX Runtime for NVIDIA Parakeet & Canary
  (`_transcribe_track_onnx`). Ignores the Language dropdown (auto-detect).
- **qwen** — Qwen3-ASR via the `qwen-asr` package + a forced aligner for
  word timestamps (`_transcribe_track_qwen`). **Critical constraint:** Qwen
  must never be fed silence — it hallucinates repeated text into gaps and
  the aligner smears it across the silence, corrupting timestamps. The code
  cuts audio into VAD speech regions, detects language once from the
  clearest sample and locks it for the whole track, then drops segments
  written in a script the locked language can't use (`_foreign_script`).
  Its languages are a closed list: `QWEN_LANGUAGE_NAMES` holds only names
  qwen-asr accepts (0.0.6 has no Norwegian or Ukrainian), and
  `language_supported` is checked twice. The window greys out what the
  chosen model lacks and refuses a selection it already had, never turning
  it into auto-detect; `TranscribeWorker.run` refuses it before any model
  loads, so the error can't reach the GPU-to-CPU retry.

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
pass reduces in rounds when the per-part notes don't fit one prompt. What a
prompt carries besides its content — the instructions, the wrapper text, the
system message and the chat template — is measured once a run by having the
server count the whole request (`/v1/chat/completions/input_tokens`), and
taken off the input limit (`_budgets`). That request comes from the same
`_request` builder as generation, because a thinking switch or reasoning
effort changes what the template renders. Instructions that leave less than
`MIN_PROMPT_TOKENS` of room fail the run with a message saying to shorten
them, before anything is generated: the minimum is a check, never a floor —
granting it anyway once sent prompts bigger than the whole window.
`CHARS_PER_TOKEN` is only a starting guess and a fallback for when
`/tokenize` is unreachable, as `TEMPLATE_OVERHEAD_TOKENS` is for the request
count — don't reintroduce either as the sizing rule; an oversized prompt is
truncated by llama-server silently. A run sized by either fallback says so.

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
Whatever loads them has to be a CUDA 12 build: torch is `+cu128` from
PyTorch's index, and onnxruntime-gpu comes from Microsoft's CUDA 12 feed,
because PyPI's is built for CUDA 13 from 1.27 on. `pip freeze` records
neither source, so the install order and the onnxruntime wheel's hash live
in the header of `requirements.lock.txt`; change it and the README's setup
steps together.

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
migrates an old registry value across once. `_notes_dirty` is set by every
change, typed or streamed, and cleared only by a write that succeeded, so
an untouched document is never rewritten and a failed write is never
forgotten. A file that could not be read is never written over: notes
added after that can only be saved elsewhere, or knowingly dropped.

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
  `test_transcriber.Decoding` pins what the worker hands a model (Whisper
  gets the decoded array) and that a file problem never moves to the CPU.
  `test_transcriber` also checks Qwen's language list against the installed
  qwen-asr's, read from its source rather than imported (that loads torch).
- `test_audio` — decoding on the file's clock, on files it writes with PyAV:
  a gap, a stream that began late, a clock that starts at 10 s, one channel
  against the whole stream, frame sizes and rates, an ordinary WAV sample
  for sample as before, two MPEG-TS recordings joined end to end, the
  timeline limit; and `_placed`'s rules on bare positions, with damaged
  packets and missing timestamps through a stand-in container.
- `test_exporters` — timestamp formats, VTT escaping, and `target_paths`
  (the "Session 2.5" case, where `with_suffix` would write a file the user
  never named). Also that every export replaces each file whole or not at
  all: all of them are staged beside their targets before any is renamed
  over one, and a rename that fails raises an `ExportError` (an `OSError`)
  naming the files already new and the ones still old. Five renames are
  not one transaction; don't let a docstring claim they are.
- `test_merger.Offsets` / `test_align` — the offset applied once, to copies,
  with all-zero offsets giving exactly the old order; and every input form
  the Offset column accepts or rejects, with its reason. `test_align` also
  pins the matcher on synthetic sessions (lag direction, what is never
  labelled strong, window agreement, the drift check), the chaining and
  session-reference rules through a scripted matcher, the worker's failure
  and cancel paths, and real PyAV decodes of files it writes itself,
  including a two-stream MKV whose second stream began a minute late.
- `test_llm` — catalog integrity, model resolution, and the identity policy,
  including that a typed `repo::file.gguf` gets the same policy as its
  alias. Prompt sizing is exercised through a `NotesWorker` subclass whose
  `_try_count_tokens` is a ratio and whose `_count_request` adds a template
  to it, so the real packing, splitting, budgets and `run()` go without a
  server.
- `test_generation` — a fake llama-server (`http.server` + SSE) that records
  the request and replays a scripted stream. It is the only way to check the
  payload fields (thinking switches, reasoning effort), the stream fields
  (`reasoning_content`, `finish_reason`, the usage-only final event) and
  what every prompt of a whole run costs, counted with the request's own
  template policy, without a 10 GB model. Extend this rather than mocking
  `requests`.
- `test_render` — the incremental transcript render, asserted equal to a
  full redraw. Qt runs on the offscreen platform; the module sets
  `QT_QPA_PLATFORM` before importing `QtWidgets`, so leave that import order
  alone.
- `test_lifecycle` — startup, clearing, caching and shutdown on a real
  window (offscreen), plus the Offset column driven through its real cell
  editor. Typed edits commit through `_OffsetDelegate`, not `itemChanged`:
  Qt drops a write of unchanged text, and retyping the shown `0:00` is a
  deliberate "starts at zero" that must still count as typed. The export
  warning is driven through `export_dialog`: a track that failed, was
  cancelled or never ran counts whether or not it produced a segment, and
  a finished track with none (silent audio) does not. `Alignment`
  runs the real `AlignWorker` thread with `run()` scripted and plays the
  review window by replacing only its `exec()`. Each test deletes its
  window in cleanup rather than leaving it to the garbage collector:
  collecting a `MainWindow` crashes PySide6 6.11.1 (the pinned version)
  with an access violation partway through a later test. `NotesOnClose`
  streams AI notes from a real thread to pin the close-time save: a
  thread that has stopped with its last chunks still queued must not let
  the window close before they are in the file.

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
  channel; stereo stays one speaker unless split. Every track, a sub-track
  (`Track.is_subtrack`) or a whole file, is decoded on demand by
  `decode_track_audio`.
