# PC → Mac port notes (July 2026)

For the Claude working on the Mac edition of TransViber TransScriber.
Everything below was built and tested on the Windows sibling after the Mac
zip was made. Apply these as **surgical edits to the existing Mac files** —
do NOT regenerate files from scratch: the user has modified the Mac app
since (including making it launch like a native app), and none of that may
be clobbered. Read the existing Mac code first; it already has features the
PC version ported FROM it (elapsed timer, stats readout) — skip anything
that already exists.

Priorities: **§1 is a real bug fix — do it first.** §2 is the big feature.
§3–§6 are small. If usage is tight, stop after §2.

---

## 1. BUG FIX: Qwen3-ASR timestamp corruption on sparse tracks

**Symptom** (user-confirmed on both platforms): Qwen segments drift/misplace
on short or sparse conversations; dense long ones look fine; whisper/parakeet
unaffected.

**Root cause** (measured with known-offset fixtures): Qwen3-ASR hallucinates
repeated text when the input mixes speech with long silences (pure silence
returns empty, but sparse speaker tracks are speech+silence), and the forced
aligner then smears the phantom words evenly across the silent stretches. A
60 s fixture with 3 utterances came back with 7 "utterances".

**Fix**: never feed Qwen silence. In the Qwen transcribe path:

1. Run silero VAD over the decoded 16 kHz mono audio to get speech regions.
   The PC uses `faster_whisper.vad.get_speech_timestamps(audio,
   vad_options=VadOptions(max_speech_duration_s=240, speech_pad_ms=400))`
   (returns `[{"start": sample, "end": sample}, ...]`; <2 s silences stay
   inside a region, longer ones split, regions capped under the aligner's
   5-min window). On the Mac use whatever silero VAD is already available
   (faster-whisper is installed there too).
2. Transcribe **only the region slices** — batch several per call if the
   runtime accepts a list; otherwise loop.
3. Add each region's start offset (`region_start_sample / 16000`) to every
   word timestamp that comes back, then do the existing 1.0 s gap-split
   grouping per region.
4. Empty-text results for a region are skipped; check the cancel flag
   between regions; per-region progress = `region_end / len(audio)`.

**Verification used on PC** (worth reproducing): build a 60 s wav with a
~3 s TTS clip at 5/25/50 s and a 20-min one with the clip every 60 s;
utterance count must be exact and worst timestamp error well under 0.5 s
(PC result after fix: 0.06 s on both Qwen sizes).

## 1b. BUG FIX: Qwen "喂。" Chinese-filler hallucinations (follow-up to §1)

**Symptom**: with Language on Auto, English tracks come back with scattered
`喂。` (and similar Chinese filler) segments. Cause is two-layered:

1. Auto-detecting each short VAD region independently is unreliable —
   breaths/"mm" regions get detected as Chinese and transcribed as Chinese
   filler. **Fix: probe once per track, lock the language.** Before the
   batched region loop, concatenate the longest VAD regions up to ~30 s
   (skip sub-1 s regions once there's at least one part), transcribe that
   probe without timestamps, take `result.language.split(",", 1)[0]`, and if
   it's in the model's supported list pass it as the forced `language` for
   every region batch.
2. Even with a forced language, Qwen still hallucinates Chinese filler on
   noise-only regions (qwen-asr's `parse_asr_output` returns forced-language
   output verbatim, no checks). **Fix: script filter on emitted segments.**
   When a track language is locked (user-picked or probed), drop any segment
   whose alphabetic characters are majority-written in a script that
   language can't use (`_foreign_script()` in the PC `transcriber.py`:
   Unicode-range table for latin/cjk/kana/hangul/cyrillic/arabic/greek/
   thai/devanagari + a language→allowed-scripts dict; unlisted languages
   are Latin). `喂。` under an English lock → dropped; genuine Chinese under
   a Chinese lock → kept; no lock (probe failed) → old behavior.

Copy `QWEN_LANGUAGE_SCRIPTS`, `_SCRIPT_RANGES`, and `_foreign_script()`
verbatim from the PC `app/transcriber.py` and apply the check in both emit
paths (aligner-words path per group, and the text-only fallback).

---

## 2. FEATURE: Tabs + Notes + local AI notes

### 2a. Tab layout

Central widget becomes a `QTabWidget`:
- **Transcribe** tab = the existing UI, with the vocabulary-hints field
  turned into a full-height panel on the left (see §4).
- **Notes** tab = vertical `QSplitter`. Top half: a horizontal splitter with
  the full-height **AI instructions** panel on the left and, on the right, a
  toolbar row + `QPlainTextEdit` notes editor. Bottom half (full width): a
  read-only `QTextBrowser` mirroring the merged-transcript HTML (updated in
  the same `_render_transcript()` — preserve the notes view's own scroll
  position, not bottom-follow).
- Two checkboxes at the left of the notes toolbar ("Show: Instructions /
  Transcript") toggle the visibility of those two panels so the editor can
  use the full window; both states persist (QSettings keys
  `notes_show_instructions`, `notes_show_transcript`). Hidden instructions
  are still sent to the model.

Notes editor behavior:
- Autosave: `QSettings` key `"notes"`, written by a 3 s single-shot QTimer
  restarted on `textChanged`, plus on close. Loaded at startup.
- **Export Notes…** button → save-file dialog (.md/.txt), writes plain text.
- `clear_tracks` clears both transcript views but NOT the notes text.

### 2b. AI notes (local LLM writes notes on the transcript)

New module (PC: `app/llm.py`) with a `NotesWorker(QThread)`:

- Signals: `status(str)`, `chunk(str)` (streamed text), `progress(int, int)`
  (percent 0-100, cumulative tokens generated), `completed()`, `failed(str)`.
- Input: the merged transcript as text (PC uses `exporters.to_text(merge_tracks(tracks))`),
  optional user instructions, model name.
- Default prompt (keep wording — user-approved):

  > Write concise, well-organized Markdown notes on this conversation
  > transcript: a short summary, key points and decisions, memorable
  > moments or quotes with speaker attribution, and open questions or
  > follow-ups. Use the speakers' names. Finish with a 'Per-person summary'
  > section: for each speaker (and any character they play), a full-session
  > summary of what they did, said, decided, and what happened to them.

  User instructions are **appended** ("Additional instructions: …"), never
  replacing the default.
- Long transcripts (> ~20k tokens): split into ~14k-token parts at line
  boundaries → notes per part (not streamed) → one final combining pass
  (streamed). `passes_total = parts + 1` drives the percent.
- Output streams into the notes editor, appended under a
  `## AI Notes — YYYY-MM-DD HH:MM` header (blank lines before it if the
  editor isn't empty). Cancellation appends `*(cancelled)*`.

**Mac runtime**: the PC no longer uses HF `transformers` for this — it was
10-20x too slow on Windows and was replaced with the official **llama.cpp
CUDA server binary** (`llama-cpp/llama-server.exe`, spawned per run with
`-ngl 999 -c 23000 -fa on -ctk/-ctv q8_0 --no-webui`, GGUF weights fetched
with `hf_hub_download`, generation streamed as SSE from
`/v1/chat/completions`, exact token counts from `/tokenize`, process killed
to free VRAM). None of that applies on Apple silicon: prefer **mlx-lm** with
mlx-community 4-bit models (streaming via `stream_generate`). Feature
contract that must hold: token-streamed output, cancel support, and the
memory rule below.

Model picker (editable combo in the notes toolbar, persisted in QSettings
along with the instructions text). The PC list is organized in **tiers**
(see §2e for the grouped-dropdown helper) — on Mac choose equivalent
mlx-community variants at similar sizes:

- Max Detail (Large Tier): qwen3-14b, gemma-3-12b, mistral-nemo-12b
- Standard Speed (Medium Tier): qwen3-8b, qwen3-8b-q8, llama-3.1-8b,
  llama-3.1-8b-q8, ministral-8b, ministral-8b-q8
- Quick Review (Small Tier): qwen3-4b (default), qwen3-4b-q8, llama-3.2-3b,
  llama-3.2-3b-q8, phi-4-mini, phi-4-mini-q8, granite-3.3-2b,
  granite-3.3-2b-q8

(On PC every entry is a GGUF: Q4_K_M by default, and a `-q8` suffix selects
the Q8_0 build of the same model — better quality at ~2x the download and
VRAM. Q8 is offered only for the small/medium tiers; a Q8 12-14B doesn't fit
16 GB alongside the context. On Mac, offer the equivalent mlx-community
4-bit/8-bit pairs if they exist, otherwise just the 4-bit ones. Qwen3-8b/14b
are hybrid thinking models and run with thinking disabled — on PC that is
`chat_template_kwargs {"enable_thinking": false}` in the request; with mlx-lm
it's `enable_thinking=False` in `apply_chat_template`.)

### 2e. Grouped model dropdowns (both tabs)

Both model combos (transcription AND notes) are populated by a shared
helper that inserts **bold, non-selectable tier headings** with separators
between groups:

```python
def _add_grouped_models(combo, groups):  # groups: list[(label, [names])]
    first_model_index = -1
    for group_number, (label, names) in enumerate(groups):
        if group_number:
            combo.insertSeparator(combo.count())
        combo.addItem(label)
        header = combo.model().item(combo.count() - 1)
        if header is not None:
            header.setEnabled(False)
            header.setSelectable(False)
            font = header.font(); font.setBold(True); header.setFont(font)
        for name in names:
            combo.addItem(name)
            if first_model_index < 0:
                first_model_index = combo.count() - 1
    if first_model_index >= 0:
        combo.setCurrentIndex(first_model_index)
```

Transcription tiers on PC (`MODEL_GROUPS` — the PC edition used to derive a
flat `MODELS` list from it "for tests", but nothing ever read it and there
is no test suite, so it was dropped; don't add one on the Mac side either):

- Max Detail (Large Tier): qwen3-asr-1.7b, large-v3, large-v2
- Standard Speed (Medium Tier): canary-1b-v2, large-v3-turbo, medium,
  medium.en
- Quick Review / Rapid Capture (Small Tier): distil-large-v3.5-ct2,
  distil-large-v3, qwen3-asr-0.6b, parakeet-tdt-0.6b-v3,
  parakeet-tdt-0.6b-v2

(`distil-large-v3.5-ct2` is a friendly alias mapped to
`distil-whisper/distil-large-v3.5-ct2` in a `WHISPER_ALIASES` dict checked
before handing the name to faster-whisper.)

### 2f. Model starring (default-model pin)

A flat ☆/★ `QPushButton` (fixed ~28 px) sits immediately after each model
combo. Behavior:

- Clicking stars the combo's current text: saved to QSettings
  (`"starred_model"` for transcription, `"starred_notes_model"` for notes).
  Clicking again on the starred model unstars (removes the key).
- The glyph refreshes on every `currentTextChanged`: ★ only when the
  current selection equals the starred value, ☆ otherwise.
- At startup the starred value **beats the last-used value**:
  `settings.value("starred_model") or settings.value("model")` (same for
  notes). Unstarred → last-used behavior as before.
- One shared factory method builds both buttons:
  `_make_star_button(combo, settings_key)` containing local `refresh()` and
  `toggle()` closures wired to `clicked` / `currentTextChanged`.

### 2c. Memory exclusivity (user requirement — must hold)

The ASR models and the notes LLM must never be loaded at the same time:
- Starting AI notes frees the cached ASR model first (PC:
  `transcriber.free_asr_models()` — cache=None, gc, empty_cache).
- Starting a transcription frees the LLM (PC: `llm.free_llm()`).
- The **AI Notes button is disabled while transcribing**, and **Transcribe
  is disabled while notes are generating**. (On unified-memory Macs this is
  less critical but keep the same behavior for consistency.)

### 2d. Notes toolbar extras

- `QProgressBar` (fixed ~170 px, hidden when idle) fed by the progress
  signal; format string shows tokens: `%p% · {tokens:,} tok`.
- Elapsed label + 1 s QTimer: during generation it ticks
  `m:ss · N tok · R tok/s`; when the run ends the label is **cleared** (it
  is a live readout only) and the final numbers go to the status bar,
  which keeps them until something else writes there:
  `AI notes done in 1:42 · 1,204 tok · 76.3 tok/s avg`.
- The AI Notes button becomes "Cancel AI Notes" while a run is in flight,
  and the notes-model combo is disabled for the duration.

---

## 3. Download progress percentages in the status bar

During model downloads the status message shows live percent + size
("Downloading model 'x' — 45% of 4.9 GB…"). Implementation: a context
manager swaps a reporting tqdm subclass into huggingface_hub while a model
loads. Gotchas discovered on PC (hf_hub ≥0.36 and 1.x both):

- `huggingface_hub.utils.tqdm` the *module* is shadowed by the re-exported
  *class* — get the module via `importlib.import_module`.
- Patch three bindings: `huggingface_hub.utils.tqdm.tqdm`,
  `huggingface_hub._snapshot_download.hf_tqdm`,
  `huggingface_hub.utils._xet_progress_reporting.tqdm`, and (for
  faster-whisper downloads) `faster_whisper.utils.disabled_tqdm`.
- In the subclass: force `disable=False`, render into an `io.StringIO()`
  (the app may run with no console), report only from the first unit="B"
  bar in flight (a `state` dict tracks it; release on `close()`), emit on
  whole-percent changes. Restore all bindings in `finally`.
- MLX model downloads may not go through hf_hub the same way — verify
  before assuming it works for mlx repos.

---

## 4. Small UI changes since the zip

- **Dark mode checkbox** in the status bar:
  `QGuiApplication.styleHints().setColorScheme(Qt.ColorScheme.Dark/Light)`,
  persisted in QSettings key `"dark"` only after the user first toggles
  (until then, follow the OS). May be redundant if macOS theme-following is
  already good — user's call.
- Transcription model dropdown: sub-medium Whisper sizes removed; the list
  is the tiered `MODEL_GROUPS` from §2e; `canary-1b-v2` is CC-BY-4.0 (fully
  free incl. commercial — earlier "non-commercial" note was wrong, that was
  the v1 model).
- Vocabulary hints is now a **full-height `QPlainTextEdit` panel** on the
  left of the Transcribe tab (in a horizontal splitter with the track
  table + transcript, initial sizes ~260/840), not a one-line field — long
  name lists stay visible. Terms may be one per line or comma-separated;
  the lines are joined with ", " into the single hotword string the engines
  want (`_hotwords_text()`). It shows for whisper AND qwen engines (qwen
  takes hints as decoder `context`), hidden for parakeet/canary. The AI
  instructions panel on the Notes tab mirrors this shape.
- Errors: if tracks fail, a QMessageBox at the end lists file + error
  (instead of only a hover tooltip).

---

## 5. Explicitly do NOT change

- The user's native-app launch setup on the Mac (app bundle / launcher /
  whatever exists there now).
- The word-gap splitting (1.0 s) and VAD/hallucination guards in all
  engines — they are what keeps speaker interleaving correct.
- The Mac's own engine choices (mlx-whisper, mlx-qwen3-asr, CoreML) — §1
  changes *what audio* the Qwen engine sees, not the engine itself.

## 6. Verification checklist (cheap)

1. Qwen fix: sparse fixture → exact utterance count, error < 0.5 s.
2. AI notes on a small transcript → contains a "Per-person summary"
   section; progress ends at 100 with a token count; cancel works.
3. Transcribe button disabled during notes generation and vice versa.
4. Notes text survives an app restart.
5. Grouped dropdowns: tier headings are bold and cannot be selected; the
   first real model is selected by default.
6. Starring: star model A, use model B, quit, relaunch → A is selected and
   its star shows ★; unstar → relaunch selects last-used again.
