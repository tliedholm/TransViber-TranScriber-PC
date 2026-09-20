# TransViber TransScriber — Comprehensive Code Audit

**Audit date:** 2026-08-07  
**Scope:** `app/`, release/installer scripts in `packaging/`, launch scripts, project documentation, the existing PyInstaller build log, and the current standalone build contents.  
**Constraint followed:** No application source, configuration, assets, or packaging files were changed. This report is the only file added.

## Executive summary

The application has a sound small-app architecture: audio probing/decoding, ASR, merging, exports, notes generation, and the UI are separated into modules; long work is moved off the UI thread; transcript HTML is escaped; model VRAM exclusivity is deliberate; and the Qwen sparse-audio workaround shows good empirical engineering.

The largest risks are reliability and data integrity rather than ordinary algorithmic errors. In particular:

1. The window can close while `QThread` workers are still running, which can cause `QThread: Destroyed while thread is still running`, crashes, or undefined shutdown behavior.
2. Failed transcription tracks may retain partial segments that remain mergeable/exportable without being identified as incomplete.
3. Long-transcript AI notes can exceed the configured context during splitting or the final combine pass.
4. Notes are autosaved as one unverified `QSettings`/Windows Registry value, which is not a robust store for a potentially large user document.
5. A frozen startup failure can happen before file logging is initialized, defeating the diagnostic path intended for the windowed build.
6. There is no dependency manifest, lockfile, automated test suite, lint/type configuration, or CI, making the working 7+ GiB package difficult to reproduce safely.

The review records **7 high**, **19 medium**, and **18 low/improvement** findings. “High” means the issue can lose or corrupt user work, crash the app, break a major feature, or substantially undermine release reliability. It does not mean every high-severity issue is frequently triggered.

## Methodology and validation performed

- Read all 11 Python modules (2,498 lines), the PyInstaller spec, release PowerShell, Inno Setup script, launch batch file, README, developer notes, and the PC-to-Mac handoff.
- Parsed all 11 Python modules successfully with the project interpreter (Python 3.14.4).
- Inspected the existing PyInstaller build log and its warning file.
- Measured the existing standalone directory: **7.18 GiB across 5,971 files**.
- Queried the installed versions of the direct runtime/build dependencies.
- Ran in-memory probes (no files written) that confirmed:
  - WebVTT is malformed when a speaker or transcript contains markup characters.
  - A single 100,000-character transcript line is emitted as one notes chunk, bypassing the intended chunk budget.
  - `_selftest([])` raises `IndexError`.
- Did not run full ASR/LLM inference because no representative audio fixture is included and doing so would download/load multi-gigabyte models. No code was changed.

## What is already done well

- Long-running transcription, notes generation, and machine-stat polling are off the main UI thread.
- Worker-to-UI communication uses Qt signals instead of directly touching widgets from worker threads.
- Model lifecycle is centralized and there is an explicit rule preventing ASR and notes models from sharing VRAM.
- The Qwen implementation avoids feeding large silent regions to the model and offsets timestamps back into track time.
- Word-gap splitting addresses a real multi-speaker ordering failure mode rather than merely formatting output.
- Transcript HTML uses `html.escape()` for both speaker and text.
- Model imports are mostly lazy, which is appropriate for a large multi-backend desktop application.
- Frozen/source path differences are centralized in `runtime.py`.
- The notes HTTP server is bound to loopback rather than a network-visible interface.
- Export responsibilities and the shared `Track`/`Segment` data model are sensibly separated from the UI.
- GPU failure has a CPU fallback, and per-track errors are surfaced in a final dialog.

## Severity guide

| Severity | Meaning |
|---|---|
| High | Possible crash, user-data loss/corruption, major feature failure, or materially unreliable releases |
| Medium | Incorrect output in realistic cases, significant inefficiency, misleading behavior, privacy/security weakness, or difficult recovery |
| Low | Edge case, maintainability issue, polish problem, or incremental improvement |

---

## High-severity findings

### H-01 — Shutdown accepts the close even when workers have not stopped

**Locations:** `app/main_window.py:948-959`, `app/transcriber.py:353-355`, `app/llm.py:226-227`

`closeEvent()` requests cancellation, waits only 5 seconds for ASR and 10 seconds for notes, and then always accepts the event. Model loading, downloads, Qwen batches, ONNX recognition, and HTTP reads can all take longer. A live `QThread` can therefore still reference `MainWindow`, its tracks, or Qt signals while Qt destroys the owning objects.

**Impact:** Intermittent exit crashes, `QThread: Destroyed while thread is still running`, incomplete cleanup, or continued background work after the UI disappears.

**Recommendation:** Treat close as an asynchronous shutdown state. Ignore the first close event, disable the UI, cancel workers, terminate the managed llama server to unblock HTTP generation, and accept only after every thread emits `finished`/`isFinished()`. If a hard timeout is necessary, make the escalation explicit and ensure worker objects outlive the UI.

### H-02 — Failed tracks can leave partial transcript content exportable

**Locations:** `app/transcriber.py:574-589`, `app/main_window.py:674-700`, `app/main_window.py:720-732`, `app/main_window.py:768-789`

Segments are emitted incrementally. If a track later fails on CPU, `_on_track_failed()` changes only its status and error; it does not discard or quarantine segments already emitted during that failed attempt. The export button is enabled whenever any merged segment exists, including segments from `ERROR` or `CANCELLED` tracks.

GPU-to-CPU retries usually clear prior segments because `_on_track_started()` is emitted again, but a final CPU failure can still leave the second attempt's partial output. Cancellation intentionally retains partial output too, yet exports contain no incomplete marker.

**Impact:** A transcript can look complete while silently omitting the tail of one or more speakers.

**Recommendation:** Build each track's result in worker-local staging and commit it only on success, or explicitly mark partial segments/tracks. Require a clear user choice before exporting incomplete results and include per-track status in every export format.

### H-03 — Long-transcript notes can exceed the LLM context

**Locations:** `app/llm.py:76-82`, `app/llm.py:302-317`, `app/llm.py:331-354`

There are two independent overflow paths:

1. `_split_transcript()` only splits at line boundaries. A single line larger than the character budget is accepted whole. The audit probe supplied one 100,000-character line and got one 100,000-character part.
2. The final combine prompt concatenates up to 2,500 generated tokens per part without checking its token count. More than roughly eight full part summaries can exceed `MAX_INPUT_TOKENS`, and instructions/chat-template overhead further reduce the available budget.

**Impact:** Context overflow errors, truncation, missing late-session content, or failed notes for very long recordings—the exact case map-reduce is intended to support.

**Recommendation:** Tokenize the complete prompt, hard-split oversized lines by token/character boundary, reserve space for system/instruction/template tokens, and recursively reduce summaries in bounded batches until the final prompt fits.

### H-04 — Notes persistence uses an unverified Registry value as a document store

**Locations:** `app/main_window.py:378-384`, `app/main_window.py:905-909`, `app/main_window.py:933-944`

The entire notes document is written to `QSettings` after edits and on close. On Windows this is normally backed by the Registry. The code does not call `sync()`, inspect `QSettings.status()`, keep a backup, or write atomically to a normal document file. Large generated notes can grow substantially across appended runs.

**Impact:** Silent autosave failure/truncation or unrecoverable loss of a user's accumulated notes. Registry storage also makes backup, migration, and user inspection unnecessarily difficult.

**Recommendation:** Store notes in a UTF-8 file under `runtime.writable_dir()` using atomic replace and a rotating backup. Keep only lightweight preferences and the notes-file path in `QSettings`; surface save errors in the UI.

### H-05 — Frozen startup logging is initialized after failure-prone imports

**Locations:** `app/main.py:3-9`, `app/main.py:65-87`, `app/main_window.py:35-46`, `app/main_window.py:123`

`main.py` imports `MainWindow` before `_setup_frozen_logging()` runs. That transitively imports NumPy, psutil, PySide, the CUDA setup module, transcription code, and other app modules. Later, `MainWindow.__init__()` probes CTranslate2 CUDA before the window appears. Any import/DLL/startup exception before line 87 occurs while the windowed executable has no console and before `app.log` exists.

**Impact:** The app can fail to launch with no useful diagnostic—the precise scenario the logging function is meant to solve.

**Recommendation:** Initialize frozen logging immediately after importing `sys` (before `MainWindow` and other application modules), then import the UI inside `main()`. Install `sys.excepthook`/Qt exception logging and append or rotate logs instead of always truncating them.

### H-06 — Builds are not reproducible or dependency-auditable

**Locations:** project root and `CLAUDE.md` (“the `.venv` is the source of truth”)

There is no `pyproject.toml`, requirements file, constraints/lock file, environment export, supported-version range, or scripted environment bootstrap. The build is implicitly defined by one mutable local `.venv`.

Current observed versions include Python 3.14.4, PySide6 6.11.1, Torch 2.11.0+cu128, ONNX Runtime GPU 1.27.1, qwen-asr 0.0.6, faster-whisper 1.2.1, and PyInstaller 6.21.0, but none are recorded as source-controlled build inputs.

**Impact:** A clean rebuild may produce different behavior or fail; security/license review cannot reliably identify what shipped; an accidental environment update can alter a release.

**Recommendation:** Add a human-maintained direct-dependency manifest plus a fully pinned lock/constraints file with hashes, document the CUDA/ONNX indexes, and build releases from a fresh scripted environment in CI.

### H-07 — There is no automated regression suite for timestamp/data-integrity logic

**Locations:** entire project; `app/main.py:12-62`

The only automated path is a manual, model-heavy self-test. It reports failure signals but does not assert transcript content, segment count, timestamp error, ordering, language handling, cancellation, fallback, exports, or UI state. No audio fixtures are included. There is no pytest, linter, type checker, or CI configuration; none of pytest/ruff/mypy/pyright is installed in the current environment.

**Impact:** Regressions in the app's most important guarantees—speaker ordering, sparse-track timestamps, fallback, and persistence—can ship undetected.

**Recommendation:** Add fast unit tests for merge/export/splitting/script filtering and deterministic short WAV fixtures for mono, stereo, multichannel, multistream, silence, overlap, and offset cases. Add mocked worker/HTTP tests and a frozen-build smoke matrix for at least one model from each engine.

---

## Medium-severity findings

### M-01 — Every transcription completion discards the ASR cache

**Locations:** `app/transcriber.py:167-187`, `app/main_window.py:702-708`

The model cache is documented as reusable across runs, but `_on_finished()` always calls `free_asr_models()`. Re-running transcription reloads the model even when the user never starts AI notes. `free_asr_models()` also imports Torch just to call `empty_cache()`, even after Whisper/ONNX-only runs.

**Impact:** Avoidable reload latency, UI stalls during cleanup, extra RAM churn, and potentially a large first-time Torch import at the end of an otherwise non-Torch run.

**Recommendation:** Keep the ASR model cached until the model/device changes, AI notes starts, or the app closes. Only invoke Torch cleanup if a Torch/Qwen model was actually loaded, and perform costly cleanup away from the UI thread.

### M-02 — Transcript rendering repeatedly rebuilds and reparses the full document

**Locations:** `app/main_window.py:127-130`, `app/main_window.py:681-684`, `app/main_window.py:736-764`

Every arriving segment schedules a refresh (throttled to 400 ms). Each refresh flattens all tracks, sorts all segments, coalesces all content, builds all HTML, and calls `setHtml()` twice. As a transcript grows this approaches quadratic total work and reparses two full rich-text documents at up to 2.5 times per second.

**Impact:** Increasing CPU use, UI stutter, scroll jumps, and large temporary strings during long sessions.

**Recommendation:** Maintain an incrementally ordered model, append/update only affected blocks, refresh less often for large jobs, and populate the hidden Notes transcript lazily. A model/view implementation would scale better than replacing full HTML.

### M-03 — Qwen/ONNX/subtrack decoding loads entire tracks into RAM with extra copies

**Locations:** `app/audio.py:83-126`, `app/transcriber.py:376-381`, `app/transcriber.py:419`, `app/transcriber.py:546`

Decoded audio is accumulated as many arrays and then concatenated into a second contiguous float32 array. At 16 kHz this is about 230 MB per hour before chunk overhead and peak-copy cost. Qwen/ONNX always use this path; Whisper uses it for channel/stream subtracks.

**Impact:** High peak RAM on multi-hour recordings, allocation failures, and avoidable decode latency.

**Recommendation:** Decode into bounded chunks, a memory-mapped temporary PCM file, or a streaming engine interface. At minimum preallocate from a reliable duration estimate and release probe/batch arrays promptly.

### M-04 — Broad GPU fallback retries non-GPU and deterministic errors on CPU

**Locations:** `app/transcriber.py:356-365`, `app/transcriber.py:574-589`

Every exception on a GPU run is treated as GPU trouble. Corrupt audio, invalid channel metadata, unsupported model IDs, programming errors, network/cache failures, and output-shape bugs can all trigger a costly CPU reload/retry. For Qwen, CPU float32 inference can be exceptionally slow.

**Impact:** Misleading diagnostics, long waits, duplicated work, and masking of real defects.

**Recommendation:** Fall back only for known CUDA/provider/OOM exceptions. Fail fast for decode, validation, model-ID, HTTP, and logic errors; retain exception chaining and engine/device context.

### M-05 — Cancellation is cooperative at very coarse boundaries

**Locations:** `app/transcriber.py:392-399`, `app/transcriber.py:473-494`, `app/transcriber.py:548-555`, `app/llm.py:178-197`, `app/llm.py:267-290`

Cancellation cannot interrupt model downloads, model construction, audio decoding, language probing, a Qwen batch, many ONNX calls, or the five-minute server-start loop. After the user cancels Qwen, the current batch's results are still emitted because the flag is checked only before the batch. The cancel button then becomes disabled, leaving no further UI action.

**Impact:** “Cancel” may appear hung for minutes and can append more output after the request.

**Recommendation:** Check cancellation during decode/probe/result emission and server startup; reduce batch sizes when cancellation latency matters; use cancellable downloads where supported; close/terminate the llama server to interrupt streaming promptly.

### M-06 — The notes server uses a free-port race and no authentication

**Locations:** `app/llm.py:130-133`, `app/llm.py:167-190`, `app/llm.py:231-262`

The app binds a socket to obtain a free port, closes it, then starts the server on that port. Another local process can claim the port in between. Health success alone does not prove the responder is the spawned llama server. Requests contain the complete transcript and have no API key.

**Impact:** Primarily a local-adversary/privacy risk: a malicious local process could race the port and receive transcript content, or another service could cause confusing startup behavior.

**Recommendation:** Use llama-server authentication if supported and pass a random per-run token; verify process identity/readiness beyond HTTP 200; consider letting the child bind port 0 and report the selected port, if supported.

### M-07 — Downloaded model artifacts are not revision/hash pinned

**Locations:** `app/transcriber.py:21-46`, `app/transcriber.py:294-329`, `app/llm.py:33-55`, `app/llm.py:154-165`

Models are resolved from mutable Hugging Face repositories without a commit revision or expected digest. Default GGUFs come from community conversion repositories, and custom editable model IDs broaden the input set further.

**Impact:** Releases/runs are not bit-reproducible; upstream replacement or compromise changes native-parser inputs and model behavior without an app update.

**Recommendation:** Pin default repos to tested revisions, record SHA-256 hashes for bundled/default files, display provenance, and clearly mark custom model IDs as unverified.

### M-08 — Existing build is very large and bundles broad unused dependency surfaces

**Locations:** `packaging/TransViber.spec`, existing `packaging/dist/`

The standalone directory is 7.18 GiB. Largest components measured:

| Component | Approximate size |
|---|---:|
| Torch | 3.98 GiB |
| Preloaded Whisper model | 1.41 GiB |
| llama.cpp runtime | 0.65 GiB |
| ONNX Runtime | 0.37 GiB |
| llvmlite | 0.11 GiB |

The bundle also includes broad packages such as pandas, sklearn, SciPy, and a large Transformers tree. It ships two ~77.5 MiB executables. CUDA 12 libraries from Torch and CUDA 13 libraries for llama.cpp coexist, some necessarily, but the overall collection has not been minimized by a validated allowlist.

**Impact:** Slow builds, downloads, installs, patching, antivirus scans, and a larger vulnerability/licensing surface.

**Recommendation:** Use PyInstaller dependency graph analysis and engine-specific clean-room smoke tests to exclude training/dev/unused modules. Consider separate optional engine packs or a small bootstrapper. Do not remove CUDA files solely by size without testing all three ASR engines and llama.cpp.

### M-09 — PyInstaller warnings are not gated or classified

**Locations:** `packaging/build.log`, `packaging/build/TransViber/warn-TransViber.txt`

The build warning file has roughly 1,200 lines. The build log reports unresolved TensorRT DLLs (`nvonnxparser_10.dll`, `nvinfer_10.dll`) and TBB (`tbb12.dll`), plus many missing optional modules. These may be harmless for the selected providers, but the release process neither filters known-safe warnings nor fails on new relevant ones.

**Impact:** A genuinely missing runtime dependency can be buried in noise and discovered only on another user's PC.

**Recommendation:** Establish a reviewed warning baseline, remove unused provider binaries where safe, fail the build on unexpected warnings, and run the frozen self-test on a clean Windows VM for every supported engine.

### M-10 — Model reuse for AI notes is also intentionally forfeited

**Locations:** `app/main_window.py:858-880`, `app/llm.py:100-117`

The llama server is killed after every notes run. A second notes request with the same model must restart and reload weights, even when no transcription occurs between the runs.

**Impact:** Repeated multi-second/minute startup costs and unnecessary model churn.

**Recommendation:** Keep the server alive while the Notes tab remains active or for a short idle timeout, and kill it immediately when transcription starts. Preserve the existing ASR/LLM exclusivity invariant.

### M-11 — Mixed-model reruns produce misleading export metadata

**Locations:** `app/main_window.py:612-625`, `app/main_window.py:778-788`, `app/exporters.py:67-92`

If some tracks are `DONE` and others failed/pending, clicking Transcribe processes only the non-done tracks. The user can change model or language first. The final transcript can therefore contain outputs from several models/settings, but JSON stores only the model currently selected in the combo.

**Impact:** Incorrect provenance and difficult debugging/comparison.

**Recommendation:** Store model, engine, device, language setting, and run ID on each track/result. Make “retry failed” and “retranscribe all” explicit actions and export per-track provenance.

### M-12 — WebVTT output is not escaped and can be syntactically corrupted

**Locations:** `app/exporters.py:58-64`

Speaker names are inserted directly into `<v ...>` and cue text is inserted as WebVTT payload. The audit probe with speaker `A <B&` and text `x <tag> & y` produced:

```text
<v A <B&>x <tag> & y
```

**Impact:** Invalid cues, unintended WebVTT tags, lost/misrendered transcript text.

**Recommendation:** Apply WebVTT-appropriate escaping to speaker and payload, validate voice names, and add tests for `&`, `<`, `>`, newlines, and `-->`.

### M-13 — Segment timestamps/text are not validated before merge/export

**Locations:** `app/models.py:18-23`, `app/merger.py:8-29`, `app/exporters.py:48-64`, `app/transcriber.py:499-538`

There is no common validation for finite/non-negative timestamps, `end >= start`, timestamps within track duration, empty text, or aligner values outside a Qwen VAD region. Invalid model output flows directly into subtitles.

**Impact:** Negative/malformed cues, overlapping or reversed subtitle times, sorting anomalies, or downstream-player rejection.

**Recommendation:** Normalize at one ingestion boundary: reject NaN/inf, clamp small numerical drift, enforce minimum cue duration where needed, bound Qwen words to their region, and record dropped/adjusted segments.

### M-14 — Qwen silently drops regions if result count differs from batch size

**Locations:** `app/transcriber.py:473-493`

`zip(batch, results)` stops at the shorter iterable and no length check is made. If the library returns fewer results than inputs, unmatched speech regions disappear while progress still advances and the track is marked done.

**Impact:** Silent missing speech.

**Recommendation:** Materialize results, verify one result per input, associate stable indices, and fail/retry individual missing regions rather than silently truncating.

### M-15 — File probing runs synchronously and hides probe failures

**Locations:** `app/audio.py:52-65`, `app/main_window.py:503-512`

Adding files calls `av.open()` on the UI thread. Any exception is swallowed and converted into a generic zero-duration track. Slow network paths or malformed media can freeze the window; invalid/no-audio containers are accepted and fail later with less context.

**Impact:** UI hangs and delayed/misleading errors.

**Recommendation:** Probe in a short-lived worker, distinguish “probe unavailable” from “no audio stream,” preserve the exception, and show per-file validation results before transcription.

### M-16 — Transcript/project work is not recoverable after a crash or restart

**Locations:** `app/main_window.py:120`, track management and shutdown code

Only notes/preferences persist. Loaded tracks, speaker names, segments, statuses, offsets, and transcription provenance live only in memory.

**Impact:** A crash or accidental close after hours of transcription loses the complete session unless the user already exported it.

**Recommendation:** Add an autosaved project/session file with atomic writes and recovery on launch. At minimum autosave the merged JSON after each successful track.

### M-17 — Editable model fields lack validation and capability checks

**Locations:** `app/main_window.py:153-185`, `app/main_window.py:325-343`, `app/transcriber.py:114-125`, `app/llm.py:119-127`

An empty value, group label, misspelling, arbitrary repo, or incompatible model can be sent directly into a backend. Routing is based on name prefixes, not inspected model capability. Notes accepts any repo even though a selected file may be non-chat or incompatible with llama.cpp.

**Impact:** Large unnecessary downloads, confusing backend errors, or a wrong engine choice.

**Recommendation:** Validate non-empty IDs, separate curated selection from an explicit “custom model” flow, show engine/capabilities before download, and require confirmation with estimated size/license for unverified repos.

### M-18 — Local user content and paths are stored/exported without privacy controls

**Locations:** `app/main_window.py:905-944`, `app/exporters.py:71-80`

Notes, AI instructions, and vocabulary hints are stored in plaintext settings. JSON exports include absolute source paths, which commonly expose Windows usernames and folder structure. There is no “clear local data” control or privacy explanation for settings/cache/log locations.

**Impact:** Sensitive meeting notes, names, campaign content, or filesystem identity may persist or be shared unintentionally.

**Recommendation:** Document all storage locations, provide clear/reset controls, omit or relativize source paths by default, and optionally encrypt sensitive persisted notes using Windows data protection.

### M-19 — Installer trust and third-party compliance need a release process

**Locations:** `packaging/installer.iss`, `packaging/README-FIRST.txt:30-34`, bundled models/binaries

The installer is unsigned and users are instructed to bypass SmartScreen. The package redistributes large third-party runtimes, model weights, and conversion artifacts, but the reviewed tree has no consolidated third-party notices/SBOM and no automated license inventory. Some model-specific terms are mentioned only in prose.

**Impact:** Reduced user trust, harder malware/tamper verification, and avoidable redistribution/compliance risk.

**Recommendation:** Sign releases, publish checksums, generate an SBOM and third-party notices, retain exact model/runtime licenses, and have distribution terms reviewed. This is a process recommendation, not a conclusion that the current bundle violates a license.

---

## Low-severity findings and improvements

### L-01 — Self-test argument handling crashes

**Locations:** `app/main.py:12-26`, `app/main.py:84-86`

`--selftest` is only recognized when `len(sys.argv) > 2`; invoking it without a path launches the GUI instead. `_selftest([])` and `--selftest --notes` without a path raise `IndexError` (confirmed by probe).

**Recommendation:** Use `argparse`, require the audio path, validate existence/type, print usage, and return exit code 2 for invalid CLI input.

### L-02 — The self-test can report OK without checking useful output

**Locations:** `app/main.py:32-62`

It considers only `track_failed`/notes `failed` signals. Zero segments, wrong timestamps, wrong count, an empty notes response, or cancellation can still appear successful.

**Recommendation:** Assert at least one segment for a known fixture, valid monotonic timestamps, expected text/count tolerances, and non-empty notes with required sections.

### L-03 — Removing the last transcribed track leaves Export visually enabled

**Locations:** `app/main_window.py:585-601`, `app/main_window.py:740-745`

`remove_selected()` schedules a transcript refresh but neither it nor the empty branch of `_render_transcript()` disables Export. The enabled button then silently does nothing.

**Recommendation:** Centralize action enablement after every track/status mutation.

### L-04 — Blank speaker edits leave UI and data inconsistent

**Locations:** `app/main_window.py:540-550`

For a blank/whitespace edit, `track.speaker` falls back to the old name, but the table cell is not restored. The UI can show blank while rendered/exported output uses the previous name.

**Recommendation:** Restore the normalized value into the cell with signals blocked and validate duplicate/blank names visibly.

### L-05 — Old error tooltips survive a successful retry

**Locations:** `app/main_window.py:619-625`, `app/main_window.py:694-700`

The error string is cleared for a retry but the status item's tooltip is not. After success, hovering can show a stale error.

**Recommendation:** Clear the tooltip when resetting/starting/completing a track.

### L-06 — Cancelled notes are reported as completed

**Locations:** `app/llm.py:356-359`, `app/main_window.py:858-880`

Cancellation appends `*(cancelled)*` but emits progress 100 and `completed`; the status bar says “AI notes done.”

**Recommendation:** Add a cancelled signal/result state, retain actual partial progress, and report “AI notes cancelled.”

### L-07 — A failed notes run leaves an orphan heading

**Locations:** `app/main_window.py:820-835`, `app/main_window.py:882-885`

The `## AI Notes — ...` header is appended before model download/server startup. Failure before the first output leaves an empty heading in the user's document.

**Recommendation:** Stage generated content and append the header on first chunk, or remove the staged header on pre-generation failure.

### L-08 — Notes export does not append the selected extension

**Locations:** `app/main_window.py:887-901`

Unlike transcript export, notes export writes exactly the typed path. Selecting Markdown and entering `notes` creates an extensionless file.

**Recommendation:** Apply the selected `.md`/`.txt` suffix consistently.

### L-09 — `export_all()` is not transactional

**Locations:** `app/exporters.py:104-121`

If writing a later format fails, earlier formats remain and the UI reports only an error. Existing files can be partially replaced across the set.

**Recommendation:** Render first, write temporary sibling files, then atomically replace targets; report any successfully retained files if full transactionality is impractical.

### L-10 — Export formatting does not normalize multi-line/control-character content

**Locations:** `app/exporters.py:32-64`

Speaker/text newlines and unusual control characters can disrupt one-line text records, SRT cue shape, Markdown, or JSON consumers. Markdown speaker names/text can also create unintended formatting.

**Recommendation:** Define format-specific escaping/normalization and test newlines, tabs, nulls, Markdown metacharacters, and Unicode direction controls.

### L-11 — Duplicate-file detection does not canonicalize paths

**Locations:** `app/main_window.py:503-511`

Path equality is used as entered. Relative vs. absolute paths, case differences, symlinks/junctions, or alternate Windows spellings can add the same file twice.

**Recommendation:** Compare normalized resolved paths (with careful handling for inaccessible/network paths) and optionally file identity.

### L-12 — Invalid channel indices silently duplicate the last channel

**Locations:** `app/audio.py:109-120`

`min(channel, data.shape[0] - 1)` converts an out-of-range channel into the final channel rather than raising. Metadata/layout inconsistencies can therefore create mislabeled duplicate speakers.

**Recommendation:** Validate the channel index for every relevant layout and fail with source/stream/channel context.

### L-13 — Coalescing hides arbitrarily long same-speaker pauses

**Locations:** `app/merger.py:15-29`

Consecutive same-speaker segments are joined regardless of time gap. If no other speaker segment lies between them, a multi-minute pause is represented as one paragraph with only the first timestamp.

**Recommendation:** Add a maximum coalescing gap or preserve paragraph breaks/timestamps after long pauses.

### L-14 — Track alignment is assumed but not checked or adjustable

**Locations:** merge architecture and `app/merger.py`

The merge assumes every recording starts at the same instant. This is documented, but there is no per-track offset, waveform alignment, drift correction, or obvious warning in the working UI.

**Recommendation:** Add an offset field and, later, optional cross-correlation/clap alignment and drift diagnostics. Persist offsets in project/export metadata.

### L-15 — Private third-party APIs increase upgrade fragility

**Locations:** `app/transcriber.py:215-291`, `app/llm.py:140`

The app monkey-patches private Hugging Face modules, imports `faster_whisper.vad` internals, and imports private `_hf_download_progress` from another app module. These are practical solutions but likely to break on dependency upgrades.

**Recommendation:** Isolate adapters behind small compatibility modules, pin tested versions, add contract tests, and gracefully disable only progress reporting when private bindings change.

### L-16 — Broad exception swallowing reduces diagnosability

**Locations:** `app/audio.py:20-27`, `app/audio.py:60-65`, `app/stats.py:20-44`, `app/transcriber.py:178-187`, `app/llm.py:109-116`

Several `except Exception` paths silently degrade without logging the exception. Users/developers cannot distinguish unsupported metadata, missing NVML, a driver error, or an actual bug.

**Recommendation:** Catch expected exception types and log unexpected failures with context once, while keeping user-facing degradation friendly.

### L-17 — Resource cleanup is incomplete or implicit

**Locations:** `app/llm.py:168-177`, `app/stats.py:20-30`

The parent-side llama server log handle is not explicitly retained/closed after startup, and NVML is initialized globally without `nvmlShutdown()`.

**Recommendation:** Own these resources in explicit lifecycle objects/context managers and close them deterministically.

### L-18 — Versioning and release provenance are duplicated/manual

**Locations:** `packaging/installer.iss:6-8`, application metadata, release outputs

The installer contains a hard-coded `1.1`; the app exposes no matching version and the build does not embed commit/revision, dependency lock hash, model revisions, or build date in a diagnostics screen.

**Recommendation:** Define one version source, inject it into app/installer/artifact names, and provide a “Copy diagnostics” view with versions, engine providers, paths, and build provenance.

---

## Architecture and maintainability observations

### `main_window.py` is becoming a monolith

At 959 lines, `MainWindow` owns UI construction, settings, file probing, track mutation, transcription orchestration, export, notes orchestration, theme, timing, and shutdown. This makes state transitions difficult to test without constructing the full GUI.

Suggested boundaries:

- `SessionController`: tracks, statuses, run provenance, retry/retranscribe policy.
- `TranscriptModel`: ordered segments, incremental coalescing, project persistence.
- `TranscriptionController` and `NotesController`: worker lifecycle/cancellation.
- Separate Transcribe and Notes tab widgets.
- `SettingsRepository` and `ProjectRepository`: typed persistence and migrations.

### State is represented by nullable worker fields and widget enablement

`worker is not None`, `notes_worker is not None`, track enums, and button states collectively represent app state. This invites impossible combinations during cancellation and shutdown.

Use a small explicit state machine, for example: `IDLE`, `LOADING_ASR`, `TRANSCRIBING`, `CANCELLING_ASR`, `LOADING_LLM`, `GENERATING_NOTES`, `CANCELLING_NOTES`, `CLOSING`. Derive action enablement from that state in one place.

### Data objects lack provenance and invariants

`Segment` and `Track` are mutable dataclasses with no IDs, source run, confidence, engine, model, device, or validation. Adding immutable segment IDs and a `TranscriptionRun` record would make retries, partial results, incremental rendering, project persistence, and trustworthy exports much easier.

### Error handling needs structured context

Errors are generally flattened with `str(exc)`. A structured error should preserve operation, engine, model, requested/actual device, source path/stream/channel, traceback/log reference, and whether retry is safe. User dialogs can remain concise while diagnostics retain the full chain.

## Product/UX opportunities

These are not necessarily bugs, but they would materially improve a transcription workflow:

- Add per-track mute/include toggles and explicit retranscribe-selected/retry-failed actions.
- Let users set track offsets and see overlap/alignment warnings.
- Show an “incomplete/failed/cancelled” banner in transcript and exports.
- Autosave a recoverable project and offer “reopen last session.”
- Add search, timestamp click/navigation, find/replace speaker names, and direct transcript correction.
- Show download size, disk space, model license, language support, and estimated VRAM before downloading.
- Add cache management (list model sizes, remove models, open cache folder).
- Show actual backend/provider/device after fallback, not only the requested combo value.
- Allow a privacy mode that avoids saving notes/hints/history and strips source paths.
- Add accessibility checks: keyboard shortcuts, focus order, screen-reader labels, high-DPI layouts, and color contrast.
- Give the user a choice between retaining an ASR/LLM model for faster repeat runs and freeing it immediately for memory.

## Recommended implementation order

### Phase 1 — Protect correctness and user work

1. Fix safe asynchronous shutdown (H-01).
2. Stage/mark partial track results and block ambiguous exports (H-02).
3. Make notes chunking/context reduction strictly bounded (H-03).
4. Move notes and session autosave to atomic files with error reporting (H-04, M-16).
5. Move logging before imports and improve crash diagnostics (H-05).
6. Fix WebVTT escaping and central timestamp validation (M-12, M-13).

### Phase 2 — Make behavior testable and releases reproducible

1. Add dependency manifests/lockfiles and a clean build script (H-06).
2. Add deterministic unit/audio fixtures and CI (H-07).
3. Record per-run/per-track provenance (M-11).
4. Validate frozen builds on clean GPU and CPU-only machines, and gate build warnings (M-09).
5. Add checksums/SBOM/signing and third-party notices (M-07, M-19).

### Phase 3 — Performance and maintainability

1. Incremental transcript model/rendering (M-02).
2. Streaming/bounded audio decode (M-03).
3. Sensible ASR/LLM retention policy (M-01, M-10).
4. Split `MainWindow` into controllers/views and introduce an explicit state machine.
5. Minimize the standalone bundle only after the engine smoke suite exists (M-08).

## Suggested minimum test matrix

| Area | Cases |
|---|---|
| Merge | ties, overlap, long gaps, empty tracks, same-name speakers, out-of-order input |
| Audio | corrupt file, no audio stream, mono, stereo split, >2 channels, multiple streams, changing layout, multi-hour fixture |
| Whisper | word gap exactly below/at/above 1.0 s, no word timestamps, cancellation, GPU fallback |
| Qwen | sparse speech at known offsets, empty VAD, noisy breath regions, forced/auto language, batch result mismatch, timestamps outside region |
| ONNX | empty results, generator exception after partial output, provider fallback, cancellation |
| Exports | empty, Unicode, markup, multiline, invalid timestamps, incomplete tracks, all-formats write failure |
| Notes | below/at/above limits, one giant line, many parts requiring recursive reduction, malformed SSE, timeout, cancel at every phase, server crash |
| Persistence | large notes, write denial, atomic recovery, migration, crash during save, privacy reset |
| GUI state | run/cancel/close races, retry failed, mixed model settings, remove/clear/split, stale buttons/tooltips |
| Packaging | clean VM, no NVIDIA GPU, supported NVIDIA GPU, each ASR engine, notes server, no network with bundled model |

## Final assessment

The app's core idea and engine-specific handling are stronger than its surrounding reliability scaffolding. The highest return will come from protecting partial work, making shutdown deterministic, bounding long-context processing, and establishing reproducible builds plus regression fixtures. Those changes should precede new models or additional UI features because they reduce the chance that long, expensive transcription sessions end in ambiguous or unrecoverable output.

