# Simplify review: Offset column and Align, 2026-10-22

A code-quality pass (reuse, simplification, efficiency, altitude) over the
most recent feature work, the per-track **Offset** column and **Align…**.
I applied the findings that hold up and listed the rest with the reason
each was left alone. This was not a bug hunt, and no behaviour change was
intended.

| | |
| --- | --- |
| Branch | `claude/simplify-transcription-app-awa4h4` |
| Commit | `a537046` Simplify the Offset/Align code: share helpers, drop dead paths |
| Size | 8 files: +258 / −309 lines. `app/`: +137 / −168. `tests/`: +118 / −140 |
| Tests | 257 run, all pass (1 skipped), before and after |
| Pull request | None opened |

Line references below are to the code as of `a537046`.

---

## 1. Scope

The request was `/simplify transcription app`. That isn't a PR number,
branch or file path, and the branch had no commits beyond `main`. The
review therefore covered the latest merged change, `git diff HEAD~1` at the
merge commit `4ff0008` (PR #1):

- `c4b1982` Add a per-track Offset for recordings that started late
- `66a00d6` Add Align: suggest track offsets from shared sound

That is 13 files, +3,239 / −37. It includes the new `app/align.py` (761
lines), +601 lines in `app/main_window.py`, and changes to `audio.py`,
`merger.py`, `models.py`, `exporters.py` and the tests.

## 2. Method

Four independent read-only reviewers ran in parallel, one per angle:

| Angle | Question asked |
| --- | --- |
| **Reuse** | Does new code re-implement a helper the codebase already has? |
| **Simplification** | Is there redundant state, copy-paste, dead code or needless nesting? |
| **Efficiency** | Is there repeated work or I/O, hot-path cost, or a closure holding big data alive? |
| **Altitude** | Is each fix at the right depth, or a special case layered on shared code? |

Each reviewer read `CLAUDE.md` first and was told not to flag anything it
documents as intentional. I then checked every finding against the code
and tests, merged duplicates (several mechanisms were flagged by two or
three reviewers), and applied what survived. The efficiency reviewer
measured its claims with real PyAV decodes and NumPy at 4-hour scale.

---

## 3. What changed

### Summary

| # | Change | Flagged by | Main files |
| --- | --- | --- | --- |
| 3.1 | One h:mm:ss formatter, `exporters.hms` | Reuse | `exporters.py`, `align.py`, `main_window.py` |
| 3.2 | One stream lookup, `audio._audio_stream` | Reuse, Simplification, Altitude | `audio.py` |
| 3.3 | Align decodes in one-second chunks | Efficiency | `audio.py` |
| 3.4 | Matcher: one `_peak`, one `_weak`, unused inputs dropped | Simplification | `align.py` |
| 3.5 | Unread `Suggestion.matched_with` removed | Simplification | `align.py` |
| 3.6 | `AlignWorker` keeps its result and error | Simplification, Altitude | `align.py`, `main_window.py` |
| 3.7 | Offset edits arrive only through `_OffsetDelegate` | Simplification, Altitude, Reuse | `main_window.py` |
| 3.8 | Review dialog works out ticks once; unreachable branches go | Simplification, Altitude | `main_window.py` |
| 3.9 | Speaker paths resolved once per file | Efficiency | `main_window.py` |
| 3.10 | Shared test helpers | Reuse | `tests/` |
| 3.11 | `CLAUDE.md` updated to match | — | `CLAUDE.md` |

### 3.1 One clock formatter

**Before.** The same formula, `f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"`,
was written out four times:

- `exporters._hms`
- `main_window._fmt_duration`
- `MainWindow._format_elapsed`
- the new `align.format_offset`, whose docstring promised "the same shape
  as the Duration column" with nothing enforcing it

**After.**
- `exporters.hms` (`app/exporters.py:13`) is public and documented as the
  clock that the table, the transcript and the text exports share.
- `format_offset` (`app/align.py:126`) builds on it and adds only the sign
  and milliseconds.
- `_fmt_duration` and `_format_elapsed` are deleted. Their seven call sites
  in `main_window.py` call `exporters.hms`.

**Behaviour.** Identical. All four truncated to whole seconds the same way,
and `tests/test_render.py` still pins the transcript output.

### 3.2 One audio-stream lookup

**Before.** `iter_stream_power` (new) copied the stream selection out of
`decode_track_audio`: the "None means 0, not `or 0`" rule, the bounds
check, and the "has the file changed?" `ValueError`.

**After.** `_audio_stream(container, path, stream_index)`
(`app/audio.py:86`) holds the rule, its explanatory comment and the error
message. Both decoders call it.

### 3.3 Align decodes in one-second chunks

**Before.** `AudioResampler` was created without `frame_size`, so it handed
back a few hundred samples per chunk: about 171 for 48 kHz AAC, or about
675,000 chunks for a 4-hour stream. Each chunk ran a dozen small NumPy
calls on the Python side. The efficiency reviewer measured, per 10 minutes
of audio:

| Path | Time |
| --- | ---: |
| Decode and resample only | 0.77 s |
| Align's read as it was | 2.47 s |
| With `frame_size=rate` | 0.86–0.97 s |

So about two thirds of Align's read time was per-chunk overhead: about
59 s → about 21 s per 4-hour stream, for every speaker file and the
reference, on every Align run. `Envelope._parts` also stops holding
hundreds of thousands of tiny arrays (about 93 MB of overhead for 11.5 MB
of data per 4-hour stream).

**After.** `AudioResampler(..., rate=rate, frame_size=rate)`
(`app/audio.py:180`), with a comment saying why.

**Equivalence check.** I decoded a 70 s, odd-length, 44.1 kHz stereo WAV
both ways with PyAV 18.0.0 (the version in the lock file):

| | Default chunks | `frame_size=8000` |
| --- | ---: | ---: |
| Chunks | 758 | 71 |
| Energy frames | 7,028 | 7,028 |
| Largest difference | — | 0.0 |

The last chunk comes out short rather than zero-padded, so file durations
don't change.

### 3.4 Matcher clean-up (`align.py`)

- **`_peak(score)`** (`app/align.py:288`) finds the best index, its score
  and the peak ÷ runner-up ratio. Before, `_best_lag` and `_part_verdict`
  each repeated "argmax, check it's finite and above 0, then
  `_ratio(peak, _rival(...))`" by hand. That is now in one place.
- **`_best_lag`** (`:300`) lost a `lag_range` parameter its only caller
  never passed, and a returned peak score its caller threw away. It now
  returns `(lag, ratio)`.
- **`_weak(match, detail)`** (`:354`) replaces five copies of the three
  lines "set strength to weak; set detail; return" in `find_offset`. The
  ladder of reasons a strong match is downgraded now reads one line per
  reason.
- `_window_agreement` lost a `per_window` alias of `MIN_WINDOW_SECONDS`.

Every detail string is unchanged, word for word.

### 3.5 `Suggestion.matched_with` removed

It was written in two places in `suggest_offsets` and passed as a 6th
positional argument in 14 test constructions, but never read. The field
and the argument are gone; the "Matched sound with X" reason text already
names the file.

### 3.6 `AlignWorker` keeps its result and error

**Before.** The worker emitted `finished_ok(AlignmentResult)` and
`failed(str)`. The window had two slots copying those into
`_align_result` and `_align_error`, reset both in two places, and read
them in `_on_align_finished`. That function also read `worker.cancelled`
straight off the worker. The result arrived only because the queued
`finished_ok` happened to be delivered before `QThread.finished`.

**After.** `AlignWorker` sets `self.result` or `self.error`
(`app/align.py:689` onwards), next to the `cancelled` it already had.
`_on_align_finished` (`app/main_window.py:1177`) reads them after
`worker.wait()`; `result is None` means failed or cancelled. This removed
two window attributes, two slots, two connects and two resets. `progress`
is still a signal, since it is shown while the worker runs.

### 3.7 Offset edits have one way in

**Before.** A typed edit arrived two ways:
- through `_OffsetDelegate.setModelData`, the real path;
- through `_on_item_changed`, which sent *any* change to an Offset cell to
  `_on_offset_edited`, which marks the offset **typed** (MANUAL).

So every code write to the cell had to block table signals, or an applied
Align result would have turned into a "typed" value. A typed value is
exactly what stops Align ticking that file next time. In the app, only the
tests used the `itemChanged` route.

**After.**
- `_OffsetDelegate.setModelData` (`app/main_window.py:282`) is the only
  entry. It dropped its unreachable non-`QLineEdit` fallback and the
  now-unused `QLineEdit` import.
- `_on_item_changed` (`:966`) handles the Speaker column only.
- `_show_offset` (`:940`) just writes the text and tooltip; no signal
  blocking.
- New `_set_offset(row, offset, source)` (`:947`) sets the value and its
  source and redraws the cell. `_on_offset_edited` and both loops in
  `_apply_alignment` use it.

`CLAUDE.md` already said typed edits commit through `_OffsetDelegate`, not
`itemChanged`; the code now matches that.

### 3.8 Review dialog and the track lock

- **Ticks worked out once.** `_starts_ticked` used to run once per
  suggestion in `__init__` and again in `_fill_row`, through a
  `_default_ticks` attribute that only `_fill_row` read. `__init__` now
  passes each row's starting tick to `_fill_row` (`:378`), which no longer
  applies the tick rule itself.
- **Unreachable branches removed.** From Align until the review closes,
  `_tracks_locked()` blocks every change to the track list, so a
  suggestion always still has its rows. Removed:
  - the "removed" text in the Now column, and the `bool(rows)` /
    `and rows` guards;
  - `_apply_alignment`'s `if not rows or not suggestion.applicable`
    (`chosen()` only returns rows that have a checkbox, which means they
    were applicable);
  - "skip files since removed" from its docstring.

  A comment in `_fill_row` records the invariant.
- **The lock is checked by every function that changes the track list.**
  `add_files`, `remove_selected` and `clear_tracks` checked
  `_tracks_locked()` themselves, but `_split_track_channels` relied on its
  only caller, the context menu, to check. It now checks itself (`:989`),
  so a future button or shortcut can't split mid-transcription.
  `TranscribeWorker` addresses tracks by position.

### 3.9 Paths resolved once per file

`add_files` and `set_session_reference` called `_canonical` (which does a
`Path.resolve()`) once per *row*, so a 6-channel file resolved 6 times on
the UI thread. That costs microseconds locally, but several round trips
each on a network share. `_speaker_file_keys()` (`:906`) resolves each
distinct path once, and both callers use it.

### 3.10 Test changes

The tests still check the same things. These changes remove duplication or
follow the code changes above.

`tests/test_lifecycle.py`
- `_WindowTest` gained three helpers that the subclasses had each written
  their own copy of:
  - `_patch(target, **kwargs)` (`:52`): replaces two patch/start/cleanup
    loops and `Alignment._patch`;
  - `add(*tracks)` (`:58`): previously `OffsetColumn.add`, and the body of
    `Alignment.stereo`;
  - `type_into_offset_cell(row, keys, enter=True)` (`:64`): also used,
    with `enter=False`, by the "offset still being typed" test, which had
    its own copy of the editor steps.
- `OffsetColumn` tests that simulated typing with `item.setText(...)` now
  use `edit()` → `_on_offset_edited`, the function the delegate calls. The
  real cell editor is still covered by the `type_into_offset_cell` tests.
  The assertion that signals weren't left blocked went, since nothing
  blocks them now.
- `Alignment.align()` takes `result=`, and scripted runs set
  `worker.result` / `worker.error` instead of emitting signals.
- `test_an_offset_typed_during_alignment_shows_as_current_and_manual` uses
  the existing `run_while` instead of its own Event/`QTimer` copy.

`tests/test_align.py`
- `_Recorder` (signal → list) is gone; tests read `worker.result` and
  `worker.error`.
- `_shaped_noise(energy, seed)` (`:743`) is shared by `_write_wav` and the
  NaN-channel test, which had copied the noise shaping by hand.

`tests/test_exporters.py`: `exporters._hms` → `exporters.hms`.

### 3.11 `CLAUDE.md`

The "Threading & signals" section said every worker hands its results to
the UI through Qt signals. It now adds that a one-shot answer can stay on
the worker for the `finished` handler to read after `wait()`, as
`AlignWorker`'s `result`, `error` and `cancelled` do.

---

## 4. Findings not applied

| Finding | From | Why not |
| --- | --- | --- |
| **Cache fingerprints between Align runs**, keyed by canonical path, streams, mtime and size. Re-pressing Align decodes every file again: K × ~21 s per 4-hour file after 3.3. | Efficiency | New window state and a new feature, not a clean-up. The best follow-up if Align is pressed repeatedly on long sessions. |
| **Memoise the reference's `rfft` and squared cumsum** in `_scores` (`app/align.py:241`). On a 4-hour strong match with window agreement: 3.48 s and 12 `_scores` calls, 0.56 s in `rfft(ref)`, 10 of the 12 transforms redundant. That is about 15% of matching. | Efficiency | Adds a cache threaded through the matcher for a modest gain. Worth doing with the cache above. |
| **A 2·3·5-smooth FFT length** instead of the next power of two (`:251`). For 3 h against 3 h: 34 ms against 77 ms per transform. | Efficiency | Changes the FFT size, so scores move in the last bits, against calibration that is still provisional. Not a pure clean-up. |
| **One demux pass for a multi-stream file.** Each stream reopens the container and reads every packet. | Efficiency | About 4% with the file in the OS cache (0.14 s per extra pass per 10 minutes). It matters on a cold disk or network share, but needs a restructured decoder. |
| **Exporters merge the tracks themselves**, so `export_dialog` (`app/main_window.py:1471`) no longer merges again by hand in each branch. | Altitude, Simplification | Changes `export` / `export_all` and their tests, which pass segments and tracks independently. The re-merge before each write stays as `CLAUDE.md` describes. |
| **`Envelope.feed` is test-only, and `feed_power` cleans NaN a second time** (`app/align.py:158`, `:169`). | Reuse, Simplification | `test_align` pins the Envelope's own NaN handling. With one-second chunks the extra pass costs almost nothing. |
| **Replace `_align_reviewing` with a pending result.** | Altitude | The flag must also cover the review window's `exec()`, after the result has been taken. A pending-result field would need the same care, so it isn't simpler. |
| **`_rows_by_file()` / `_file_offset()` helpers, and dropping the worker's `current_offsets` snapshot.** | Altitude, Reuse | The places that work out "this file's rows" need different shapes: indices, tracks, an "any" check, a "varies" check. Dropping the snapshot would lose the "offset changed while Align ran" note. |
| **`_update_busy` sets all three Align button states** (`:1116`, `:1140`). | Altitude | Low value; the current code follows the Transcribe and AI Notes button pattern. |
| **`closeEvent` lists every worker twice** (`:1720`). | Altitude | Pre-existing pattern; Align added one line to each list. |
| `rebase`'s mode guard repeats its caller's check. | Simplification | A test pins it, and it keeps `rebase` safe on its own. |
| `_RANK["none"]` is never looked up. | Simplification | It makes the mapping complete; harmless. |
| `is_ticked` re-checks `is_checkable`. | Simplification | Redundant but explicit, and costs nothing. |
| `_split_track_channels`'s own stream-index check. | Reuse | It works on probe results, not an open container, and raises a different error for a different dialog. |
| Small costs: export merges again after its emptiness check; `_set_running` merges to test for segments; `_rival` copies `score` (2.6 ms); `_active_seconds` runs twice for one range (3.7 ms); `merge_tracks` copies segments (1.8 ms per 400 ms refresh at 10,000 segments). | Efficiency | None is material. The `merge_tracks` copy is the documented design. |

## 5. Checked and found fine

- **No existing helper** for `_canonical`, `parse_offset`, a
  `QMessageBox` wrapper or a worker base class. `AlignWorker`'s
  `stop` / `cancelled` idiom matches `TranscribeWorker` and `NotesWorker`.
- **Already at the right depth:**
  - `merge_tracks` as the one place an offset is applied;
  - `_tracks_locked()` as the one definition of "locked";
  - the session reference kept apart from `Track`;
  - the canonical-path check in both import routes;
  - `rebase` / `measured`;
  - waiting for other modal prompts before the review opens;
  - the automatic reference turning DEFAULT into MATCHED.
- **No closures holding large data.** Fingerprints are local to
  `AlignWorker._run`. The 250 ms retry lambda captures only the small
  `AlignmentResult`, and the progress lambda only `self`.
- **Startup is unaffected.** `align` still imports no NumPy, PyAV or Qt
  work at load; it now imports `exporters`, which `main_window` already
  loads.
- **Don't parallelise the decode.** Two files decoded in threads took
  4.58 s against 1.43 s sequentially, on 4 CPUs.

## 6. Verification

| Check | Result |
| --- | --- |
| Offline suite, before any change | 257 run, all pass (1 skipped) |
| Offline suite, after (re-run at `a537046`) | 257 run, all pass (1 skipped) |
| pyflakes on `app/` and `tests/` | No new warnings. The only output is the intentional `import tests`, `cuda_setup` and `QTextCursor` imports. |
| Resampler chunk-size equivalence | Identical energies (section 3.3) |

**Environment.** A Linux cloud container with Qt on the offscreen platform:
Python 3.13.16, PySide6 6.12.0, NumPy 2.4.6, PyAV 18.0.0. The project's
`.venv` is Windows Python 3.14 with PySide6 6.11.1, so it is worth running
the suite once there:

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -t .
```

**Not verified here:**
- `--selftest`: no GPU in the container. These changes don't touch
  transcription or notes, but `decode_track_audio` now goes through
  `_audio_stream`.
- The GUI on Windows.
- Align on real recordings.
- The standalone build.

## 7. Suggested follow-ups, most valuable first

1. **Fingerprint cache plus `rfft(ref)` memo** (section 4, first two rows).
   Together these make a second Align press cost only the matching.
2. **One demux pass per multi-stream file**, if sessions often live on
   network or USB storage.
3. **Exporters take tracks and merge themselves**, so a future export path
   can't reuse a stale `segments` variable.
4. Run the suite in the Windows `.venv` and do one real Align on a
   recorded session. It is the first real-world check of the 1-second
   chunking, and the thresholds still need calibrating on real audio
   (see "Calibration status" in `CLAUDE.md`).
