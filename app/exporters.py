"""Export merged transcripts to Markdown, plain text, SRT, VTT, and JSON."""

from __future__ import annotations

import contextlib
import json
import os
import tempfile
from datetime import datetime
from pathlib import Path

from merger import coalesce
from models import Segment, Track


def hms(seconds: float) -> str:
    """Whole seconds as `m:ss`, or `h:mm:ss` from an hour: the clock the
    track table, the transcript and the text exports all show."""
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def _ts_srt(seconds: float) -> str:
    ms = round(seconds * 1000)
    h, rem = divmod(ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _ts_vtt(seconds: float) -> str:
    return _ts_srt(seconds).replace(",", ".")


def _one_line(text: str) -> str:
    """Collapse whitespace to single spaces.

    A blank line ends a cue in both SRT and WebVTT, so a newline inside
    transcript text (or a speaker name) would silently truncate the cue and
    turn the rest into a malformed one.
    """
    return " ".join(text.split())


def _vtt_escape(text: str) -> str:
    """Escape the WebVTT payload's three markup characters.

    Unescaped, a transcript containing `<`, `>` or `&` is read as cue tags or
    entity references: `x <tag> & y` loses "<tag>" entirely. Escaping `<`
    also neutralises a literal `-->` in the text, which would otherwise look
    like a timing line.
    """
    return (_one_line(text).replace("&", "&amp;")
            .replace("<", "&lt;").replace(">", "&gt;"))


def to_markdown(segments: list[Segment], tracks: list[Track] | None = None,
                model_name: str = "") -> str:
    lines = ["# Transcript", ""]
    for seg in coalesce(segments):
        lines.append(f"**{seg.speaker}** \\[{hms(seg.start)}\\]: {seg.text}")
        lines.append("")
    return "\n".join(lines)


def to_text(segments: list[Segment], tracks: list[Track] | None = None,
            model_name: str = "") -> str:
    lines = []
    for seg in coalesce(segments):
        lines.append(f"[{hms(seg.start)}] {seg.speaker}: {seg.text}")
        lines.append("")
    return "\n".join(lines)


def to_srt(segments: list[Segment], tracks: list[Track] | None = None,
           model_name: str = "") -> str:
    lines = []
    for i, seg in enumerate(segments, 1):
        lines.append(str(i))
        lines.append(f"{_ts_srt(seg.start)} --> {_ts_srt(seg.end)}")
        lines.append(f"{_one_line(seg.speaker)}: {_one_line(seg.text)}")
        lines.append("")
    return "\n".join(lines)


def to_vtt(segments: list[Segment], tracks: list[Track] | None = None,
           model_name: str = "") -> str:
    lines = ["WEBVTT", ""]
    for seg in segments:
        lines.append(f"{_ts_vtt(seg.start)} --> {_ts_vtt(seg.end)}")
        lines.append(f"<v {_vtt_escape(seg.speaker)}>{_vtt_escape(seg.text)}")
        lines.append("")
    return "\n".join(lines)


def to_json(segments: list[Segment], tracks: list[Track] | None = None,
            model_name: str = "") -> str:
    data = {
        "created": datetime.now().astimezone().isoformat(timespec="seconds"),
        "model": model_name,
        "tracks": [
            {
                "file": str(t.path),
                "speaker": t.speaker,
                "duration": round(t.duration, 3),
                "language": t.language,
                "stream": t.stream_index,
                "channel": t.channel,
                # Segment times below are session times: this track's own
                # file times plus this offset.
                "offset": round(t.offset, 3),
                "offset_source": t.offset_source.value,
                # A track that errored or was cancelled may still have
                # segments — whatever was transcribed before it stopped. The
                # status is the only thing marking that text as partial.
                "status": t.status.value,
                "error": t.error,
            }
            for t in (tracks or [])
        ],
        "segments": [
            {
                "speaker": seg.speaker,
                "start": round(seg.start, 3),
                "end": round(seg.end, 3),
                "text": seg.text.strip(),
            }
            for seg in segments
        ],
    }
    return json.dumps(data, ensure_ascii=False, indent=2)


# Every exporter takes the same (segments, tracks, model_name) arguments —
# only to_json uses the last two — so EXPORTERS[ext](segments, tracks, model)
# is safe for any key in here. Mixed signatures made the obvious call raise
# TypeError for exactly one format.
EXPORTERS = {
    ".md": to_markdown,
    ".txt": to_text,
    ".srt": to_srt,
    ".vtt": to_vtt,
    ".json": to_json,
}


class ExportError(OSError):
    """An export that stopped partway, and exactly how far it got.

    `written` were replaced in full before `failed` could not be; every
    path in `unchanged` (`failed` first) still holds whatever it held
    before, if it existed at all. An OSError, so a caller that only reports
    `str(exc)` still says all of that.
    """

    def __init__(self, written: list[Path], failed: Path,
                 unchanged: list[Path], cause: OSError):
        self.written = written
        self.failed = failed
        self.unchanged = unchanged
        self.cause = cause
        reason = cause.strerror or str(cause)
        if written:
            message = (
                f"Exported {_names(written)}, but could not write "
                f"{failed.name}: {reason}\n\nNot updated: {_names(unchanged)}. "
                "Any earlier copies of those were left as they were, so they "
                "may not match the files just exported.")
        else:
            message = (f"Could not write {failed.name}: {reason}\n\n"
                       "Nothing was exported, and no existing file was changed.")
        super().__init__(message)


def _names(paths: list[Path]) -> str:
    return ", ".join(p.name for p in paths)


def _stage(path: Path, text: str) -> Path:
    """Write `text` to a new temporary file beside `path`, flushed to disk.

    Beside it, so the rename that publishes it stays on one volume; a
    unique name, so two exports to one folder can't share a temp file.
    """
    fd, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    temp = Path(name)
    try:
        with open(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        _discard(temp)
        raise
    return temp


def _discard(temp: Path) -> None:
    """Remove a staged file, quietly: a temp file left behind must never
    replace the error that explains why the export failed."""
    with contextlib.suppress(OSError):
        temp.unlink()


def _publish(files: list[tuple[Path, str]]) -> list[Path]:
    """Replace each file with its text, whole or not at all.

    Every file is staged before any is replaced, so a full disk or a
    folder that can't be written leaves every existing file as it was. The
    replacements are separate renames, not one transaction: if one fails
    (the file is open in another program, say), the ones before it are
    already new and the rest still old, and the ExportError says which.
    """
    paths = [path for path, _text in files]
    staged: list[Path] = []
    written: list[Path] = []
    current = paths[0]   # the file being staged or replaced when one fails
    try:
        for current, text in files:
            staged.append(_stage(current, text))
        for current, temp in zip(paths, staged):
            os.replace(temp, current)
            written.append(current)
    except OSError as exc:
        unchanged = [current] + [p for p in paths if p not in written and p != current]
        raise ExportError(written, current, unchanged, exc) from exc
    finally:
        for temp in staged[len(written):]:
            _discard(temp)
    return written


def write_file(path: Path, text: str) -> None:
    """Replace `path` with `text` whole, or leave it as it was."""
    _publish([(path, text)])


def export(path: Path, segments: list[Segment], tracks: list[Track], model_name: str) -> None:
    render = EXPORTERS.get(path.suffix.lower())
    if render is None:
        raise ValueError(f"Unsupported export format: {path.suffix}")
    write_file(path, render(segments, tracks, model_name))


def target_paths(base_path: Path) -> list[Path]:
    """Every file export_all would write for `base_path`.

    The extension is only *replaced* when the name already ends in one of
    ours: `Path.with_suffix` would read the ".5" of "Session 2.5" as a suffix
    and write "Session 2.md" — a name the user never typed, over a file they
    may already have.
    """
    stem = (base_path.with_suffix("")
            if base_path.suffix.lower() in EXPORTERS else base_path)
    return [stem.with_name(stem.name + ext) for ext in EXPORTERS]


def export_all(base_path: Path, segments: list[Segment], tracks: list[Track], model_name: str) -> list[Path]:
    """Write every format, each file replaced whole or not at all.

    All five are rendered and staged before any file is replaced, so an
    error in rendering, or a disk that fills while staging, changes
    nothing. The set as a whole is not atomic: if one replacement fails,
    the ExportError raised names the formats already written and the ones
    still holding their old contents.
    """
    return _publish([(path, EXPORTERS[path.suffix.lower()](segments, tracks, model_name))
                     for path in target_paths(base_path)])
