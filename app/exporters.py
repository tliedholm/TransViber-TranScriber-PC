"""Export merged transcripts to Markdown, plain text, SRT, VTT, and JSON."""

from __future__ import annotations

import json
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


def export(path: Path, segments: list[Segment], tracks: list[Track], model_name: str) -> None:
    render = EXPORTERS.get(path.suffix.lower())
    if render is None:
        raise ValueError(f"Unsupported export format: {path.suffix}")
    path.write_text(render(segments, tracks, model_name), encoding="utf-8")


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
    """Write every format. Renders all of them before writing any, so a
    failure partway through leaves no half-updated set behind."""
    rendered = [(path, EXPORTERS[path.suffix.lower()](segments, tracks, model_name))
                for path in target_paths(base_path)]
    for path, content in rendered:
        path.write_text(content, encoding="utf-8")
    return [path for path, _content in rendered]
