"""Merge per-track segments into a single chronological conversation."""

from __future__ import annotations

from models import Segment, Track

# A same-speaker pause this long ends a coalesced block, so the next one gets
# its own timestamp instead of being hidden inside the previous paragraph.
MAX_COALESCE_GAP = 10.0

# Hard ceiling on a coalesced block, in characters. Without it a recording
# with only one speaker — a single track, or one person talking while the
# others stay quiet — collapses into a single block, i.e. one enormous line
# of text. `to_text()` output is what AI notes summarize, and llm.py splits
# that into context-sized parts *at line boundaries*: one unsplittable line
# means a long solo recording always overflows the model context. ~1500
# chars is a couple of paragraphs and comfortably under any part budget.
MAX_COALESCE_CHARS = 1500


def merge_tracks(tracks: list[Track]) -> list[Segment]:
    """All segments from all tracks, on the session clock, ordered by start time.

    Segment times inside a track are relative to its own file; each track's
    offset is added here, to copies, so the tracks keep their raw times and
    the offset can never be applied twice — however often this runs or the
    offset changes. What this returns is a view: editing it changes no
    track. Python's sort is stable, so exact ties keep track order.
    """
    segments = [
        Segment(seg.speaker, seg.start + track.offset, seg.end + track.offset, seg.text)
        for track in tracks for seg in track.segments
    ]
    segments.sort(key=lambda s: (s.start, s.end))
    return segments


def coalesce(segments: list[Segment], max_gap: float = MAX_COALESCE_GAP,
             max_chars: int = MAX_COALESCE_CHARS) -> list[Segment]:
    """Join consecutive segments from the same speaker into one block.

    Used for readable (Markdown/plain text) output. SRT/VTT keep the
    original fine-grained segments for accurate subtitle timing.

    A block ends when the speaker changes, when more than `max_gap` seconds
    of silence follow it, or when it reaches `max_chars` — see the notes on
    those constants above.
    """
    blocks: list[Segment] = []
    for seg in segments:
        prev = blocks[-1] if blocks else None
        if (prev is not None
                and prev.speaker == seg.speaker
                and seg.start - prev.end <= max_gap
                and len(prev.text) < max_chars):
            prev.text = (prev.text.rstrip() + " " + seg.text.strip()).strip()
            prev.end = max(prev.end, seg.end)
        else:
            blocks.append(Segment(seg.speaker, seg.start, seg.end, seg.text.strip()))
    return blocks
