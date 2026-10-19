"""Data model for tracks and transcript segments."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class OffsetSource(Enum):
    """How a track's offset was set, shown in its Offset cell tooltip."""
    DEFAULT = "default"   # never set; assumed to start with the session
    MANUAL = "manual"     # typed by the user (including a typed 0)
    MATCHED = "matched"   # applied from an Align sound match
    GUESS = "guess"       # applied from Align's "stopped together" guess


class TrackStatus(Enum):
    PENDING = "Pending"
    TRANSCRIBING = "Transcribing"
    DONE = "Done"
    ERROR = "Error"
    CANCELLED = "Cancelled"


@dataclass
class Segment:
    speaker: str
    start: float
    end: float
    text: str


@dataclass
class Track:
    path: Path
    speaker: str
    duration: float = 0.0
    status: TrackStatus = TrackStatus.PENDING
    language: str = ""
    error: str = ""
    segments: list[Segment] = field(default_factory=list)
    # Set when this track is a slice of a multitrack file: an audio stream
    # within the container, and optionally a single channel of that stream.
    stream_index: int | None = None
    channel: int | None = None
    # Seconds into the session at which this file began recording: non-zero
    # when someone joined late or recorded on a separate device. Applied by
    # merge_tracks, so segment times stay relative to the track's own file.
    offset: float = 0.0
    offset_source: OffsetSource = OffsetSource.DEFAULT

    @property
    def is_subtrack(self) -> bool:
        return self.stream_index is not None or self.channel is not None

    @property
    def source_label(self) -> str:
        parts = []
        if self.stream_index is not None:
            parts.append(f"track {self.stream_index + 1}")
        if self.channel is not None:
            parts.append(f"ch {self.channel + 1}")
        return f"{self.path.name} [{', '.join(parts)}]" if parts else self.path.name


AUDIO_EXTENSIONS = {
    ".flac", ".m4a", ".mp4", ".aac", ".mp3", ".wav", ".ogg",
    ".opus", ".wma", ".webm", ".mka", ".mkv", ".mov", ".aif", ".aiff",
}


def default_speaker_name(path: Path) -> str:
    """Derive a speaker name from a filename: 'alice_track-01.flac' -> 'Alice Track 01'."""
    name = path.stem.replace("_", " ").replace("-", " ").strip()
    return " ".join(w.capitalize() if w.islower() else w for w in name.split()) or path.stem
