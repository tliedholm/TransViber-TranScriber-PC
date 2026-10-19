"""Placing tracks on one session clock when they didn't start recording together.

A track's offset is when its file began, measured from the start of the
session — not when its speaker first talked. Leading silence already in a
file is correct timing and stays where it is. merge_tracks adds the offset
to copies of the segments, so changing it never needs a re-transcription.

Pure Python here: the window imports this module at startup, and
test_lifecycle pins that importing the window loads no inference libraries.
"""

from __future__ import annotations

import re

_LAST_FIELD = re.compile(r"\d+(\.\d+)?", re.ASCII)
_WHOLE_FIELD = re.compile(r"\d+", re.ASCII)

# No session runs anywhere near this (it is over 100 days), but a pasted run
# of digits would otherwise overflow to infinity, which nothing downstream —
# the transcript, SRT, VTT — can format.
MAX_OFFSET_SECONDS = 10_000_000


def parse_offset(text: str) -> float:
    """Read a typed offset: `150`, `2:30`, `2:30.5`, `1:02:30`, `2:30,5`.

    The first field can be any size (`90:00` is 90 minutes; hours may pass
    24); later fields must be under 60. Only the last field may have a
    fraction, and a comma works as the decimal point. Raises ValueError with
    a reason fit to show the user. The result is rounded to milliseconds,
    the precision SRT, VTT and JSON keep.
    """
    cleaned = text.strip().replace(",", ".")
    if not cleaned:
        raise ValueError("Enter a time such as 2:30, or 0 to reset it.")
    if cleaned.startswith("-"):
        raise ValueError("An offset can't be negative.")
    fields = cleaned.split(":")
    if len(fields) > 3:
        raise ValueError("Use at most hours:minutes:seconds, e.g. 1:02:30.")
    *whole, last = fields
    if not all(_WHOLE_FIELD.fullmatch(f) for f in whole) or not _LAST_FIELD.fullmatch(last):
        raise ValueError(f"\"{text.strip()}\" isn't a time. Use seconds (150), "
                         f"minutes:seconds (2:30) or hours:minutes:seconds (1:02:30).")
    if any(len(f.split(".")[0]) > 9 for f in fields):
        raise ValueError("That offset is too large.")
    numbers = [int(f) for f in whole] + [float(last)]
    if any(n >= 60 for n in numbers[1:]):
        raise ValueError("Minutes and seconds after a colon must be under 60.")
    seconds = 0.0
    for n in numbers:
        seconds = seconds * 60 + n
    if seconds > MAX_OFFSET_SECONDS:
        raise ValueError("That offset is too large.")
    return round(seconds, 3)


def format_offset(seconds: float) -> str:
    """`150.5` -> `2:30.5`, `3750` -> `1:02:30`, `0` -> `0:00`.

    The same shape as the Duration column, plus up to three decimals with
    trailing zeros dropped, so a matched offset keeps its milliseconds.
    """
    ms_total = round(seconds * 1000)
    whole, ms = divmod(ms_total, 1000)
    h, rem = divmod(whole, 3600)
    m, s = divmod(rem, 60)
    text = f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"
    if ms:
        text += "." + f"{ms:03d}".rstrip("0")
    return text
