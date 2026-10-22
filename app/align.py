"""Placing tracks on one session clock when they didn't start recording together.

A track's offset is when its file began, measured from the start of the
session — not when its speaker first talked. Leading silence already in a
file is correct timing and stays where it is. merge_tracks adds the offset
to copies of the segments, so changing it never needs a re-transcription.

Align suggests offsets from sound the recordings share — a full-session mix
that holds everyone's voice, mic bleed, a room — and the window only ever
applies what the user ticks. The evidence is a compact "onset fingerprint"
per file: how sharply its loudness rises in each 10 ms frame. Two files that
heard the same sound rise together, so sliding one fingerprint along the
other (normalised cross-correlation through an FFT) finds the lag at which
they line up. Rises rather than raw loudness because a speaker's isolated
file and the session mix share the *onsets* of that speaker's syllables even
while other voices fill the mix: on synthetic sessions, loudness itself
matched with its peak barely clear of its own neighbourhood, onsets with the
peak several times clear. A match is only "strong" when that peak stands
well clear of the best other lag; anything less is offered unticked or not
at all — a wrong offset moves every line a speaker said.

No NumPy, PyAV or Qt work at import time: the window imports this module at
startup, and test_lifecycle pins that importing the window loads no
inference libraries (NumPy counts).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from pathlib import Path

from PySide6.QtCore import QThread, Signal

from exporters import hms

_LAST_FIELD = re.compile(r"\d+(\.\d+)?", re.ASCII)
_WHOLE_FIELD = re.compile(r"\d+", re.ASCII)

# No session runs anywhere near this (it is over 100 days), but a pasted run
# of digits would otherwise overflow to infinity, which nothing downstream —
# the transcript, SRT, VTT — can format.
MAX_OFFSET_SECONDS = 10_000_000

FRAME_RATE = 100            # fingerprint frames per second (10 ms resolution)
DECODE_RATE = 8000          # plenty for loudness; half the data of 16 kHz
# Lags are only scored where the two recordings overlap by at least this
# much (or half the shorter one). The plan's 30 s let unrelated files throw up
# a lone "strong" peak now and then, always at a lag overlapping 30-70 s:
# scored on so little, a score is mostly noise.
MIN_OVERLAP_SECONDS = 120
# And a peak is only strong when it rests on at least this much overlap, or
# on nearly all of the shorter recording.
STRONG_MIN_OVERLAP_SECONDS = 240
# Peak-to-runner-up ratios. Provisional: CLAUDE.md ("Track alignment") has
# what they were tried on, none of it a real recording.
MATCH_RATIO = 2.0           # strong match: ticked in the review window
WEAK_RATIO = 1.5            # weak match: offered, unticked
RUNNER_UP_EXCLUSION_SECONDS = 0.5   # the peak's own shoulders aren't a rival
# A strong match is also found again in its first and last thirds, sixths,
# twelfths and MIN_PART_SECONDS, each part on its own; if one clearly lands
# more than this away from the whole, one fixed offset can't be right (clock
# drift, a paused recorder, an edited file). Shrinking the parts towards each
# end is what reaches a cut near the start or the end: the part straddling it
# mostly agrees, the smaller part beyond it doesn't. A cut within about 40 s
# of either end is still beyond them.
CONSISTENCY_SECONDS = 0.1
MIN_PART_SECONDS = 80       # a part shorter than this is too little to judge
# Against a full-session reference a strong match also has to be found again,
# independently, in separate stretches of the speaker's active audio, to
# within CONSISTENCY_SECONDS.
MIN_WINDOW_SECONDS = 20     # of active audio per stretch
MIN_AGREEING_WINDOWS = 2
ACTIVE_SHARE = 0.1          # a second is active above this share of the busiest seconds
# ...and above this, whatever the busiest seconds are. Per second of
# fingerprint, a microphone's hiss at any level stays under about 0.6 (its
# rises are relative); syllables score from about 1 when barely above the
# hiss to 100 when clear. Without it a file that is nearly all hiss has hiss
# for its "busiest" seconds, and nearly all digital silence has none at all.
SOUND_ENERGY = 1.0
# Two mostly silent files — a few words each on gated headsets — are sparse
# trains of onsets: two coinciding by chance outscore one, and that alone can
# look strong. A strong match needs this many seconds of sound in each file
# within the overlap, and a part needs it before it can contradict one.
MIN_ACTIVE_SECONDS = 15
SILENCE_ENERGY = 1e-7       # mean square below about -70 dBFS: nothing to match

SESSION_REFERENCE = "session_reference"
AUTOMATIC_REFERENCE = "automatic_reference"


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

    The Duration column's shape (`hms`), plus up to three decimals with
    trailing zeros dropped, so a matched offset keeps its milliseconds.
    Negative values (a rejected suggestion, never an applied offset) get a
    leading minus.
    """
    sign = "-" if seconds < 0 else ""
    whole, ms = divmod(round(abs(seconds) * 1000), 1000)
    text = hms(whole)
    if ms:
        text += "." + f"{ms:03d}".rstrip("0")
    return sign + text


# ---------- Fingerprints ----------

class Envelope:
    """Streaming frame energies: feed decoded chunks, read mean power per frame.

    Chunks from the decoder end anywhere, so the samples past the last whole
    frame carry over into the next chunk; the result is the same however the
    audio is cut up. `feed` takes samples, `feed_power` their power already
    summed across channels.
    """

    def __init__(self, sample_rate: int = DECODE_RATE, frame_rate: int = FRAME_RATE) -> None:
        self._hop = sample_rate // frame_rate
        self._parts: list = []
        self._carry = None

    def feed(self, chunk) -> None:
        import numpy as np

        samples = np.asarray(chunk, dtype=np.float64).reshape(-1)
        self.feed_power(samples * samples)

    def feed_power(self, chunk) -> None:
        import numpy as np

        # A NaN or infinite sample (a damaged float file) would otherwise
        # poison every frame after it through the sums and the log.
        power = np.nan_to_num(np.asarray(chunk, dtype=np.float64).reshape(-1),
                              nan=0.0, posinf=0.0, neginf=0.0)
        if self._carry is not None and self._carry.size:
            power = np.concatenate([self._carry, power])
        frames = power.size // self._hop
        if frames:
            self._parts.append(power[:frames * self._hop].reshape(frames, self._hop).mean(axis=1))
        self._carry = power[frames * self._hop:]

    def energy(self):
        import numpy as np

        if not self._parts:
            return np.zeros(0)
        return np.concatenate(self._parts)


def sum_energies(energies: list):
    """One file's streams heard as a whole: frame energies added, shorter ones padded."""
    import numpy as np

    if not energies:
        return np.zeros(0)
    total = np.zeros(max(len(e) for e in energies))
    for e in energies:
        total[:len(e)] += e
    return total


def fingerprint(energy):
    """Onset strength per frame: how sharply loudness rises, centred on zero.

    `log10(energy + 1e-8)` puts every level on one scale; the floor (about
    -80 dBFS) keeps digital silence from dominating while leaving room for
    faint bleed. Only rises count (a syllable starting, a door, laughter),
    and a 1-second moving average is taken off so busy and quiet stretches
    weigh alike.
    """
    import numpy as np

    level = np.log10(np.asarray(energy, dtype=np.float64) + 1e-8)
    if level.size == 0:
        return level
    rises = np.maximum(np.diff(level, prepend=level[0]), 0.0)
    return rises - _moving_average(rises, FRAME_RATE + 1)


def _moving_average(values, width: int):
    import numpy as np

    n = values.size
    sums = np.concatenate([[0.0], np.cumsum(values)])
    idx = np.arange(n)
    lo = np.clip(idx - width // 2, 0, n)
    hi = np.clip(idx + width // 2 + 1, 0, n)
    return (sums[hi] - sums[lo]) / (hi - lo)


# ---------- Matching ----------

@dataclass
class Match:
    """Where `other` starts on `ref`'s clock, and how sure that is."""
    lag: float = 0.0          # seconds; positive: other began after ref did
    strength: str = "none"    # "strong" | "weak" | "none"
    ratio: float = 0.0        # peak over the best rival lag
    detail: str = ""          # why it isn't stronger, in plain English
    windows_agreeing: int = 0     # separate parts that found the same lag
    windows_checked: int = 0      # (both 0 when agreement wasn't asked for)
    windows_spread: float = 0.0   # seconds between the agreeing parts' lags


def _scores(ref, other, lag_range: tuple[int, int] | None = None):
    """(lags, score): normalised cross-correlation of `other` slid along `ref`.

    Scored at every lag where the two overlap by at least MIN_OVERLAP_SECONDS
    (or half the shorter recording, if that is less), and -inf elsewhere.
    `lag_range` limits the lags to [lo, hi] frames.
    """
    import numpy as np

    n, m = ref.size, other.size
    size = 1 << int(np.ceil(np.log2(n + m)))
    corr = np.fft.irfft(np.fft.rfft(ref, size) * np.conj(np.fft.rfft(other, size)), size)
    lags = np.arange(-(m - 1), n)
    corr = np.concatenate([corr[size - (m - 1):], corr[:n]]) if m > 1 else corr[:n]
    ref_sq = np.concatenate([[0.0], np.cumsum(ref * ref)])
    other_sq = np.concatenate([[0.0], np.cumsum(other * other)])
    r0, r1 = np.maximum(0, lags), np.minimum(n, lags + m)
    o0, o1 = np.maximum(0, -lags), np.minimum(m, n - lags)
    denom = np.sqrt((ref_sq[r1] - ref_sq[r0]) * (other_sq[o1] - other_sq[o0]))
    with np.errstate(invalid="ignore", divide="ignore"):
        score = np.where(denom > 0, corr / np.where(denom > 0, denom, 1.0), -np.inf)
    min_overlap = max(1, min(int(MIN_OVERLAP_SECONDS * FRAME_RATE), n // 2, m // 2))
    usable = (r1 - r0) >= min_overlap
    if lag_range is not None:
        usable &= (lags >= lag_range[0]) & (lags <= lag_range[1])
    return lags, np.where(usable, score, -np.inf)


def _ratio(peak: float, rival: float) -> float:
    import math

    if not math.isfinite(rival):
        return 0.0          # nothing to stand out from: no evidence
    return peak / rival if rival > 0 else float("inf")


def _rival(score, *centres: int) -> float:
    """The best score outside the shoulders of each of `centres` (indices)."""
    import numpy as np

    rivals = score.copy()
    exclude = int(RUNNER_UP_EXCLUSION_SECONDS * FRAME_RATE)
    for at in centres:
        rivals[max(0, at - exclude):max(0, at + exclude + 1)] = -np.inf
    return float(np.max(rivals))


def _peak(score) -> tuple[int, float, float]:
    """(index, score, peak/runner-up) of the best lag; score and ratio are 0
    when no lag scored above zero."""
    import numpy as np

    best = int(np.argmax(score))
    peak = float(score[best])
    if not np.isfinite(peak) or peak <= 0:
        return best, 0.0, 0.0
    return best, peak, _ratio(peak, _rival(score, best))


def _best_lag(ref, other) -> tuple[int, float]:
    """(lag in frames, peak/runner-up) of `other` slid along `ref`."""
    if ref.size == 0 or other.size == 0:
        return 0, 0.0
    lags, score = _scores(ref, other)
    best, _score, ratio = _peak(score)
    return int(lags[best]), ratio


def _strength(ratio: float) -> str:
    if ratio >= MATCH_RATIO:
        return "strong"
    if ratio >= WEAK_RATIO:
        return "weak"
    return "none"


def _part_verdict(ref, other, part_start: int, part_end: int,
                  lag: int) -> tuple[str, int]:
    """("agrees" | "disagrees" | "unclear", lag found) for one part of `other`.

    The part searches on its own, as if the whole-file match had never been
    made, but only at lags where nearly all of it lies inside `ref`: scored
    at the edge on a sliver of overlap, a part throws up chance peaks. It
    agrees if its clear peak is the whole-file lag, and disagrees if its
    clear peak is somewhere else. A part straddling a cut has two real
    peaks, one either side of it, and neither stands out from the other;
    that also disagrees, but only when the whole-file lag really is the
    second of them — a part with little to go on has a chance peak first and
    the true lag merely second, and that is not a contradiction.
    """
    import numpy as np

    part = other[part_start:part_end]
    size = part.size
    lags, score = _scores(ref, part, (-(size // 10), ref.size - size + size // 10))
    best, peak, ratio = _peak(score)
    found = int(lags[best]) - part_start
    if peak <= 0:
        return "unclear", found
    clear = ratio >= WEAK_RATIO
    if abs(found - lag) <= CONSISTENCY_SECONDS * FRAME_RATE:
        return ("agrees" if clear else "unclear"), found
    if clear:
        return "disagrees", found
    at = lag + part_start + (size - 1)            # the whole-file lag's index
    near = int(CONSISTENCY_SECONDS * FRAME_RATE)
    expected = float(np.max(score[max(0, at - near):at + near + 1], initial=-np.inf))
    floor = _rival(score, best, at)
    if _ratio(peak, floor) >= WEAK_RATIO and _ratio(expected, floor) >= WEAK_RATIO:
        return "disagrees", found
    return "unclear", found


def _weak(match: Match, detail: str) -> Match:
    """`match` demoted to weak, with why it isn't strong."""
    match.strength, match.detail = "weak", detail
    return match


def find_offset(ref, other, require_window_agreement: bool = False) -> Match:
    """Match `other`'s fingerprint against `ref`'s: where does `other` begin?

    A strong peak is then checked before it is called strong. It must rest
    on enough overlap and enough sound in each file, and no part of the
    overlap — its first and last thirds, sixths, twelfths and 80 s — may
    clearly find its own match somewhere else: one fixed offset can't fit both
    (drift, a pause, an edit). With `require_window_agreement` — every match
    against a full-session reference, direct or chained — separate stretches
    of `other`'s active audio must each find the same lag on their own, and
    none may find its own match somewhere else.
    """
    lag, ratio = _best_lag(ref, other)
    match = Match(lag=lag / FRAME_RATE, strength=_strength(ratio), ratio=ratio)
    if match.strength == "none":
        match.detail = "no usable shared sound"
        return match
    if match.strength == "weak":
        return _weak(match, "the best match barely stands out")

    start = max(0, -lag)
    end = min(other.size, ref.size - lag)
    overlap = end - start
    needed = min(STRONG_MIN_OVERLAP_SECONDS * FRAME_RATE, int(0.9 * min(ref.size, other.size)))
    if overlap < needed:
        return _weak(match, f"it rests on only {overlap // FRAME_RATE} s of overlap")

    active = min(_active_seconds(ref, start + lag, end + lag).size,
                 _active_seconds(other, start, end).size)
    if active < MIN_ACTIVE_SECONDS:
        return _weak(match, f"it rests on only {active} s of sound")

    smallest = MIN_PART_SECONDS * FRAME_RATE
    sizes = [overlap // parts for parts in (3, 6, 12) if overlap // parts >= smallest]
    if sizes:
        sizes.append(smallest)      # reaches a cut down to about 40 s from either end
    for size in sizes:
        for part_start in (start, end - size):
            if _active_seconds(other, part_start, part_start + size).size < MIN_ACTIVE_SECONDS:
                continue
            verdict, found = _part_verdict(ref, other, part_start, part_start + size, lag)
            if verdict == "disagrees":
                return _weak(match, f"start and end disagree by "
                                    f"{abs(found - lag) / FRAME_RATE:.1f} s: clock drift, "
                                    f"a paused recorder or an edited file")

    if require_window_agreement:
        agreeing, checked, spread, conflict = _window_agreement(ref, other, lag, start, end)
        match.windows_agreeing, match.windows_checked = agreeing, checked
        match.windows_spread = spread
        if conflict is not None:
            return _weak(match, f"one part of the file matches {conflict:.1f} s away: "
                                f"clock drift, a paused recorder or an edited file")
        if agreeing < MIN_AGREEING_WINDOWS:
            return _weak(match, f"only {agreeing} of {checked} separate parts of the file "
                                f"confirm it" if checked else
                                "too little audio to confirm it in separate parts")
    return match


def _active_seconds(other, start: int, end: int):
    """Seconds of `other[start:end]` with sound in them, as offsets from `start`.

    Measured on the fingerprint itself: a second counts when its onsets
    reach both a share of the busiest seconds and SOUND_ENERGY, so neither
    digital silence nor hiss of any level counts as sound.
    """
    import numpy as np

    seconds = (end - start) // FRAME_RATE
    if seconds <= 0:
        return np.zeros(0, dtype=int)
    stretch = other[start:start + seconds * FRAME_RATE].reshape(seconds, FRAME_RATE)
    per_second = (stretch * stretch).sum(axis=1)
    busiest = float(np.percentile(per_second, 99))
    return np.flatnonzero(per_second > max(ACTIVE_SHARE * busiest, SOUND_ENERGY))


def _window_agreement(ref, other, lag: int, start: int,
                      end: int) -> tuple[int, int, float, float | None]:
    """(agreeing, checked, spread s, conflict s): `other` matched in separate stretches.

    The stretches are cut from `other`'s active audio within the overlap —
    equal shares of its active seconds — so a speaker who only talks in one
    part of a long file is checked where they talk, and silence is never
    asked to confirm anything. Each is judged like a part of the overlap
    (`_part_verdict`): it counts if it agrees, and one that disagrees is a
    conflict.
    """
    active = _active_seconds(other, start, end)
    count = (3 if active.size >= 3 * MIN_WINDOW_SECONDS
             else 2 if active.size >= 2 * MIN_WINDOW_SECONDS else 0)
    found = []
    conflict = None
    for k in range(count):
        share = active[k * active.size // count:(k + 1) * active.size // count]
        part_start = start + int(share[0]) * FRAME_RATE
        part_end = start + (int(share[-1]) + 1) * FRAME_RATE
        verdict, shift = _part_verdict(ref, other, part_start, part_end, lag)
        if verdict == "agrees":
            found.append(shift)
        elif verdict == "disagrees" and conflict is None:
            conflict = abs(shift - lag) / FRAME_RATE
    spread = (max(found) - min(found)) / FRAME_RATE if found else 0.0
    return len(found), count, spread, conflict


# ---------- Suggestions ----------

class Cancelled(Exception):
    """Raised inside alignment work when the user cancels it."""


_RANK = {"none": 0, "weak": 1, "strong": 2}


@dataclass
class Suggestion:
    """One speaker file's row in the review window."""
    path: Path
    offset: float | None          # None: nothing to suggest (the file couldn't be read)
    method: str                   # "reference" | "matched" | "guess" | "error"
    reason: str
    strength: str | None = None   # "strong" | "weak" for a match

    @property
    def applicable(self) -> bool:
        """Whether the review window may offer it: never a negative start."""
        return self.offset is not None and self.offset >= 0


@dataclass
class AlignmentResult:
    """Everything the review window needs, captured when Align was pressed."""
    suggestions: list[Suggestion]     # as offered: on the session clock
    mode: str                         # SESSION_REFERENCE | AUTOMATIC_REFERENCE
    reference: Path                   # the session reference, or the automatic one
    note: str = ""                    # e.g. why every suggestion moved later
    anchor: float = 0.0               # the automatic reference's offset they hang from
    shift: float = 0.0                # how far all of them moved later to stay >= 0:00
    # Automatic mode: the same rows measured from the reference's own start,
    # so rebase() can hang them from a different offset without re-matching.
    measured: list[Suggestion] = field(default_factory=list)


def _match_reason(match: Match, via: Path) -> str:
    reason = f"Matched sound with {via.name} ({match.strength} match"
    if match.detail:
        reason += f": {match.detail}"
    elif match.windows_checked:
        reason += f"; {match.windows_agreeing} of {match.windows_checked} parts of the file agree"
        if match.windows_spread >= 0.005:
            reason += f" to within {match.windows_spread:.2f} s"
    return reason + ")"


def _anchored(measured: list[Suggestion], reference: Path,
              anchor: float) -> tuple[list[Suggestion], float, str]:
    """Measured rows put on the session clock, with the reference at `anchor`.

    If a row that starts ticked — the reference or a strong match — would
    then start before 0:00, every row moves later by the same amount. Weak
    matches don't get to move the others: one still before 0:00 afterwards
    is shown with the reason and can't be applied.
    """
    offered = [replace(s, offset=None if s.offset is None else s.offset + anchor)
               for s in measured]
    trusted = [s.offset for s in offered if s.offset is not None
               and (s.method == "reference" or s.strength == "strong")]
    shift = round(max(0.0, -min(trusted, default=0.0)), 3)
    note = ""
    if shift:
        note = (f"Some recordings started before {reference.name}, so every suggestion "
                f"was moved {format_offset(shift)} later to put the earliest at 0:00.")
    for s in offered:
        if s.offset is None:
            continue
        s.offset = round(s.offset + shift, 3)
        if s.offset < 0:
            s.reason += (f"; it would start {format_offset(-s.offset)} before 0:00, "
                         f"so it can't be applied")
    return offered, shift, note


def rebase(result: AlignmentResult, anchor: float) -> AlignmentResult:
    """The same automatic-mode suggestions, hung from a new reference offset.

    For when the reference file's offset was typed while Align was running:
    every match was measured from that file, so they all move with it. A
    session reference is fixed at 0:00 and is returned unchanged.
    """
    if result.mode != AUTOMATIC_REFERENCE:
        return result
    offered, shift, note = _anchored(result.measured, result.reference, anchor)
    return replace(result, suggestions=offered, note=note, anchor=anchor, shift=shift)


def suggest_offsets(units: list[Path], fingerprints: dict, current_offsets: dict,
                    session_reference: Path | None = None,
                    matcher=find_offset, should_stop=None) -> AlignmentResult:
    """Turn fingerprints into one suggestion per speaker file.

    `fingerprints` maps each path (and the session reference, if any) to its
    fingerprint, or to an Exception when the file couldn't be read.

    With a session reference its start is session 0:00, fixed: a file that
    would start before it is reported, never applicable, and the origin
    never moves to fit it. Without one, the longest readable speaker file is
    the reference, anchored on its *current* offset, and if a strong match
    lands before zero everything moves up together so it starts at 0:00.

    Each file is matched against the reference; one with no strong match
    there is tried against files already placed by a strong match, so a weak
    match never places another file. A file whose strong match puts it
    before the session reference is reported as that, not chained around.
    A file with no match at all gets the "stopped together" guess — the
    reference's end minus this file's length — which is a labelled
    hypothesis, not a measurement. A weak match is kept as the suggestion
    rather than being replaced by that guess.
    """
    explicit = session_reference is not None
    duration = {p: len(fp) / FRAME_RATE for p, fp in fingerprints.items()
                if not isinstance(fp, Exception)}
    if explicit:
        ref, candidates = session_reference, list(units)
    else:
        readable = [u for u in units if u in duration]
        if not readable:
            raise ValueError("none of the files could be read")
        ref = max(readable, key=lambda u: duration[u])
        candidates = [u for u in units if u != ref]

    # Measured from the reference's start; automatic mode anchors them after.
    out: dict[Path, Suggestion] = {}
    placed: dict[Path, float] = {ref: 0.0}
    best_unplaced: dict[Path, tuple[Match, Path, float]] = {}
    before_reference: set[Path] = set()

    def consider(u: Path, via: Path) -> bool:
        if should_stop is not None and should_stop():
            raise Cancelled
        match = matcher(fingerprints[via], fingerprints[u],
                        require_window_agreement=explicit)
        if match.strength == "none":
            return False
        offset = placed[via] + match.lag
        if match.strength == "strong" and not (explicit and offset < 0):
            placed[u] = offset
            out[u] = Suggestion(u, offset, "matched", _match_reason(match, via), "strong")
            return True
        if match.strength == "strong" and via == ref:
            before_reference.add(u)
        kept = best_unplaced.get(u)
        if kept is None or (_RANK[match.strength], match.ratio) > (_RANK[kept[0].strength],
                                                                    kept[0].ratio):
            best_unplaced[u] = (match, via, offset)
        return False

    pending = []
    for u in candidates:
        if isinstance(fingerprints.get(u), Exception):
            out[u] = Suggestion(u, None, "error", f"Couldn't read it: {fingerprints[u]}")
        else:
            pending.append(u)
    tried: set[tuple[Path, Path]] = set()
    progress = True
    while progress:
        progress = False
        for u in pending:
            if u in placed or u in before_reference:
                continue
            for via in list(placed):
                if (u, via) in tried:
                    continue
                tried.add((u, via))
                if consider(u, via):
                    progress = True
                    break
                if u in before_reference:
                    break

    for u in pending:
        if u in out:
            continue
        if u in best_unplaced:
            match, via, offset = best_unplaced[u]
            if explicit and offset < 0:
                reason = (f"Starts before the session reference by "
                          f"{format_offset(-offset)} ({match.strength} match with {via.name})")
            else:
                reason = _match_reason(match, via)
            out[u] = Suggestion(u, offset, "matched", reason, match.strength)
            continue
        guess = duration[ref] - duration[u]
        if explicit and guess < 0:
            reason = (f"Starts before the session reference by {format_offset(-guess)} "
                      f"if it stopped when {ref.name} ended (a guess: no sound match)")
        else:
            reason = f"Guess: assumes this recorder stopped when {ref.name} ended"
        out[u] = Suggestion(u, guess, "guess", reason)

    if explicit:
        rows = [out[u] for u in units if u in out]
        for row in rows:
            if row.offset is not None:
                row.offset = round(row.offset, 3)
        return AlignmentResult(rows, SESSION_REFERENCE, ref)

    out[ref] = Suggestion(ref, 0.0, "reference", "Reference (longest recording)")
    measured = [out[u] for u in units if u in out]
    anchor = float(current_offsets.get(ref, 0.0))
    offered, shift, note = _anchored(measured, ref, anchor)
    return AlignmentResult(offered, AUTOMATIC_REFERENCE, ref, note, anchor, shift, measured)


# ---------- Worker ----------

def decode_energy(path: Path, stream_index: int | None, should_stop):
    """Frame energies of one stream, every channel added, decoded at DECODE_RATE."""
    from audio import iter_stream_power

    envelope = Envelope()
    for power in iter_stream_power(path, stream_index, DECODE_RATE):
        if should_stop():
            raise Cancelled
        envelope.feed_power(power)
    return envelope.energy()


class AlignWorker(QThread):
    """Works out offset suggestions from the speaker files; applies nothing.

    Each distinct stream is decoded once, at DECODE_RATE. `units` holds one
    (path, stream indices) pair per speaker file, as the table showed them
    when Align was pressed, and `session_reference` the same pair for the
    optional timing-only file. A run ends with `result` (an AlignmentResult)
    set, `error` (a message) set, or — when cancelled — neither, for the
    window to read once QThread.finished arrives.
    """

    progress = Signal(int, str)       # percent, message

    def __init__(self, units: list[tuple[Path, list[int | None]]], current_offsets: dict,
                 session_reference: tuple[Path, list[int | None]] | None = None,
                 parent=None) -> None:
        super().__init__(parent)
        self.units = list(units)
        self.current_offsets = dict(current_offsets)
        self.session_reference = session_reference
        self.result: AlignmentResult | None = None
        self.error = ""
        self.cancelled = False
        self._stop = False

    def stop(self) -> None:
        self._stop = True

    def _should_stop(self) -> bool:
        return self._stop

    def _read(self, path: Path, streams: list[int | None]):
        energy = sum_energies([decode_energy(path, s, self._should_stop) for s in streams])
        if energy.size == 0 or float(energy.max()) < SILENCE_ENERGY:
            raise ValueError("there is no usable audio in it")
        return fingerprint(energy)

    def run(self) -> None:
        try:
            self._run()
        except Cancelled:
            self.cancelled = True
        except Exception as exc:  # a bug, not a bad file: say so rather than hang
            self.error = f"Alignment failed: {exc}"

    def _run(self) -> None:
        reference = self.session_reference[0] if self.session_reference else None
        jobs = ([self.session_reference] if self.session_reference else []) + self.units
        fingerprints: dict = {}
        for number, (path, streams) in enumerate(jobs):
            if self._stop:
                raise Cancelled
            self.progress.emit(int(number * 100 / len(jobs)),
                               f"Aligning: reading {path.name} ({number + 1} of {len(jobs)})…")
            try:
                fingerprints[path] = self._read(path, streams)
            except Cancelled:
                raise
            except Exception as exc:
                if path == reference:
                    self.error = (f"Couldn't use the session reference {path.name}: "
                                  f"{exc}. No offsets were changed.")
                    return
                fingerprints[path] = exc
        units = [path for path, _streams in self.units]
        if reference is None and all(isinstance(fingerprints[u], Exception) for u in units):
            self.error = ("None of the files could be read, so there was nothing "
                          "to compare. No offsets were changed.")
            return
        self.progress.emit(100, "Aligning: comparing recordings…")
        result = suggest_offsets(units, fingerprints, self.current_offsets, reference,
                                 should_stop=self._should_stop)
        if self._stop:
            raise Cancelled
        self.result = result
