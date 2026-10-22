"""Track offsets: what the Offset column accepts, and how Align suggests them.

The matcher tests run on synthetic frame energies: talkers made of syllables
of random length and loudness, mixed and cropped the way real sessions are.
They pin the logic — lag direction, what counts as strong, what may chain —
not the thresholds, which are still to be calibrated on real recordings.
"""

from __future__ import annotations

import os
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

import numpy as np

import tests  # noqa: F401  (puts app/ on sys.path)

import align
from align import (
    AUTOMATIC_REFERENCE,
    FRAME_RATE,
    SESSION_REFERENCE,
    AlignWorker,
    Cancelled,
    Envelope,
    Match,
    find_offset,
    fingerprint,
    format_offset,
    parse_offset,
    suggest_offsets,
)


class ParseOffset(unittest.TestCase):
    def test_accepted_forms(self):
        cases = {
            "150": 150.0,
            "150.25": 150.25,
            "2:30": 150.0,
            "2:30.5": 150.5,
            "2:30,5": 150.5,          # decimal comma
            "1:02:30": 3750.0,
            "1:02:30.250": 3750.25,
            "90:00": 5400.0,          # the first field can be any size
            "30:00:00": 108000.0,     # hours are not limited to 24
            "0": 0.0,
            "  2:30  ": 150.0,
            "0:00": 0.0,
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertEqual(parse_offset(text), expected)

    def test_rounds_to_milliseconds(self):
        self.assertEqual(parse_offset("1.23456"), 1.235)

    def test_rejected_forms_say_why(self):
        cases = {
            "": "0 to reset",
            "   ": "0 to reset",
            "-5": "negative",
            "-0:30": "negative",
            "abc": "isn't a time",
            "nan": "isn't a time",
            "inf": "isn't a time",
            "1:75": "under 60",
            "1:60": "under 60",
            "1:02:60": "under 60",
            "1:2:3:4": "at most",
            "1.5:30": "isn't a time",  # only the last field takes a fraction
            "2:": "isn't a time",
            ":30": "isn't a time",
            "2:30.": "isn't a time",
            "1e3": "isn't a time",
            "+30": "isn't a time",
            "\u0661\u0665\u0660": "isn't a time",  # non-ASCII digits
            "9" * 400: "too large",
            "9" * 400 + ":00": "too large",
            "1" + "0" * 309: "too large",
            "999999999": "too large",   # over 100 days
        }
        for text, reason in cases.items():
            with self.subTest(text=text):
                with self.assertRaises(ValueError) as caught:
                    parse_offset(text)
                self.assertIn(reason, str(caught.exception))


class FormatOffset(unittest.TestCase):
    def test_standard_form(self):
        cases = {
            0: "0:00",
            5: "0:05",
            150: "2:30",
            150.5: "2:30.5",
            312.4: "5:12.4",
            150.25: "2:30.25",
            150.125: "2:30.125",
            3750: "1:02:30",
            108000: "30:00:00",       # no 24-hour wrap
            59.9999: "1:00",          # rounds to milliseconds first
        }
        for seconds, expected in cases.items():
            with self.subTest(seconds=seconds):
                self.assertEqual(format_offset(seconds), expected)

    def test_round_trips(self):
        for seconds in (0.0, 0.001, 1.5, 59.999, 61.0, 150.25, 3599.5, 3750.125, 90061.0):
            with self.subTest(seconds=seconds):
                self.assertEqual(parse_offset(format_offset(seconds)), seconds)


# ---------- Synthetic sessions ----------

def speech(seconds: float, seed: int, talk: float = 0.35):
    """Frame energies of one talker: turns of 1-8 s made of uneven syllables."""
    rng = np.random.default_rng(seed)
    n = int(seconds * FRAME_RATE)
    energy = np.zeros(n)
    t, on = 0, rng.random() < talk
    while t < n:
        run = int(rng.uniform(1, 8) * FRAME_RATE)
        if on:
            s, stop = t, min(n, t + run)
            while s < stop:
                length = int(rng.uniform(0.08, 0.3) * FRAME_RATE)
                energy[s:min(s + length, stop)] = rng.uniform(0.1, 1.0)
                s += length + int(rng.uniform(0.02, 0.15) * FRAME_RATE)
        t += run
        on = rng.random() < (talk if not on else 0.3)
    return energy * rng.gamma(10, 0.1, n) * 1e-2


def hiss(seconds: float, seed: int):
    """A quiet microphone's own noise floor, about -60 dBFS."""
    return np.random.default_rng(seed).gamma(40, 1 / 40, int(seconds * FRAME_RATE)) * 1e-6


SESSION = 600          # seconds
LATE = 7345            # frames: 73.45 s


class _Session(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.a, cls.b, cls.c, cls.d = (speech(SESSION, seed) for seed in (1, 2, 3, 4))
        cls.mix = cls.a + 0.5 * cls.b + 0.3 * cls.c + hiss(SESSION, 10)
        cls.late = (cls.c + hiss(SESSION, 11))[LATE:]

    def assertNotWronglyStrong(self, match: Match, expected_lag: float | None):
        if expected_lag is None:
            self.assertNotEqual(match.strength, "strong")
        elif match.strength == "strong":
            self.assertAlmostEqual(match.lag, expected_lag, places=2)


class EnvelopeTests(unittest.TestCase):
    def test_same_result_in_one_piece_or_in_chunks(self):
        samples = np.random.default_rng(0).standard_normal(10007)
        whole = Envelope()
        whole.feed(samples)
        for size in (1, 79, 333):
            with self.subTest(size=size):
                pieces = Envelope()
                for i in range(0, samples.size, size):
                    pieces.feed(samples[i:i + size])
                np.testing.assert_allclose(pieces.energy(), whole.energy())
        self.assertEqual(whole.energy().size, 10007 // 80)

    def test_frame_energy_is_the_mean_square(self):
        envelope = Envelope()
        envelope.feed(np.full(800, 0.5))
        np.testing.assert_allclose(envelope.energy(), np.full(10, 0.25))

    def test_power_fed_directly_matches_samples(self):
        samples = np.random.default_rng(1).standard_normal(4000)
        from_samples, from_power = Envelope(), Envelope()
        from_samples.feed(samples)
        from_power.feed_power(samples * samples)
        np.testing.assert_allclose(from_power.energy(), from_samples.energy())

    def test_a_damaged_sample_costs_one_frame_not_the_rest_of_the_file(self):
        samples = np.full(800, 0.5)
        samples[85] = np.nan
        samples[400] = np.inf
        energy = Envelope()
        energy.feed(samples)
        self.assertTrue(np.all(np.isfinite(energy.energy())))
        self.assertEqual(int(np.sum(energy.energy() == 0.25)), 8)

    def test_streams_of_one_file_are_heard_together(self):
        np.testing.assert_allclose(align.sum_energies([np.ones(3), np.ones(5)]),
                                   [2, 2, 2, 1, 1])


class FindOffset(_Session):
    def test_finds_a_late_start_in_a_full_session_mix(self):
        match = find_offset(fingerprint(self.mix), fingerprint(self.late),
                            require_window_agreement=True)
        self.assertEqual(match.strength, "strong")
        self.assertAlmostEqual(match.lag, 73.45, places=2)
        self.assertEqual((match.windows_agreeing, match.windows_checked), (3, 3))

    def test_lag_direction(self):
        # The cropped copy began 30 s after the original did, not before.
        original = self.a + hiss(SESSION, 12)
        cropped = original[30 * FRAME_RATE:]
        self.assertAlmostEqual(find_offset(fingerprint(original), fingerprint(cropped)).lag, 30.0)
        self.assertAlmostEqual(find_offset(fingerprint(cropped), fingerprint(original)).lag, -30.0)

    def test_recordings_that_share_no_sound_are_no_match(self):
        match = find_offset(fingerprint(self.a + hiss(SESSION, 13)),
                            fingerprint(self.b + hiss(SESSION, 14)))
        self.assertEqual(match.strength, "none")

    def test_a_speaker_absent_from_the_mix_is_not_matched(self):
        absent = (self.d + hiss(SESSION, 15))[LATE:]
        match = find_offset(fingerprint(self.mix), fingerprint(absent), require_window_agreement=True)
        self.assertNotWronglyStrong(match, None)

    def test_leading_silence_is_kept_as_timing(self):
        for name, quiet_start in (("hiss", hiss(60, 17)), ("digital silence", np.zeros(6000))):
            with self.subTest(name):
                own = self.c + hiss(SESSION, 16)
                own[:6000] = quiet_start
                match = find_offset(fingerprint(self.mix), fingerprint(own),
                                    require_window_agreement=True)
                self.assertEqual(match.strength, "strong")
                self.assertEqual(match.lag, 0.0)

    def test_a_speaker_who_also_left_early(self):
        own = (self.c + hiss(SESSION, 18))[LATE:LATE + 300 * FRAME_RATE]
        match = find_offset(fingerprint(self.mix), fingerprint(own), require_window_agreement=True)
        self.assertEqual(match.strength, "strong")
        self.assertAlmostEqual(match.lag, 73.45, places=2)

    def test_a_quiet_speaker_is_never_placed_wrongly(self):
        for gain in (0.05, 0.01, 0.002):
            with self.subTest(gain=gain):
                mix = self.a + 0.5 * self.b + gain * self.c + hiss(SESSION, 10)
                match = find_offset(fingerprint(mix), fingerprint(self.late),
                                    require_window_agreement=True)
                self.assertNotWronglyStrong(match, 73.45)

    def test_consistency_check_flags_an_edited_file(self):
        # Two people in one room; Bob's file lost 0.3 s two thirds of the way in.
        length = 1500
        alice, bob = speech(length, 40), speech(length, 41)
        room = alice + 0.1 * bob + hiss(length, 42)
        bobs_file = bob + 0.1 * alice + hiss(length, 43)
        whole = find_offset(fingerprint(room), fingerprint(bobs_file[5000:125000]))
        self.assertEqual((whole.strength, whole.lag), ("strong", 50.0))
        edited = np.concatenate([bobs_file[5000:85000], bobs_file[85030:125000]])
        match = find_offset(fingerprint(room), fingerprint(edited))
        self.assertEqual(match.strength, "weak")
        self.assertIn("start and end disagree by 0.3 s", match.detail)
        # Long cuts are caught too, wherever they fall: each part searches
        # every lag, and a part straddling the cut isn't outvoted by the
        # agreeing side of it. A quarter of the way in, two thirds, and late
        # enough that the last third mostly agrees and only the sixth doesn't.
        for cut_at in (35000, 85000, 105000):
            for cut in (30, 6000):
                with self.subTest(cut_at=cut_at, cut=cut):
                    edited = np.concatenate([bobs_file[5000:cut_at],
                                             bobs_file[cut_at + cut:125000]])
                    match = find_offset(fingerprint(room), fingerprint(edited))
                    self.assertEqual(match.strength, "weak")
                    self.assertIn(f"start and end disagree by {cut / FRAME_RATE:.1f} s",
                                  match.detail)
        # 30 s lost 90 s in: the first sixth straddles the cut almost evenly,
        # two real peaks with neither standing out from the other.
        # And 60 s from either end, where only the 80 s end part sees it.
        for kept in ((14000, 17000), (11000, 14000), (116000, 119000)):
            with self.subTest(kept=kept):
                edited = np.concatenate([bobs_file[5000:kept[0]], bobs_file[kept[1]:125000]])
                match = find_offset(fingerprint(room), fingerprint(edited))
                self.assertEqual(match.strength, "weak")
                self.assertIn("start and end disagree by 30.0 s", match.detail)

    def test_a_short_edited_recording_is_checked(self):
        # A file that lost 5 s 50 s in. Under 240 s of overlap no third is
        # 80 s long, but the first and last 80 s fit without overlapping, and
        # the first finds the cut. Under 160 s there is no room for two: an
        # edit goes unseen there, which is the known limit, pinned so that it
        # changes on purpose. The same files unedited stay strong throughout.
        for length, strength in ((158, "strong"), (162, "weak"), (180, "weak"),
                                 (238, "weak"), (242, "weak")):
            with self.subTest(length=length):
                energy = speech(length + 5, 1, talk=0.8) + hiss(length + 5, 50)
                edited = np.concatenate([energy[:5000], energy[5500:]])
                match = find_offset(fingerprint(energy), fingerprint(edited))
                self.assertEqual((match.strength, match.lag), (strength, 5.0))
                if strength == "weak":
                    self.assertIn("start and end disagree by 5.0 s", match.detail)
                clean = find_offset(fingerprint(energy), fingerprint(energy[500:]))
                self.assertEqual((clean.strength, clean.lag), ("strong", 5.0))

    def test_part_sizes_around_the_boundaries(self):
        # 80 s parts once two fit; thirds and smaller once a third is 80 s.
        # A fraction that comes out at 80 s is the same part, checked once.
        cases = {15999: [], 16000: [8000], 23999: [8000], 24000: [8000],
                 24003: [8001, 8000], 48000: [16000, 8000], 96000: [32000, 16000, 8000]}
        for overlap, sizes in cases.items():
            with self.subTest(overlap=overlap):
                self.assertEqual(align._part_sizes(overlap), sizes)

    def test_a_sparse_speaker_is_not_contradicted_by_a_chance_peak(self):
        # Says little, mostly late on: in the last third a chance lag comes
        # first and the true one only second. That is too little to go on,
        # not a part of the file that matches somewhere else.
        base = 13_022_360
        a, b, c = speech(1200, base), speech(1200, base + 1), speech(1200, base + 2, talk=0.03)
        mix = a + 0.5 * b + 0.3 * c + hiss(1200, base + 3)
        own = (c + 10 * hiss(1200, base + 4))[6321:]
        match = find_offset(fingerprint(mix), fingerprint(own), require_window_agreement=True)
        self.assertEqual((match.strength, match.lag), ("strong", 63.21))

    def test_a_part_with_little_sound_cannot_contradict(self):
        # A gated headset, silent but for a few words in its first and last
        # 221 s: the small end parts hold only those, and their chance peaks
        # are no evidence against the rest.
        n, edge = 1400 * FRAME_RATE, 22130
        a, c = speech(1400, 1477), speech(1400, 1479)
        keep = np.zeros(n, bool)
        keep[edge:n - edge] = True
        for word in (38307, 133628, 115608, 37745, 9803, 124772, 76737):
            keep[word:word + 40] = True
        c = np.where(keep, c, 0.0)
        room = a + 0.1 * c + hiss(1400, 1480)
        match = find_offset(fingerprint(room), fingerprint(c[3372:]))
        self.assertEqual((match.strength, match.lag), ("strong", 33.72))

    def test_edits_found_by_the_smaller_parts(self):
        # Sparse talkers against a session reference whose files lost 5 s or
        # 30 s some 75 s in. The first is seen by a twelfth (the sixth
        # straddles the cut and mostly agrees); in the second the part's
        # match elsewhere is clear but not strong, and that is enough.
        for seed, talk, late, cut, at in ((335, 0.08, 7253, 500, 9605),
                                          (109, 0.04, 4726, 3000, 7252)):
            with self.subTest(seed=seed):
                a, b = speech(1600, 5 * seed + 31), speech(1600, 5 * seed + 32)
                c = speech(1600, 5 * seed + 33, talk=talk)
                mix = a + 0.5 * b + 0.3 * c + hiss(1600, 5 * seed + 34)
                own = (c + hiss(1600, 5 * seed + 35))[late:]
                edited = np.concatenate([own[:at], own[at + cut:]])
                match = find_offset(fingerprint(mix), fingerprint(edited),
                                    require_window_agreement=True)
                self.assertEqual(match.strength, "weak")
                self.assertIn(f"start and end disagree by {cut / FRAME_RATE:.1f} s", match.detail)

    def test_two_peaks_means_the_true_lag_is_one_of_them(self):
        # Stand-in fingerprints of bare onsets, every stretch of the reference
        # holding 20. The part finds 16 of its onsets at one lag and 10 at a
        # third; at its true lag, 15 or 12. Neither leader stands out from the
        # other. With 15 both stand out from everything else: a cut. With 12
        # the true lag is no clearer than chance, so the leader is a lone
        # chance peak, and that is too little to go on, not a contradiction.
        rng = np.random.default_rng(0)
        size, length, true_lag = 20000, 80000, 30000
        onsets = rng.choice(size, 100, replace=False)
        part = np.zeros(size)
        part[onsets] = 1.0
        for at_true, verdict in ((15, "disagrees"), (12, "unclear")):
            with self.subTest(at_true=at_true):
                ref = np.zeros(length)
                for lag, count in ((5000, 16), (true_lag, at_true), (55000, 10)):
                    ref[lag + onsets[:count]] = 1.0
                    spare = np.setdiff1d(np.arange(size), onsets)
                    ref[lag + rng.choice(spare, 20 - count, replace=False)] = 1.0
                self.assertEqual(align._part_verdict(ref, part, 0, size, true_lag)[0], verdict)

    def test_a_part_is_only_scored_where_it_lies_inside_the_reference(self):
        # The first 60 s of this part are a copy of the reference's last 60 s:
        # slid to the reference's very end, half the part matches perfectly.
        # Scored on that sliver it would contradict the true lag.
        ref = fingerprint(self.mix)
        other = fingerprint(self.late).copy()
        other[:6000] = ref[-6000:]
        self.assertNotEqual(align._part_verdict(ref, other, 0, 12000, LATE)[0], "disagrees")

    def test_two_mostly_silent_files_are_not_a_strong_match(self):
        # Gated headsets: a handful of words each, digital silence between.
        # Two onsets meeting by chance outscore one; scored alone, that has
        # passed for strong.
        def gated(seed, words):
            rng = np.random.default_rng(seed)
            energy = np.zeros(600 * FRAME_RATE)
            for at in rng.uniform(0, 599, words):
                start = int(at * FRAME_RATE)
                energy[start:start + int(rng.uniform(0.2, 0.6) * FRAME_RATE)] = rng.uniform(0.1, 1)
            return energy * 1e-2
        # The silence between can be digital zeros, or a floor far too quiet
        # to hear (dither, a noise suppressor's residue): neither is sound.
        for floor in (0.0, 1e-10):
            for seed, words in ((24, 3), (109, 5), (139, 3)):
                with self.subTest(seed=seed, floor=floor):
                    ref, other = (gated(s, words) + hiss(600, s) * floor * 1e6
                                  for s in (seed, 10000 + seed))
                    match = find_offset(fingerprint(ref), fingerprint(other))
                    self.assertNotEqual(match.strength, "strong")

    def test_seconds_of_sound_are_counted_however_little_there_is(self):
        # Half a minute of speech in an hour, between digital silence or a
        # microphone's hiss: either way it is the seconds of speech that
        # count, not none of them (the busiest 1% of seconds are silent) and
        # not every second of hiss.
        speaking = np.zeros(3600 * FRAME_RATE)
        for second in np.random.default_rng(3).choice(3600, 30, replace=False):
            speaking[second * FRAME_RATE:second * FRAME_RATE + 30] = 0.01
        for name, floor in (("digital silence", 0.0), ("hiss", 1.0)):
            with self.subTest(name):
                fp = fingerprint(speaking + floor * hiss(3600, 4))
                active = align._active_seconds(fp, 0, fp.size).size
                self.assertGreaterEqual(active, 30)
                self.assertLessEqual(active, 60)    # a word's rise can spill into the next second

    def test_a_peak_resting_on_little_overlap_is_not_strong(self):
        # They share 150 s of sound: the end of one and the start of the other.
        shared = self.c[-15000:] + hiss(150, 23)
        ref = np.concatenate([speech(450, 24) + hiss(450, 25), shared])
        other = np.concatenate([shared, speech(450, 26)])
        match = find_offset(fingerprint(ref), fingerprint(other))
        self.assertEqual(match.lag, 450.0)
        self.assertEqual(match.strength, "weak")
        self.assertIn("only 150 s of overlap", match.detail)

    def test_repeated_audio_is_not_a_strong_match(self):
        loop = speech(120, 30)
        repeated = np.tile(loop, 5)
        match = find_offset(fingerprint(repeated + hiss(600, 31)),
                            fingerprint((repeated + hiss(600, 32))[4000:40000]),
                            require_window_agreement=True)
        self.assertNotWronglyStrong(match, None)


class WindowAgreement(_Session):
    """Against a session reference, separate parts must each find the lag again."""

    def test_too_little_audio_to_confirm_stays_weak(self):
        short = (self.c + hiss(SESSION, 19))[LATE:LATE + 60 * FRAME_RATE]
        whole = find_offset(fingerprint(self.mix), fingerprint(short))
        self.assertEqual(whole.strength, "strong")  # the same peak, unconfirmed
        match = find_offset(fingerprint(self.mix), fingerprint(short), require_window_agreement=True)
        self.assertEqual(match.strength, "weak")
        self.assertIn("too little audio", match.detail)

    def test_stretches_come_from_where_the_speaker_talks(self):
        # Talking only in the first third, hiss after: confirmed there, not
        # failed by stretches of hiss.
        own = self.late.copy()
        third = own.size // 3
        own[third:] = hiss(SESSION, 20)[:own.size - third]
        fp = fingerprint(own)
        active = align._active_seconds(fp, 0, fp.size)
        self.assertLess(active.max(), third // FRAME_RATE + 1)
        match = find_offset(fingerprint(self.mix), fp, require_window_agreement=True)
        self.assertEqual(match.strength, "strong")
        self.assertEqual((match.windows_agreeing, match.windows_checked), (3, 3))

    def test_silence_never_counts_as_activity(self):
        for name, quiet in (("hiss", hiss(SESSION, 27)), ("digital silence", np.zeros(60000))):
            with self.subTest(name):
                fp = fingerprint(quiet + 0)
                busy = np.concatenate([fingerprint(self.mix)[:3000], fp[3000:]])
                active = align._active_seconds(busy, 0, busy.size)
                self.assertLess(active.max(), 30)

    def test_a_speaker_heard_in_the_mix_for_only_part_of_their_file_is_weak(self):
        # Muted in the call a quarter of the way into their own file: only
        # one stretch can find them, and one is not enough.
        in_mix = self.c.copy()
        in_mix[LATE + (self.c.size - LATE) // 4:] = 0
        mix = self.a + 0.5 * self.b + 0.3 * in_mix + hiss(SESSION, 10)
        match = find_offset(fingerprint(mix), fingerprint(self.late), require_window_agreement=True)
        self.assertEqual(match.strength, "weak")
        self.assertEqual((match.windows_agreeing, match.windows_checked), (1, 3))
        self.assertIn("only 1 of 3 separate parts of the file confirm it", match.detail)

    def conflict(self, ref, other):
        """find_offset's verdict, and what the stretches alone make of it."""
        match = find_offset(ref, other, require_window_agreement=True)
        lag = int(round(match.lag * FRAME_RATE))
        start, end = max(0, -lag), min(other.size, ref.size - lag)
        return match, align._window_agreement(ref, other, lag, start, end)[3]

    def test_a_short_cut_is_a_conflict(self):
        own = self.c + hiss(SESSION, 11)
        cut = np.concatenate([own[LATE:30000], own[30030:]])      # 0.3 s lost
        match, conflict = self.conflict(fingerprint(self.mix), fingerprint(cut))
        self.assertEqual(match.strength, "weak")
        self.assertIn("0.3 s", match.detail)
        self.assertAlmostEqual(conflict, 0.3)

    def test_each_disagreeing_stretch_counts_however_it_scores(self):
        # 30 s cut out of a 500 s file. In the first the stretch holding the
        # cut finds the other lag less clearly than a strong match would; in
        # the second it splits nearly evenly across the cut, two real peaks
        # neither of which stands out from the other.
        for seed, start, cut_at in ((677, 4241, 33786), (11, 3246, 8625)):
            with self.subTest(seed=seed):
                a, b, c = (speech(500, seed * 3 + 700000 + k) for k in range(3))
                mix = a + 0.5 * b + 0.3 * c + hiss(500, seed + 800000)
                own = c + hiss(500, seed + 900000)
                edited = np.concatenate([own[start:cut_at], own[cut_at + 3000:]])
                match, conflict = self.conflict(fingerprint(mix), fingerprint(edited))
                self.assertEqual(match.strength, "weak")
                self.assertIn("30.0 s", match.detail)
                self.assertAlmostEqual(conflict, 30.0)

    def test_a_part_matching_somewhere_else_is_a_conflict(self):
        length, start = 900, 2000
        mix = speech(length, 1) + 0.5 * speech(length, 2) + 0.3 * speech(length, 3) + hiss(length, 10)
        own = speech(length, 3) + hiss(length, 21)
        # 300 s, then a 60 s gap cut out, then 200 s more; and 140 s, the gap,
        # then 400 s.
        for first, after in ((30000, 20000), (14000, 40000)):
            with self.subTest(first=first):
                edited = np.concatenate([own[start:start + first],
                                         own[start + first + 6000:start + first + 6000 + after]])
                match, conflict = self.conflict(fingerprint(mix), fingerprint(edited))
                self.assertEqual(match.strength, "weak")
                self.assertIn("60.0 s", match.detail)
                self.assertAlmostEqual(conflict, 60.0)

    def test_parts_from_different_places_are_not_strong(self):
        own = self.c + hiss(SESSION, 21)
        spliced = np.concatenate([own[1000:15000], own[40000:54000], own[20000:34000]])
        match = find_offset(fingerprint(self.mix), fingerprint(spliced),
                            require_window_agreement=True)
        self.assertNotWronglyStrong(match, None)


class _Fp:
    """A stand-in fingerprint: suggest_offsets only measures it and passes it on."""

    def __init__(self, name: str, seconds: float) -> None:
        self.name, self.frames = name, int(seconds * FRAME_RATE)

    def __len__(self) -> int:
        return self.frames


class _Matcher:
    """Scripted matches between stand-in fingerprints, keyed (reference, other)."""

    def __init__(self, **edges: tuple[str, float]) -> None:
        self.edges = {tuple(key.split("_")): Match(lag=lag, strength=strength, ratio=2.5)
                      for key, (strength, lag) in edges.items()}
        self.calls: list[tuple[str, str, bool]] = []

    def __call__(self, ref, other, require_window_agreement=False) -> Match:
        self.calls.append((ref.name, other.name, require_window_agreement))
        return self.edges.get((ref.name, other.name), Match())


def _run(lengths: dict, matcher: _Matcher, offsets: dict | None = None,
         reference: str | None = None, broken: dict | None = None):
    paths = {name: Path(f"{name}.wav") for name in lengths}
    fingerprints = {paths[n]: _Fp(n, sec) for n, sec in lengths.items()}
    for name, exc in (broken or {}).items():
        fingerprints[paths[name]] = exc
    units = [p for n, p in paths.items() if n != reference]
    result = suggest_offsets(units, fingerprints,
                             {paths[n]: v for n, v in (offsets or {}).items()},
                             paths[reference] if reference else None, matcher=matcher)
    return result, {s.path.stem: s for s in result.suggestions}


class SuggestOffsets(unittest.TestCase):
    def test_chains_through_a_strong_match(self):
        matcher = _Matcher(a_b=("strong", 5.0), b_c=("strong", 2.0))
        result, rows = _run({"a": 100, "b": 90, "c": 80}, matcher)
        self.assertEqual(result.mode, AUTOMATIC_REFERENCE)
        self.assertEqual(rows["a"].method, "reference")
        self.assertEqual((rows["b"].offset, rows["c"].offset), (5.0, 7.0))
        self.assertIn("Matched sound with b.wav", rows["c"].reason)
        self.assertTrue(all(not agree for *_names, agree in matcher.calls))

    def test_never_chains_through_a_weak_match(self):
        matcher = _Matcher(a_b=("weak", 5.0), b_c=("strong", 2.0))
        _result, rows = _run({"a": 100, "b": 90, "c": 80}, matcher)
        self.assertEqual((rows["b"].method, rows["b"].strength), ("matched", "weak"))
        self.assertEqual(rows["c"].method, "guess")
        self.assertNotIn(("b", "c", False), matcher.calls)

    def test_a_weak_match_is_kept_rather_than_replaced_by_a_guess(self):
        _result, rows = _run({"a": 100, "b": 40}, _Matcher(a_b=("weak", 5.0)))
        self.assertEqual((rows["b"].offset, rows["b"].strength), (5.0, "weak"))

    def test_the_guess_assumes_everyone_stopped_together(self):
        _result, rows = _run({"a": 100, "b": 40}, _Matcher(), offsets={"a": 10.0})
        self.assertEqual((rows["b"].method, rows["b"].offset), ("guess", 70.0))
        self.assertIn("stopped when a.wav ended", rows["b"].reason)

    def test_anchors_on_the_automatic_references_current_offset(self):
        # The reference starts at 10:00; an event 3:00 into it is 1:00 into b.
        _result, rows = _run({"a": 1000, "b": 500}, _Matcher(a_b=("strong", 120.0)),
                             offsets={"a": 600.0})
        self.assertEqual(rows["a"].offset, 600.0)
        self.assertEqual(format_offset(rows["b"].offset), "12:00")

    def test_without_a_reference_everything_moves_up_to_zero(self):
        result, rows = _run({"a": 100, "b": 90}, _Matcher(a_b=("strong", -30.0)))
        self.assertEqual((rows["a"].offset, rows["b"].offset), (30.0, 0.0))
        self.assertIn("moved 0:30 later", result.note)

    def test_a_weak_match_does_not_move_everyone_else(self):
        result, rows = _run({"a": 100, "b": 90, "c": 80},
                            _Matcher(a_b=("weak", -30.0), a_c=("strong", 5.0)))
        self.assertEqual((result.shift, result.note), (0.0, ""))
        self.assertEqual((rows["a"].offset, rows["c"].offset), (0.0, 5.0))
        self.assertEqual(rows["b"].offset, -30.0)
        self.assertFalse(rows["b"].applicable)
        self.assertIn("would start 0:30 before 0:00", rows["b"].reason)

    def test_rebasing_follows_a_new_reference_offset(self):
        result, _rows = _run({"a": 100, "b": 90, "c": 80},
                             _Matcher(a_b=("strong", -30.0), a_c=("strong", 5.0)))
        self.assertEqual(result.shift, 30.0)
        moved = align.rebase(result, 120.0)
        rows = {s.path.stem: s.offset for s in moved.suggestions}
        self.assertEqual(rows, {"a": 120.0, "b": 90.0, "c": 125.0})
        self.assertEqual((moved.anchor, moved.shift, moved.note), (120.0, 0.0, ""))
        # The original result is untouched, and the reference result can't move.
        self.assertEqual({s.path.stem: s.offset for s in result.suggestions},
                         {"a": 30.0, "b": 0.0, "c": 35.0})
        session, _rows = _run({"session": 100, "b": 90}, _Matcher(session_b=("strong", 5.0)),
                              reference="session")
        self.assertIs(align.rebase(session, 50.0), session)

    def test_a_session_reference_wins_even_when_shorter(self):
        result, rows = _run({"session": 1000, "a": 2000}, _Matcher(session_a=("strong", 0.0)),
                            reference="session")
        self.assertEqual(result.mode, SESSION_REFERENCE)
        self.assertEqual(result.reference, Path("session.wav"))
        self.assertEqual(list(rows), ["a"])  # the reference is no speaker row
        self.assertEqual(rows["a"].method, "matched")

    def test_the_session_reference_fixes_zero(self):
        # A sentence 17:10 into the reference is 2:10 into Bob's file.
        matcher = _Matcher(session_bob=("strong", 17 * 60 + 10 - (2 * 60 + 10)))
        result, rows = _run({"session": 3600, "bob": 3000}, matcher,
                            offsets={"bob": 99.0}, reference="session")
        self.assertEqual(format_offset(rows["bob"].offset), "15:00")
        self.assertEqual(result.note, "")
        self.assertEqual(matcher.calls, [("session", "bob", True)])

    def test_every_edge_needs_window_agreement_with_a_session_reference(self):
        matcher = _Matcher(session_b=("strong", 5.0), b_c=("strong", 2.0))
        _result, rows = _run({"session": 100, "b": 90, "c": 80}, matcher, reference="session")
        self.assertEqual(rows["c"].offset, 7.0)
        self.assertTrue(all(agree for *_names, agree in matcher.calls))

    def test_a_file_starting_before_the_session_reference_cannot_apply(self):
        matcher = _Matcher(session_b=("strong", -5.0), b_c=("strong", 3.0),
                           session_d=("strong", 10.0))
        result, rows = _run({"session": 100, "b": 90, "c": 50, "d": 60}, matcher,
                            reference="session")
        self.assertEqual(rows["b"].offset, -5.0)
        self.assertFalse(rows["b"].applicable)
        self.assertIn("Starts before the session reference by 0:05", rows["b"].reason)
        # Not a stepping stone, and no guess hides it.
        self.assertEqual(rows["b"].method, "matched")
        self.assertNotIn(("b", "c", True), matcher.calls)
        self.assertEqual(rows["c"].method, "guess")
        # Zero stays put and the valid match still applies.
        self.assertEqual((rows["d"].offset, rows["d"].applicable), (10.0, True))
        self.assertEqual(result.note, "")

    def test_a_strong_start_before_the_reference_is_not_chained_around(self):
        # b is strongly before the reference; d would place it after it.
        # In either table order: b tried before d is placed, or after.
        for lengths in ({"session": 100, "b": 90, "d": 60}, {"session": 100, "d": 60, "b": 90}):
            with self.subTest(order=list(lengths)[1:]):
                matcher = _Matcher(session_b=("strong", -5.0), session_d=("strong", 10.0),
                                   d_b=("strong", 30.0))
                _result, rows = _run(lengths, matcher, reference="session")
                self.assertEqual(rows["b"].offset, -5.0)
                self.assertFalse(rows["b"].applicable)
                self.assertIn("Starts before the session reference", rows["b"].reason)
                self.assertNotIn(("d", "b", True), matcher.calls)

    def test_a_negative_guess_against_the_session_reference_cannot_apply(self):
        _result, rows = _run({"session": 100, "long": 150}, _Matcher(), reference="session")
        self.assertEqual(rows["long"].offset, -50.0)
        self.assertFalse(rows["long"].applicable)
        self.assertIn("Starts before the session reference by 0:50 if it stopped when "
                      "session.wav ended", rows["long"].reason)

    def test_an_unreadable_file_only_fails_its_own_row(self):
        _result, rows = _run({"a": 100, "b": 90, "c": 80}, _Matcher(a_c=("strong", 4.0)),
                             broken={"b": OSError("bad header")})
        self.assertEqual((rows["b"].method, rows["b"].offset), ("error", None))
        self.assertFalse(rows["b"].applicable)
        self.assertIn("bad header", rows["b"].reason)
        self.assertEqual(rows["c"].offset, 4.0)

    def test_offsets_are_kept_to_milliseconds(self):
        _result, rows = _run({"a": 100, "b": 90}, _Matcher(a_b=("strong", 0.1 + 0.2)))
        self.assertEqual(rows["b"].offset, 0.3)

    def test_cancelling_stops_the_comparisons(self):
        paths = [Path("a.wav"), Path("b.wav")]
        fingerprints = {p: _Fp(p.stem, 10) for p in paths}
        with self.assertRaises(Cancelled):
            suggest_offsets(paths, fingerprints, {}, matcher=_Matcher(), should_stop=lambda: True)


class Worker(_Session):
    """AlignWorker.run() called directly, with decoding replaced by energies."""

    def _worker(self, energies: dict, reference: str | None = None, units=("a", "b")):
        def decode(path, stream, should_stop):
            value = energies[path.stem]
            if isinstance(value, Exception):
                raise value
            return value
        worker = AlignWorker([(Path(f"{u}.wav"), [None]) for u in units], {},
                             (Path(f"{reference}.wav"), [None]) if reference else None)
        with patch("align.decode_energy", side_effect=decode):
            worker.run()
        return worker

    def test_an_unreadable_session_reference_fails_the_run(self):
        for name, value in (("unreadable", OSError("no such file")),
                            ("silent", np.zeros(60000))):
            with self.subTest(name):
                worker = self._worker({"session": value, "a": self.late, "b": self.mix},
                                      reference="session")
                self.assertIsNone(worker.result)
                self.assertIn("session reference session.wav", worker.error)
                self.assertIn("No offsets were changed", worker.error)

    def test_an_unreadable_speaker_file_only_fails_its_row(self):
        worker = self._worker({"session": self.mix, "a": OSError("bad header"),
                               "b": self.late}, reference="session")
        rows = {s.path.stem: s for s in worker.result.suggestions}
        self.assertEqual(rows["a"].method, "error")
        self.assertAlmostEqual(rows["b"].offset, 73.45)

    def test_a_silent_speaker_file_gets_a_reason(self):
        worker = self._worker({"a": self.mix, "b": np.zeros(60000)})
        rows = {s.path.stem: s for s in worker.result.suggestions}
        self.assertIn("no usable audio", rows["b"].reason)

    def test_a_file_that_cannot_be_looked_at_is_not_kept(self):
        # These paths don't exist on disk, so there is no telling later
        # whether they changed.
        worker = self._worker({"a": self.mix, "b": self.late})
        self.assertIsNotNone(worker.result)
        self.assertEqual(worker.fingerprints, {})

    def test_nothing_readable_is_an_error(self):
        worker = self._worker({"a": OSError("x"), "b": OSError("y")})
        self.assertIsNone(worker.result)
        self.assertIn("None of the files could be read", worker.error)

    def test_cancelling_while_reading_the_reference_emits_nothing(self):
        worker = AlignWorker([(Path("a.wav"), [None])], {}, (Path("session.wav"), [None]))

        def chunks(path, stream, rate):
            yield np.zeros(800)
            worker.stop()
            yield np.zeros(800)
            self.fail("decoding went on after cancel")

        with patch("audio.iter_stream_power", side_effect=chunks):
            worker.run()
        self.assertTrue(worker.cancelled)
        self.assertEqual((worker.result, worker.error), (None, ""))


def _shaped_noise(energy, seed: int):
    """8 kHz noise whose frame energies are `energy`."""
    rng = np.random.default_rng(seed)
    hop = align.DECODE_RATE // FRAME_RATE
    return rng.standard_normal(energy.size * hop) * np.sqrt(np.repeat(energy, hop))


def _write_wav(path: Path, energy, seed: int, channels: int = 1, channel: int = 0) -> Path:
    """Noise shaped by frame energies, as 8 kHz 16-bit, on one of `channels`."""
    samples = _shaped_noise(energy, seed)
    frames = np.zeros((samples.size, channels))
    frames[:, channel] = samples
    with wave.open(str(path), "wb") as out:
        out.setnchannels(channels)
        out.setsampwidth(2)
        out.setframerate(align.DECODE_RATE)
        out.writeframes((np.clip(frames, -1, 1) * 32767).astype("<i2").tobytes())
    return path


def _write_float_wav(path: Path, frames) -> Path:
    """32-bit float WAV (which `wave` can't write): NaN and inf survive in it."""
    import struct
    data = np.asarray(frames, dtype="<f4").tobytes()
    channels = frames.shape[1]
    with open(path, "wb") as out:
        out.write(b"RIFF" + struct.pack("<I", 36 + len(data)) + b"WAVE")
        out.write(b"fmt " + struct.pack("<IHHIIHH", 16, 3, channels, align.DECODE_RATE,
                                        align.DECODE_RATE * 4 * channels, 4 * channels, 32))
        out.write(b"data" + struct.pack("<I", len(data)) + data)
    return path


class RealDecode(unittest.TestCase):
    """The whole path through PyAV: files on disk to a suggestion."""

    def _session(self, folder):
        session = speech(300, 50) + 0.5 * speech(300, 51) + hiss(300, 52)
        own = speech(300, 50) + hiss(300, 53)
        return (_write_wav(Path(folder, "session.wav"), session, 1),
                _write_wav(Path(folder, "late.wav"), own[4137:], 2))

    def test_a_second_run_reuses_what_the_first_read(self):
        with tempfile.TemporaryDirectory() as folder:
            reference, late = self._session(folder)
            first = AlignWorker([(late, [None])], {}, (reference, [None]))
            first.run()
            self.assertEqual(set(first.fingerprints), {(reference, (None,)), (late, (None,))})
            again = AlignWorker([(late, [None])], {}, (reference, [None]),
                                known=first.fingerprints)
            with patch("align.decode_energy", side_effect=AssertionError("decoded again")):
                again.run()
        self.assertEqual(again.error, "")
        self.assertEqual(again.result.suggestions, first.result.suggestions)
        self.assertEqual(again.fingerprints, first.fingerprints)

    def test_a_file_changed_since_is_read_again(self):
        with tempfile.TemporaryDirectory() as folder:
            reference, late = self._session(folder)
            first = AlignWorker([(late, [None])], {}, (reference, [None]))
            first.run()
            st = late.stat()
            os.utime(late, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))
            again = AlignWorker([(late, [None])], {}, (reference, [None]),
                                known=first.fingerprints)
            with patch("align.decode_energy", wraps=align.decode_energy) as decode:
                again.run()
            self.assertEqual([c.args[0] for c in decode.call_args_list], [late])
            # Kept against the file as it is now, not as it was.
            self.assertEqual(again.fingerprints[(late, (None,))][0],
                             (late.stat().st_mtime_ns, late.stat().st_size))
        self.assertEqual(again.result.suggestions, first.result.suggestions)

    def test_other_streams_of_the_same_file_are_read_again(self):
        with tempfile.TemporaryDirectory() as folder:
            reference, late = self._session(folder)
            first = AlignWorker([(late, [None])], {}, (reference, [None]))
            first.run()
            again = AlignWorker([(late, [0])], {}, (reference, [None]),
                                known=first.fingerprints)
            with patch("align.decode_energy", wraps=align.decode_energy) as decode:
                again.run()
        self.assertEqual([c.args[:2] for c in decode.call_args_list], [(late, 0)])

    def test_aligns_a_late_file_against_a_session_reference(self):
        with tempfile.TemporaryDirectory() as folder:
            session = speech(300, 50) + 0.5 * speech(300, 51) + hiss(300, 52)
            own = speech(300, 50) + hiss(300, 53)
            reference = _write_wav(Path(folder, "session.wav"), session, 1)
            late = _write_wav(Path(folder, "late.wav"), own[4137:], 2)
            worker = AlignWorker([(late, [None])], {}, (reference, [None]))
            worker.run()
        self.assertEqual(worker.error, "")
        (row,) = worker.result.suggestions
        self.assertEqual((row.method, row.strength), ("matched", "strong"))
        self.assertAlmostEqual(row.offset, 41.37, places=2)

    def test_an_edited_short_file_is_not_strong(self):
        # Three minutes that lost 5 s 50 s in, against the whole recording.
        with tempfile.TemporaryDirectory() as folder:
            energy = speech(180, 1, talk=0.8) + hiss(180, 50)
            whole = _write_wav(Path(folder, "whole.wav"), energy, 1)
            edited = _write_wav(Path(folder, "edited.wav"),
                                np.concatenate([energy[:5000], energy[5500:]]), 2)
            worker = AlignWorker([(whole, [None]), (edited, [None])], {})
            worker.run()
        self.assertEqual(worker.error, "")
        rows = {s.path.stem: s for s in worker.result.suggestions}
        self.assertEqual(rows["edited"].strength, "weak")
        self.assertIn("start and end disagree by 5.0 s", rows["edited"].reason)

    def test_a_stream_that_began_late_keeps_its_place(self):
        # One file holding two speakers' streams, the second begun a minute
        # in, against a room recording that heard both. Read from 0:00, the
        # late stream (the first speaker says little) made the whole file look
        # a minute late — strong, with all three parts agreeing — and applying
        # that moved the first speaker's words a minute late too.
        from tests.test_audio import write_mkv

        rate = align.DECODE_RATE
        a, b = speech(360, 70, talk=0.1), speech(300, 71)
        room = a + np.concatenate([np.zeros(60 * FRAME_RATE), b]) + hiss(360, 72)
        with tempfile.TemporaryDirectory() as folder:
            reference = _write_wav(Path(folder, "room.wav"), room, 1)
            both = write_mkv(Path(folder, "both.mkv"),
                             [[(0, _shaped_noise(a + hiss(360, 73), 2)[:, None])],
                              [(60, _shaped_noise(b + hiss(300, 74), 3)[:, None])]],
                             rate=rate)
            worker = AlignWorker([(both, [0, 1])], {}, (reference, [None]))
            worker.run()
        self.assertEqual(worker.error, "")
        (row,) = worker.result.suggestions
        self.assertEqual((row.strength, row.offset), ("strong", 0.0))

    def test_every_channel_of_a_multichannel_file_is_heard(self):
        # A mono downmix weights the LFE position (the 3rd of three channels,
        # the 4th of six) at zero; that can be someone's microphone.
        for channels, speaker in ((3, 2), (6, 3)):
            with self.subTest(channels=channels), tempfile.TemporaryDirectory() as folder:
                session = speech(300, 50) + 0.5 * speech(300, 51) + hiss(300, 52)
                own = speech(300, 50) + hiss(300, 53)
                reference = _write_wav(Path(folder, "session.wav"), session, 1)
                late = _write_wav(Path(folder, "late.wav"), own[4137:], 2, channels, speaker)
                worker = AlignWorker([(late, [0])], {}, (reference, [None]))
                worker.run()
                self.assertEqual(worker.error, "")
                (row,) = worker.result.suggestions
                self.assertEqual(row.strength, "strong")
                self.assertAlmostEqual(row.offset, 41.37, places=2)

    def test_a_channel_gone_to_nan_leaves_the_others_heard(self):
        # A filter gone unstable on one channel: NaN from 60 s to the end.
        with tempfile.TemporaryDirectory() as folder:
            session = speech(300, 50) + 0.5 * speech(300, 51) + hiss(300, 52)
            own = (speech(300, 50) + hiss(300, 53))[4137:]
            reference = _write_wav(Path(folder, "session.wav"), session, 1)
            noise = _shaped_noise(own, 2)
            frames = np.zeros((noise.size, 3))
            frames[:, 0] = noise
            frames[60 * align.DECODE_RATE:, 1] = np.nan
            late = _write_float_wav(Path(folder, "late.wav"), frames)
            worker = AlignWorker([(late, [None])], {}, (reference, [None]))
            worker.run()
        self.assertEqual(worker.error, "")
        (row,) = worker.result.suggestions
        self.assertEqual(row.strength, "strong")
        self.assertAlmostEqual(row.offset, 41.37, places=2)


if __name__ == "__main__":
    unittest.main()
