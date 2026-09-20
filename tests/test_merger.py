"""The interleaving and coalescing rules everything downstream depends on."""

from __future__ import annotations

import unittest
from pathlib import Path

import tests  # noqa: F401  (puts app/ on sys.path)

from merger import MAX_COALESCE_CHARS, MAX_COALESCE_GAP, coalesce, merge_tracks
from models import Segment, Track


def track(speaker: str, *segments: Segment) -> Track:
    return Track(path=Path(f"{speaker}.flac"), speaker=speaker,
                 segments=list(segments))


class MergeTracks(unittest.TestCase):
    def test_orders_every_track_by_start_time(self):
        alice = track("Alice", Segment("Alice", 0.0, 2.0, "First."),
                      Segment("Alice", 8.0, 9.0, "Third."))
        bob = track("Bob", Segment("Bob", 4.0, 5.0, "Second."))
        merged = merge_tracks([alice, bob])
        self.assertEqual([s.text for s in merged],
                         ["First.", "Second.", "Third."])

    def test_ties_break_on_end_time(self):
        a = track("A", Segment("A", 1.0, 9.0, "long"))
        b = track("B", Segment("B", 1.0, 2.0, "short"))
        self.assertEqual([s.text for s in merge_tracks([a, b])],
                         ["short", "long"])

    def test_empty_tracks_merge_to_nothing(self):
        self.assertEqual(merge_tracks([track("A"), track("B")]), [])


class Coalesce(unittest.TestCase):
    def test_joins_a_speakers_consecutive_segments(self):
        blocks = coalesce([Segment("A", 0.0, 1.0, "One."),
                           Segment("A", 1.2, 2.0, "Two.")])
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0].text, "One. Two.")
        self.assertEqual(blocks[0].end, 2.0)

    def test_a_different_speaker_starts_a_block(self):
        blocks = coalesce([Segment("A", 0.0, 1.0, "Hi."),
                           Segment("B", 1.1, 2.0, "Hello.")])
        self.assertEqual([b.speaker for b in blocks], ["A", "B"])

    def test_a_long_pause_starts_a_block(self):
        blocks = coalesce([Segment("A", 0.0, 1.0, "Before."),
                           Segment("A", 1.0 + MAX_COALESCE_GAP + 1, 99.0, "After.")])
        self.assertEqual(len(blocks), 2)
        self.assertEqual(blocks[1].start, 1.0 + MAX_COALESCE_GAP + 1)

    def test_does_not_mutate_the_input_segments(self):
        segments = [Segment("A", 0.0, 1.0, "One."), Segment("A", 1.1, 2.0, "Two.")]
        coalesce(segments)
        self.assertEqual(segments[0].text, "One.")
        self.assertEqual(segments[0].end, 1.0)

    def test_a_solo_speaker_is_still_cut_into_blocks(self):
        """The invariant llm.py's line-boundary splitting relies on.

        One person talking without a pause would otherwise coalesce into a
        single unsplittable line, and a notes part is cut at line boundaries
        — so a long solo recording would overflow the model's context every
        time. MAX_COALESCE_CHARS is what stops that, across two modules.
        """
        segments = [Segment("A", i * 1.0, i * 1.0 + 0.9, "word " * 20)
                    for i in range(200)]
        blocks = coalesce(segments)
        self.assertGreater(len(blocks), 1)
        longest = max(len(b.text) for b in blocks)
        # A block is only checked *before* appending, so it can overshoot by
        # one segment — but not by an order of magnitude.
        self.assertLess(longest, MAX_COALESCE_CHARS * 2)


if __name__ == "__main__":
    unittest.main()
