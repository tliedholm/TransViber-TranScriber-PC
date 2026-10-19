"""Export formats: timings, escaping, and which files get written."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import tests  # noqa: F401  (puts app/ on sys.path)

import exporters
from merger import merge_tracks
from models import OffsetSource, Segment, Track, TrackStatus


SEGMENTS = [
    Segment("Alice", 0.0, 2.5, "The dragon attacked."),
    Segment("Bob", 3661.25, 3663.0, "We ran <fast> & far."),
]
TRACKS = [Track(path=Path("alice.flac"), speaker="Alice", duration=10.0,
                status=TrackStatus.DONE)]


class Timestamps(unittest.TestCase):
    def test_srt_uses_commas_and_hours(self):
        self.assertEqual(exporters._ts_srt(3661.25), "01:01:01,250")

    def test_vtt_uses_periods(self):
        self.assertEqual(exporters._ts_vtt(3661.25), "01:01:01.250")

    def test_readable_timestamps_drop_a_zero_hour(self):
        self.assertEqual(exporters._hms(75), "1:15")
        self.assertEqual(exporters._hms(3675), "1:01:15")


class Escaping(unittest.TestCase):
    def test_vtt_escapes_markup(self):
        self.assertEqual(exporters._vtt_escape("a <b> & c"), "a &lt;b&gt; &amp; c")

    def test_a_newline_cannot_truncate_a_cue(self):
        self.assertEqual(exporters._one_line("two\nlines"), "two lines")

    def test_vtt_output_has_no_stray_arrow(self):
        text = exporters.to_vtt([Segment("A", 0.0, 1.0, "x --> y")])
        self.assertEqual(text.count("-->"), 1)  # the timing line, nothing else


class Formats(unittest.TestCase):
    def test_srt_numbers_cues_from_one(self):
        lines = exporters.to_srt(SEGMENTS).splitlines()
        self.assertEqual(lines[0], "1")
        self.assertIn("2", lines)

    def test_json_round_trips(self):
        data = json.loads(exporters.to_json(SEGMENTS, TRACKS, "qwen3-4b"))
        self.assertEqual(data["model"], "qwen3-4b")
        self.assertEqual(len(data["segments"]), 2)
        self.assertEqual(data["tracks"][0]["status"], "Done")

    def test_json_records_each_tracks_offset_and_how_it_was_set(self):
        bob = Track(path=Path("bob.m4a"), speaker="Bob", offset=312.4567,
                    offset_source=OffsetSource.MANUAL)
        data = json.loads(exporters.to_json(SEGMENTS, TRACKS + [bob], ""))
        self.assertEqual(data["tracks"][0]["offset"], 0.0)
        self.assertEqual(data["tracks"][0]["offset_source"], "default")
        self.assertEqual(data["tracks"][1]["offset"], 312.457)
        self.assertEqual(data["tracks"][1]["offset_source"], "manual")

    def test_offsets_reach_subtitle_times_through_the_merge(self):
        bob = Track(path=Path("bob.m4a"), speaker="Bob", offset=150.25,
                    segments=[Segment("Bob", 1.1, 2.4, "Hi.")])
        srt = exporters.to_srt(merge_tracks([bob]))
        vtt = exporters.to_vtt(merge_tracks([bob]))
        self.assertIn("00:02:31,350 --> 00:02:32,650", srt)
        self.assertIn("00:02:31.350 --> 00:02:32.650", vtt)

    def test_every_exporter_takes_the_same_arguments(self):
        """EXPORTERS[ext](segments, tracks, model) must work for every key."""
        for ext, render in exporters.EXPORTERS.items():
            with self.subTest(ext=ext):
                self.assertIsInstance(render(SEGMENTS, TRACKS, "m"), str)


class TargetPaths(unittest.TestCase):
    def test_a_dotted_name_keeps_all_of_its_name(self):
        paths = exporters.target_paths(Path("Session 2.5"))
        self.assertIn(Path("Session 2.5.md"), paths)
        self.assertNotIn(Path("Session 2.md"), paths)

    def test_a_known_extension_is_replaced_not_appended(self):
        paths = exporters.target_paths(Path("session.md"))
        self.assertIn(Path("session.json"), paths)
        self.assertNotIn(Path("session.md.json"), paths)

    def test_covers_every_format(self):
        paths = exporters.target_paths(Path("x.md"))
        self.assertEqual({p.suffix for p in paths}, set(exporters.EXPORTERS))


class ExportAll(unittest.TestCase):
    def test_writes_one_file_per_format(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder) / "session.md"
            written = exporters.export_all(base, SEGMENTS, TRACKS, "m")
            self.assertEqual(len(written), len(exporters.EXPORTERS))
            for path in written:
                self.assertTrue(path.is_file(), path)

    def test_renders_everything_before_writing_anything(self):
        """A failure partway through must not leave a half-updated set."""
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder) / "session.md"
            broken = [Segment("A", 0.0, 1.0, None)]  # type: ignore[arg-type]
            with self.assertRaises(Exception):
                exporters.export_all(base, broken, TRACKS, "m")
            self.assertEqual(list(Path(folder).iterdir()), [])

    def test_export_rejects_an_unknown_format(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(ValueError):
                exporters.export(Path(folder) / "x.docx", SEGMENTS, TRACKS, "m")


if __name__ == "__main__":
    unittest.main()
