"""Export formats: timings, escaping, and which files get written."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

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
        self.assertEqual(exporters.hms(75), "1:15")
        self.assertEqual(exporters.hms(3675), "1:01:15")


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


def _failing_on(call, error=None):
    """A stand-in for `real` that raises `error` on its `call`th use (from 1)
    and otherwise does what the real function does."""
    error = error or PermissionError(13, "Access is denied")

    def wrap(real):
        count = {"n": 0}

        def stand_in(*args, **kwargs):
            count["n"] += 1
            if count["n"] == call:
                raise error
            return real(*args, **kwargs)
        return stand_in
    return wrap


class ExportFailures(unittest.TestCase):
    """Each file is replaced whole or not at all, and a failure says exactly
    which files are new and which still hold their old contents."""

    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.folder = Path(temp.name)
        self.base = self.folder / "session.md"
        self.targets = exporters.target_paths(self.base)
        for path in self.targets:
            path.write_text("old " + path.suffix, encoding="utf-8")

    def contents(self):
        return {p.name: p.read_text(encoding="utf-8") for p in self.targets}

    def assert_untouched(self):
        self.assertEqual(self.contents(), {p.name: "old " + p.suffix for p in self.targets})

    def assert_no_staged_files_left(self):
        self.assertEqual(sorted(self.folder.iterdir()), sorted(self.targets))

    def test_a_full_disk_while_staging_changes_nothing(self):
        full = OSError(28, "No space left on device")
        with patch("exporters.os.fsync", _failing_on(3, full)(os.fsync)):
            with self.assertRaises(exporters.ExportError) as caught:
                exporters.export_all(self.base, SEGMENTS, TRACKS, "m")
        self.assert_untouched()
        self.assert_no_staged_files_left()
        error = caught.exception
        self.assertEqual(error.written, [])
        self.assertEqual(error.failed, self.targets[2])
        self.assertEqual(set(error.unchanged), set(self.targets))
        self.assertIn("No space left on device", str(error))
        self.assertIn("Nothing was exported", str(error))

    def test_a_failed_first_replacement_reports_nothing_written(self):
        with patch("exporters.os.replace", _failing_on(1)(os.replace)):
            with self.assertRaises(exporters.ExportError) as caught:
                exporters.export_all(self.base, SEGMENTS, TRACKS, "m")
        self.assert_untouched()
        self.assert_no_staged_files_left()
        self.assertEqual(caught.exception.written, [])
        self.assertEqual(caught.exception.unchanged, self.targets)

    def test_a_failed_second_replacement_names_what_is_new_and_what_is_old(self):
        with patch("exporters.os.replace", _failing_on(2)(os.replace)):
            with self.assertRaises(exporters.ExportError) as caught:
                exporters.export_all(self.base, SEGMENTS, TRACKS, "m")
        error = caught.exception
        md, txt = self.targets[:2]
        self.assertEqual(error.written, [md])
        self.assertEqual(error.failed, txt)
        self.assertEqual(error.unchanged, self.targets[1:])
        self.assertIsInstance(error, OSError)   # the window's `except OSError` still catches it
        self.assertIsInstance(error.__cause__, PermissionError)
        # The one that was replaced is whole; the rest are exactly as they were.
        contents = self.contents()
        self.assertEqual(contents.pop(md.name), exporters.to_markdown(SEGMENTS, TRACKS, "m"))
        self.assertEqual(contents, {p.name: "old " + p.suffix for p in self.targets[1:]})
        self.assert_no_staged_files_left()
        message = str(error)
        self.assertIn("Exported session.md, but could not write session.txt: Access is denied",
                      message)
        self.assertIn("Not updated: session.txt, session.srt, session.vtt, session.json", message)

    def test_a_cleanup_failure_never_hides_the_real_error(self):
        with patch("exporters.os.replace", _failing_on(1)(os.replace)), \
                patch("exporters.Path.unlink", side_effect=OSError("locked by antivirus")):
            with self.assertRaises(exporters.ExportError) as caught:
                exporters.export_all(self.base, SEGMENTS, TRACKS, "m")
        self.assertIn("Access is denied", str(caught.exception))
        self.assertNotIn("antivirus", str(caught.exception))
        self.assert_untouched()

    def test_a_failed_single_format_export_keeps_the_old_file(self):
        srt = self.folder / "session.srt"
        with patch("exporters.os.replace", _failing_on(1)(os.replace)):
            with self.assertRaises(exporters.ExportError) as caught:
                exporters.export(srt, SEGMENTS, TRACKS, "m")
        self.assertEqual(srt.read_text(encoding="utf-8"), "old .srt")
        self.assert_no_staged_files_left()
        self.assertEqual(str(caught.exception).splitlines()[0],
                         "Could not write session.srt: Access is denied")

    def test_a_successful_export_replaces_every_file_whole(self):
        written = exporters.export_all(self.base, SEGMENTS, TRACKS, "m")
        self.assertEqual(written, self.targets)
        self.assertEqual(self.contents()["session.vtt"], exporters.to_vtt(SEGMENTS))
        self.assert_no_staged_files_left()


if __name__ == "__main__":
    unittest.main()
