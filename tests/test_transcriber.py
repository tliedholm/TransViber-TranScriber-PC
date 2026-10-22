"""ASR-side logic that runs without a model: splitting, routing, script checks,
and what the worker hands a model."""

from __future__ import annotations

import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import tests  # noqa: F401  (puts app/ on sys.path)

import transcriber
from models import Track
from tests.test_audio import noise, write_mkv
from transcriber import SPLIT_GAP_SECONDS, TranscribeWorker, engine_for, split_on_word_gaps


@dataclass
class Word:
    start: float
    end: float
    word: str


@dataclass
class FWSegment:
    start: float
    end: float
    text: str
    words: list


class SplitOnWordGaps(unittest.TestCase):
    """The fix that keeps multi-track interleaving honest.

    Whisper can return one segment spanning a long silence. Left whole, the
    speaker's later remark sorts at the time of their earlier one and the
    merged conversation is wrong — so segments are cut at word-level gaps.
    """

    def test_splits_at_a_silence_longer_than_the_gap(self):
        words = [Word(0.0, 0.5, "Hello"),
                 Word(0.5, 1.0, " there"),
                 Word(1.0 + SPLIT_GAP_SECONDS + 0.5, 12.0, " again")]
        pieces = split_on_word_gaps(FWSegment(0.0, 12.0, "x", words), "Alice")
        self.assertEqual(len(pieces), 2)
        self.assertEqual(pieces[0].text, "Hello there")
        self.assertEqual(pieces[0].end, 1.0)
        self.assertEqual(pieces[1].text, " again")
        self.assertEqual(pieces[1].start, 1.0 + SPLIT_GAP_SECONDS + 0.5)

    def test_leaves_continuous_speech_alone(self):
        words = [Word(0.0, 0.5, "Hello"), Word(0.6, 1.0, " there")]
        pieces = split_on_word_gaps(FWSegment(0.0, 1.0, "x", words), "Alice")
        self.assertEqual(len(pieces), 1)

    def test_a_gap_exactly_at_the_threshold_does_not_split(self):
        words = [Word(0.0, 1.0, "a"), Word(1.0 + SPLIT_GAP_SECONDS, 3.0, " b")]
        pieces = split_on_word_gaps(FWSegment(0.0, 3.0, "x", words), "Alice")
        self.assertEqual(len(pieces), 1)

    def test_falls_back_to_the_whole_segment_without_word_timings(self):
        pieces = split_on_word_gaps(FWSegment(0.0, 5.0, "whole", []), "Alice")
        self.assertEqual(len(pieces), 1)
        self.assertEqual(pieces[0].text, "whole")

    def test_every_piece_carries_the_speaker(self):
        words = [Word(0.0, 0.5, "a"), Word(9.0, 9.5, " b")]
        pieces = split_on_word_gaps(FWSegment(0.0, 9.5, "x", words), "Bob")
        self.assertTrue(all(p.speaker == "Bob" for p in pieces))


class EngineRouting(unittest.TestCase):
    def test_whisper_is_the_default(self):
        self.assertEqual(engine_for("large-v3"), "whisper")
        self.assertEqual(engine_for("distil-large-v3.5-ct2"), "whisper")
        self.assertEqual(engine_for("openai/whisper-large-v3"), "whisper")

    def test_qwen_names_route_to_the_qwen_wrapper(self):
        self.assertEqual(engine_for("qwen3-asr-1.7b"), "qwen")
        self.assertEqual(engine_for("Qwen/Qwen3-ASR-1.7B"), "qwen")

    def test_nvidia_shaped_names_route_to_onnx(self):
        for name in ("parakeet-tdt-0.6b-v2", "canary-1b-v2",
                     "nemo-parakeet-tdt-0.6b-v3", "parakeet-typo"):
            with self.subTest(name=name):
                self.assertEqual(engine_for(name), "onnx")

    def test_routing_ignores_case(self):
        self.assertEqual(engine_for("PARAKEET-TDT-0.6B-V2"), "onnx")


class ForeignScript(unittest.TestCase):
    """Suppresses Qwen's hallucinated filler without eating real words."""

    def test_accepts_text_in_the_locked_language(self):
        self.assertFalse(transcriber._foreign_script("Hello there", "en"))

    def test_rejects_text_in_a_script_the_language_cannot_use(self):
        self.assertTrue(transcriber._foreign_script("字幕由社群提供", "en"))

    def test_punctuation_and_digits_do_not_decide(self):
        self.assertFalse(transcriber._foreign_script("123 — ...!", "en"))


class _Whisper:
    """Stands in for a loaded faster-whisper model: keeps what it was given."""

    def transcribe(self, source, **_options):
        self.source = source
        return [], SimpleNamespace(duration=len(source) / 16000, language="en")


class Decoding(unittest.TestCase):
    """What the worker hands a model, and what it does when the file is the problem."""

    def run_worker(self, track: Track, device: str = "cpu"):
        model = _Whisper()
        worker = TranscribeWorker([track], [0], "large-v3", device, None)
        notes, failures = [], []
        worker.track_note.connect(lambda _i, note: notes.append(note))
        worker.track_failed.connect(lambda _i, message: failures.append(message))
        with patch.object(worker, "_load_model", return_value=model), \
                patch("transcriber._get_model") as reload:
            worker.run()
        return model, worker, notes, failures, reload

    def test_whisper_reads_through_the_apps_decoder(self):
        # Given the path, faster-whisper would decode it itself and close the
        # two-second gap: "later" would be heard a second after "earlier".
        with tempfile.TemporaryDirectory() as folder:
            path = write_mkv(Path(folder, "gap.mkv"), [[(0, noise(1, 1)), (3, noise(1, 2))]])
            model, _worker, notes, failures, _reload = self.run_worker(Track(path, "Alice"))
        self.assertIsInstance(model.source, np.ndarray)
        self.assertEqual(len(model.source), 4 * 16000)
        self.assertEqual((notes, failures), ([], []))

    def test_a_timing_note_goes_out_with_the_track(self):
        with tempfile.TemporaryDirectory() as folder:
            first, second = Path(folder, "one.ts"), Path(folder, "two.ts")
            for path, seconds in ((first, 2), (second, 1)):
                write_mkv(path, [[(0, noise(seconds, 3))]], frame=1152, fmt="mpegts",
                          codec="mp2")
            joined = Path(folder, "joined.ts")
            joined.write_bytes(first.read_bytes() + second.read_bytes())
            _model, _worker, notes, failures, _reload = self.run_worker(Track(joined, "Alice"))
        self.assertEqual(failures, [])
        (note,) = notes
        self.assertIn("jump back at 0:02", note)

    def test_a_problem_with_the_file_fails_the_track_without_moving_to_the_cpu(self):
        with tempfile.TemporaryDirectory() as folder:
            path = write_mkv(Path(folder, "one.mkv"), [[(0, noise(1, 4))]])
            track = Track(path, "Alice", stream_index=3)
            _model, worker, _notes, failures, reload = self.run_worker(track, "cuda")
        (failure,) = failures
        self.assertIn("has no audio stream 4", failure)
        self.assertEqual(worker.device, "cuda")
        reload.assert_not_called()


if __name__ == "__main__":
    unittest.main()
