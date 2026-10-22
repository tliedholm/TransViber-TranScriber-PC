"""ASR-side logic that runs without a model: splitting, routing, script checks."""

from __future__ import annotations

import ast
import importlib.util
import unittest
from dataclasses import dataclass
from pathlib import Path

import tests  # noqa: F401  (puts app/ on sys.path)

import transcriber
from transcriber import SPLIT_GAP_SECONDS, engine_for, language_supported, split_on_word_gaps


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


class LanguageSupport(unittest.TestCase):
    """Qwen's Language choices are a closed list; the others take any."""

    def test_qwen_refuses_what_qwen_asr_does_not_support(self):
        for model in ("qwen3-asr-1.7b", "qwen3-asr-0.6b", "Qwen/Qwen3-ASR-0.6B"):
            for language in ("no", "uk"):
                with self.subTest(model=model, language=language):
                    self.assertFalse(language_supported(model, language))

    def test_qwen_takes_auto_detect_and_its_own_languages(self):
        for language in (None, "en", "sv", "zh"):
            with self.subTest(language=language):
                self.assertTrue(language_supported("qwen3-asr-1.7b", language))

    def test_whisper_and_nvidia_models_take_every_choice(self):
        for model in ("large-v3", "parakeet-tdt-0.6b-v3"):
            for language in ("no", "uk", "en", None):
                with self.subTest(model=model, language=language):
                    self.assertTrue(language_supported(model, language))

    def test_every_qwen_language_is_one_the_installed_qwen_asr_accepts(self):
        # Read the list out of the package's source, not by importing it:
        # importing qwen_asr loads torch and transformers.
        spec = importlib.util.find_spec("qwen_asr")
        if spec is None:
            self.skipTest("qwen-asr is not installed")
        source = Path(spec.origin).with_name("inference") / "utils.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        supported = next(
            ast.literal_eval(node.value) for node in ast.walk(tree)
            if isinstance(node, ast.AnnAssign)
            and getattr(node.target, "id", None) == "SUPPORTED_LANGUAGES")
        self.assertEqual(set(transcriber.QWEN_LANGUAGE_NAMES.values()) - set(supported), set())


class ForeignScript(unittest.TestCase):
    """Suppresses Qwen's hallucinated filler without eating real words."""

    def test_accepts_text_in_the_locked_language(self):
        self.assertFalse(transcriber._foreign_script("Hello there", "en"))

    def test_rejects_text_in_a_script_the_language_cannot_use(self):
        self.assertTrue(transcriber._foreign_script("字幕由社群提供", "en"))

    def test_punctuation_and_digits_do_not_decide(self):
        self.assertFalse(transcriber._foreign_script("123 — ...!", "en"))


if __name__ == "__main__":
    unittest.main()
