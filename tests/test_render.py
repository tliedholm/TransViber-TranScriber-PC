"""The incremental transcript render must match a full redraw exactly.

The whole point of appending instead of rebuilding is that nobody can see
the difference, so the test is equality against the naive version.
Everything here runs on Qt's offscreen platform — no window appears.
"""

from __future__ import annotations

import os
import unittest

import tests  # noqa: F401  (puts app/ on sys.path)

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtGui import QTextCursor  # noqa: E402
from PySide6.QtWidgets import QApplication, QTextBrowser  # noqa: E402

from main_window import _TranscriptPane  # noqa: E402

_app = QApplication.instance() or QApplication([])


def block(speaker: str, ts: str, text: str) -> str:
    return (f"<p><b>{speaker}</b> <span style='color:gray'>[{ts}]</span>: "
            f"{text}</p>")


def rendered_whole(blocks: list[str]) -> str:
    """What a from-scratch redraw of these blocks looks like."""
    view = QTextBrowser()
    cursor = view.textCursor()
    for index, fragment in enumerate(blocks):
        if index:
            cursor.insertBlock()
        cursor.insertHtml(fragment)
    return view.toPlainText()


class IncrementalRender(unittest.TestCase):
    def setUp(self):
        self.view = QTextBrowser()
        self.view.show()  # isVisible() gates the update
        self.pane = _TranscriptPane(self.view, stick_to_bottom=True)

    def feed(self, states: list[list[str]]) -> None:
        for blocks in states:
            self.pane.update(blocks)

    def test_a_growing_last_block_is_redrawn_correctly(self):
        states = [
            [block("Alice", "0:00", "The dragon attacked.")],
            [block("Alice", "0:00", "The dragon attacked. We ran.")],
            [block("Alice", "0:00", "The dragon attacked. We ran. Fast.")],
        ]
        self.feed(states)
        self.assertEqual(self.view.toPlainText(), rendered_whole(states[-1]))
        self.assertEqual(self.view.document().blockCount(), 1)

    def test_appending_speakers_matches_a_full_redraw(self):
        states = [
            [block("Alice", "0:00", "One.")],
            [block("Alice", "0:00", "One."), block("Bob", "0:12", "Two.")],
            [block("Alice", "0:00", "One."), block("Bob", "0:12", "Two."),
             block("Carl", "0:31", "Three.")],
        ]
        self.feed(states)
        self.assertEqual(self.view.toPlainText(), rendered_whole(states[-1]))
        self.assertEqual(self.view.document().blockCount(), 3)

    def test_an_out_of_order_segment_forces_a_correct_rebuild(self):
        """A second track can finish late and sort in ahead of drawn blocks."""
        early = [block("Alice", "0:00", "One."), block("Carl", "0:31", "Three.")]
        later = [block("Alice", "0:00", "One."), block("Dana", "0:25", "Two."),
                 block("Carl", "0:31", "Three.")]
        self.feed([early, later])
        self.assertEqual(self.view.toPlainText(), rendered_whole(later))
        self.assertEqual(self.view.document().blockCount(), 3)

    def test_formatting_survives_the_append_path(self):
        self.feed([[block("Alice", "0:00", "One.")],
                   [block("Alice", "0:00", "One."), block("Bob", "0:12", "Two.")]])
        self.assertEqual(self.view.toHtml().count("font-weight:700"), 2)

    def test_clearing_empties_the_view(self):
        self.feed([[block("Alice", "0:00", "One.")], []])
        self.assertEqual(self.view.toPlainText(), "")

    def test_an_unchanged_transcript_does_no_work(self):
        blocks = [block("Alice", "0:00", "One.")]
        self.pane.update(blocks)
        before = self.view.document().revision()
        self.pane.update(list(blocks))
        self.assertEqual(self.view.document().revision(), before)

    def test_a_hidden_pane_is_skipped_and_catches_up_later(self):
        hidden = QTextBrowser()  # never shown: isVisible() is False
        pane = _TranscriptPane(hidden, stick_to_bottom=False)
        blocks = [block("Alice", "0:00", "One."), block("Bob", "0:12", "Two.")]
        pane.update(blocks)
        self.assertEqual(hidden.toPlainText(), "")
        hidden.show()
        pane.refresh(blocks)
        self.assertEqual(hidden.toPlainText(), rendered_whole(blocks))

    def test_refreshing_an_up_to_date_pane_rebuilds_nothing(self):
        blocks = [block("Alice", "0:00", "One."), block("Bob", "0:12", "Two.")]
        self.pane.update(blocks)
        before = self.view.document().revision()
        self.pane.refresh(list(blocks))
        self.assertEqual(self.view.document().revision(), before)

    def test_a_long_transcript_stays_correct(self):
        states = []
        blocks = []
        for i in range(120):
            blocks = blocks + [block(f"S{i % 4}", f"{i}:00", f"Line {i}.")]
            states.append(list(blocks))
        self.feed(states)
        self.assertEqual(self.view.toPlainText(), rendered_whole(states[-1]))
        self.assertEqual(self.view.document().blockCount(), 120)


if __name__ == "__main__":
    unittest.main()
