"""Startup, clearing, caching and shutdown regressions; no inference or GPU."""

import json
import os
import threading
import time
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import tests  # noqa: F401

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QCoreApplication, QEvent, QSettings, Qt, QTimer
from PySide6.QtGui import QCloseEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QDialog, QLineEdit

import align
import llm
import main_window
import transcriber
from align import AUTOMATIC_REFERENCE, SESSION_REFERENCE, AlignmentResult, Suggestion
from models import OffsetSource, Segment, Track, TrackStatus
from stats import StatsPoller

_app = QApplication.instance() or QApplication([])


class _WindowTest(unittest.TestCase):
    """A real MainWindow with settings, notes and device probing isolated."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        folder = Path(self.temp.name)
        self.notes_path = folder / "notes.md"
        self.settings = QSettings(str(folder / "settings.ini"), QSettings.Format.IniFormat)
        self._patch("main_window.QSettings", return_value=self.settings)
        self._patch("main_window.runtime.notes_path", return_value=self.notes_path)
        self._patch("main_window.StatsPoller.start")
        self._patch("main_window.DeviceProbe.start")
        self.window = main_window.MainWindow()
        self.addCleanup(self._dispose_window)
        _app.processEvents()  # consume the deferred, mocked device probe
        self.window._on_device_detected(True)

    def _dispose_window(self):
        """Close the window and delete it now. Left to the garbage collector,
        a MainWindow crashes PySide6 6.11 (an access violation partway through
        some later test), though 6.12 collects it cleanly."""
        self.window.close()
        self.window.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)

    def _patch(self, target, **kwargs):
        p = patch(target, **kwargs)
        mock = p.start()
        self.addCleanup(p.stop)
        return mock

    def add(self, *tracks):
        """Put tracks straight into the list, as add_files would."""
        for track in tracks:
            self.window.tracks.append(track)
            self.window._append_row(track)

    def type_into_offset_cell(self, row, keys, enter=True):
        """Open the cell's real editor, type `keys` (None: type nothing), and
        press Enter unless `enter` is False, which leaves the edit open."""
        w = self.window
        w.show()
        w.table.editItem(w.table.item(row, main_window.COL_OFFSET))
        editor = w.table.viewport().findChild(QLineEdit)
        self.assertIsNotNone(editor)
        if keys is not None:
            editor.selectAll()
            QTest.keyClicks(editor, keys)
        if enter:
            QTest.keyClick(editor, Qt.Key.Key_Return)
            _app.processEvents()


class WindowLifecycle(_WindowTest):
    def test_clear_catches_up_a_hidden_transcript_on_tab_switch(self):
        w = self.window
        w.show()
        w.tracks = [Track(Path("audio.wav"), "Alice", segments=[Segment("Alice", 0, 1, "Hello")])]
        w._render_transcript()
        w.tabs.setCurrentIndex(1)
        self.assertIn("Hello", w.notes_transcript_view.toPlainText())
        w.tabs.setCurrentIndex(0)
        w.clear_tracks()
        w.tabs.setCurrentIndex(1)
        self.assertEqual(w.notes_transcript_view.toPlainText(), "")
        self.assertEqual(w._transcript_blocks, [])

    def test_transcription_completion_keeps_model_for_the_next_run(self):
        with patch("main_window.free_asr_models") as release:
            self.window._on_finished(False)
            release.assert_not_called()

    def test_close_waits_for_running_worker_without_destroying_it(self):
        worker = Mock()
        worker.isRunning.return_value = True
        self.window.worker = worker
        event = QCloseEvent()
        self.window.closeEvent(event)
        self.assertFalse(event.isAccepted())
        worker.stop.assert_called_once()
        worker.wait.assert_not_called()
        self.assertTrue(self.window._close_timer.isActive())
        worker.isRunning.return_value = False
        final_event = QCloseEvent()
        self.window.closeEvent(final_event)
        self.assertTrue(final_event.isAccepted())
        self.assertFalse(self.window._close_timer.isActive())
        self.window.worker = None

    def test_loading_saved_notes_does_not_schedule_a_rewrite(self):
        self.notes_path.write_text("Saved notes", encoding="utf-8")
        self.window._load_settings()
        self.assertEqual(self.window.notes_edit.toPlainText(), "Saved notes")
        self.assertFalse(self.window._notes_save_timer.isActive())

    def test_invalid_saved_notes_are_never_overwritten_on_close(self):
        original = b"\xff\xfe\x00"
        self.notes_path.write_bytes(original)
        self.window._load_settings()
        self.window._save_settings()
        self.assertEqual(self.notes_path.read_bytes(), original)
        self.assertTrue(self.window._notes_load_failed)

    def test_notes_save_error_is_visible(self):
        with patch("main_window.runtime.write_atomic", side_effect=OSError("disk full")):
            self.window._save_notes_text()
        self.assertIn("disk full", self.window.statusBar().currentMessage())

    def test_unavailable_gpu_selects_cpu(self):
        self.window._on_device_detected(False)
        self.assertEqual(self.window.combo_device.currentData(), "cpu")
        self.assertFalse(self.window.combo_device.model().item(0).isEnabled())
        self.assertTrue(self.window.btn_transcribe.isEnabled())

    def test_status_reset_removes_an_old_error_tooltip(self):
        self.window._append_row(Track(Path("audio.wav"), "Alice"))
        item = self.window.table.item(0, main_window.COL_STATUS)
        item.setToolTip("old GPU error")
        self.window._set_status(0, "Pending")
        self.assertEqual(item.toolTip(), "")


class LanguageChoices(_WindowTest):
    """Languages the chosen model can't take: greyed out, and flagged and
    refused when one is already selected, never quietly turned into
    auto-detect."""

    def language_row(self, code):
        return self.window.combo_lang.findData(code)

    def item(self, code):
        return self.window.combo_lang.model().item(self.language_row(code))

    def test_qwen_greys_out_what_it_lacks_and_whisper_gets_it_back(self):
        w = self.window
        w.combo_model.setCurrentText("qwen3-asr-1.7b")
        self.assertFalse(self.item("no").isEnabled())
        self.assertFalse(self.item("uk").isEnabled())
        self.assertTrue(self.item("en").isEnabled())
        self.assertTrue(self.item(None).isEnabled())
        w.combo_model.setCurrentText("large-v3")
        self.assertTrue(self.item("no").isEnabled())
        self.assertTrue(self.item("uk").isEnabled())

    def test_a_typed_qwen_repo_gets_the_same_choices(self):
        self.window.combo_model.setCurrentText("Qwen/Qwen3-ASR-0.6B")
        self.assertFalse(self.item("no").isEnabled())

    def test_switching_to_qwen_keeps_the_language_and_flags_it(self):
        w = self.window
        w.combo_model.setCurrentText("large-v3")
        w.combo_lang.setCurrentIndex(self.language_row("no"))
        self.assertTrue(w.label_lang_conflict.isHidden())
        w.combo_model.setCurrentText("qwen3-asr-1.7b")
        self.assertEqual(w.combo_lang.currentData(), "no")
        self.assertFalse(w.label_lang_conflict.isHidden())
        self.assertIn("Norwegian", w.label_lang_conflict.toolTip())
        w.combo_lang.setCurrentIndex(self.language_row("en"))
        self.assertTrue(w.label_lang_conflict.isHidden())

    def test_transcribe_refuses_the_conflict_and_changes_nothing(self):
        w = self.window
        self.add(Track(Path("alice.wav"), "Alice"))
        w.combo_model.setCurrentText("qwen3-asr-1.7b")
        w.combo_lang.setCurrentIndex(self.language_row("uk"))
        status = w.table.item(0, main_window.COL_STATUS).text()
        with patch("main_window.TranscribeWorker") as worker_class, \
                patch("main_window.QMessageBox.warning") as warning, \
                patch("main_window.llm.free_llm") as free_llm:
            w.toggle_transcription()
        worker_class.assert_not_called()
        free_llm.assert_not_called()
        self.assertIn("Ukrainian", warning.call_args.args[2])
        self.assertIsNone(w.worker)
        self.assertEqual(w.table.item(0, main_window.COL_STATUS).text(), status)

    def test_a_saved_conflict_is_flagged_at_startup(self):
        self.settings.setValue("model", "qwen3-asr-0.6b")
        self.settings.setValue("language", "no")
        self.window._load_settings()
        self.assertEqual(self.window.combo_lang.currentData(), "no")
        self.assertFalse(self.window.label_lang_conflict.isHidden())


class OffsetColumn(_WindowTest):
    """A track's Offset cell: what a committed edit does (`edit`), and the
    real cell editor that commits one (`type_into_offset_cell`)."""

    def offset_cell(self, row):
        return self.window.table.item(row, main_window.COL_OFFSET)

    def edit(self, row, text):
        self.window._on_offset_edited(row, text)

    def test_typing_an_offset_reorders_the_transcript(self):
        w = self.window
        w.show()
        alice = Track(Path("alice.wav"), "Alice",
                      segments=[Segment("Alice", 180, 181, "Three minutes in.")])
        bob = Track(Path("bob.wav"), "Bob",
                    segments=[Segment("Bob", 60, 61, "Sorry I'm late.")])
        self.add(alice, bob)
        w._render_transcript()
        text = w.transcript_view.toPlainText()
        self.assertLess(text.index("Sorry"), text.index("Three"))

        self.edit(1, "5:00")
        self.assertEqual(bob.offset, 300.0)
        self.assertEqual(bob.offset_source, OffsetSource.MANUAL)
        w._render_transcript()
        text = w.transcript_view.toPlainText()
        self.assertLess(text.index("Three"), text.index("Sorry"))
        self.assertIn("[6:00]", text)
        self.assertEqual(bob.segments[0].start, 60)  # raw times stay raw

    def test_a_bad_offset_is_rejected_and_the_old_value_stays(self):
        bob = Track(Path("bob.wav"), "Bob")
        self.add(bob)
        self.edit(0, "2:30")
        self.edit(0, "1:75")
        self.assertEqual(bob.offset, 150.0)
        self.assertEqual(self.offset_cell(0).text(), "2:30")
        self.assertIn("under 60", self.window.statusBar().currentMessage())

    def test_the_cell_is_rewritten_in_standard_form(self):
        bob = Track(Path("bob.wav"), "Bob")
        self.add(bob)
        self.assertEqual(self.offset_cell(0).text(), "0:00")
        self.assertIn("Not set", self.offset_cell(0).toolTip())
        self.edit(0, "150,5")
        self.assertEqual(self.offset_cell(0).text(), "2:30.5")
        self.assertIn("Typed by you", self.offset_cell(0).toolTip())

    def test_a_typed_zero_counts_as_set_by_hand(self):
        bob = Track(Path("bob.wav"), "Bob")
        self.add(bob)
        self.edit(0, "0")
        self.assertEqual(bob.offset, 0.0)
        self.assertEqual(bob.offset_source, OffsetSource.MANUAL)

    def test_typing_in_the_real_editor_sets_the_offset(self):
        bob = Track(Path("bob.wav"), "Bob")
        self.add(bob)
        self.type_into_offset_cell(0, "2:30,5")
        self.assertEqual(bob.offset, 150.5)
        self.assertEqual(self.offset_cell(0).text(), "2:30.5")

    def test_retyping_the_shown_zero_counts_as_set_by_hand(self):
        bob = Track(Path("bob.wav"), "Bob")
        self.add(bob)
        self.type_into_offset_cell(0, "0:00")
        self.assertEqual(bob.offset_source, OffsetSource.MANUAL)
        self.assertIn("Typed by you", self.offset_cell(0).toolTip())

    def test_opening_and_closing_a_cell_without_typing_changes_nothing(self):
        bob = Track(Path("bob.wav"), "Bob")
        self.add(bob)
        self.type_into_offset_cell(0, None)
        self.assertEqual(bob.offset_source, OffsetSource.DEFAULT)

    def test_an_absurd_offset_is_rejected_and_the_table_keeps_working(self):
        bob = Track(Path("bob.wav"), "Bob")
        self.add(bob)
        self.edit(0, "9" * 400)
        self.assertEqual(bob.offset, 0.0)
        self.assertEqual(self.offset_cell(0).text(), "0:00")
        self.edit(0, "1:00")
        self.assertEqual(bob.offset, 60.0)

    def test_renaming_a_speaker_keeps_the_offset(self):
        bob = Track(Path("bob.wav"), "Bob", segments=[Segment("Bob", 1, 2, "Hi.")])
        self.add(bob)
        self.edit(0, "1:00")
        self.window.table.item(0, main_window.COL_SPEAKER).setText("Robert")
        self.assertEqual(bob.offset, 60.0)
        self.window._render_transcript()
        self.assertIn("Robert", self.window._transcript_blocks[0])
        self.assertIn("[1:01]", self.window._transcript_blocks[0])

    def test_splitting_a_track_keeps_its_offset(self):
        track = Track(Path("room.wav"), "Room", offset=300.0,
                      offset_source=OffsetSource.MANUAL)
        self.add(track)
        with patch("main_window.probe_streams",
                   return_value=[{"channels": 2, "duration": 10.0}]):
            self.window._split_track_channels(0)
        self.assertEqual(len(self.window.tracks), 2)
        for row, new in enumerate(self.window.tracks):
            self.assertEqual(new.offset, 300.0)
            self.assertEqual(new.offset_source, OffsetSource.MANUAL)
            self.assertEqual(self.offset_cell(row).text(), "5:00")


def _wait_until(condition, timeout=5.0):
    deadline = time.monotonic() + timeout
    while not condition():
        if time.monotonic() > deadline:
            raise AssertionError("timed out waiting for the window")
        QTest.qWait(10)


class Alignment(_WindowTest):
    """Align from the window: what it locks, what the review offers, what Apply writes.

    The worker is the real AlignWorker thread with its run() scripted, so the
    signals cross threads as they do in the app; the review window is real
    too, with only its exec() replaced so a test can play the user.
    """

    def setUp(self):
        super().setUp()
        self.folder = Path(self.temp.name)
        self._patch("main_window.tracks_for_file",
                    side_effect=lambda path: [Track(path, path.stem.title(), duration=600.0)])
        self._patch("main_window.probe_streams",
                    return_value=[{"channels": 1, "duration": 3600.0}])
        self.info = self._patch("main_window.QMessageBox.information")
        self.warning = self._patch("main_window.QMessageBox.warning")

    def file(self, name):
        path = self.folder / name
        path.touch()
        return path

    def stereo(self, name):
        """A file loaded as two rows, the way a split stereo track is."""
        path = self.file(name)
        self.add(*(Track(path, path.stem.title() + suffix, duration=600.0, channel=ch)
                   for ch, suffix in enumerate((" L", " R"))))
        return path

    def align(self, rows, review=None, *, mode=AUTOMATIC_REFERENCE, reference=None,
              note="", result=None, run=None):
        """Press Align with the worker scripted to return `result` (by default
        one made of `rows`), or to `run(worker)`, then play the review window
        with `review(dialog) -> accepted`. Returns what was seen."""
        seen = {"constructed": None, "dialogs": 0}
        rows = list(rows)

        class Scripted(align.AlignWorker):
            def __init__(inner, *args, **kwargs):
                super().__init__(*args, **kwargs)
                seen["constructed"] = (inner.units, inner.session_reference)
                seen["known"] = inner.known

            def run(inner):
                if run is not None:
                    run(inner)
                    return
                inner.result = result or AlignmentResult(
                    rows, mode, reference or rows[0].path, note)

        w = self.window

        def play(dialog):
            # The window deletes the dialog once it closes: keep what it showed.
            seen["dialogs"] += 1
            seen["basis"] = dialog.basis_label.text()
            seen["cells"] = [[dialog.table.item(row, col).text() for col in range(5)]
                             for row in range(dialog.table.rowCount())]
            seen["locked_in_review"] = [not b.isEnabled() for b in (
                w.btn_add, w.btn_remove, w.btn_clear, w.btn_reference,
                w.btn_remove_reference, w.btn_align)]
            seen["ticks"] = {s.path.name + f"#{i}": dialog.is_ticked(i)
                             for i, s in enumerate(dialog.suggestions)}
            return review(dialog) if review else False

        with patch("main_window.AlignWorker", Scripted), \
                patch.object(main_window.AlignReviewDialog, "exec", play):
            w.toggle_alignment()
            seen["while_running"] = {
                "align_text": w.btn_align.text(),
                "locked": [not b.isEnabled() for b in (
                    w.btn_add, w.btn_remove, w.btn_clear, w.btn_reference,
                    w.btn_remove_reference)],
                "transcribe": w.btn_transcribe.isEnabled(),
                "offset_editable": bool(w.table.item(0, main_window.COL_OFFSET).flags()
                                        & Qt.ItemFlag.ItemIsEditable) if w.tracks else None,
            }
            _wait_until(lambda: w.align_worker is None and not w._align_reviewing)
        return seen

    def test_a_reference_alone_has_nothing_to_align_or_transcribe(self):
        w = self.window
        self.assertTrue(w.set_session_reference(self.file("session.wav")))
        w.toggle_alignment()
        self.assertIsNone(w.align_worker)
        self.assertIn("Nothing to align", self.info.call_args.args[1])
        w.toggle_transcription()
        self.assertIsNone(w.worker)
        self.assertEqual(self.info.call_args.args[1], "No tracks")
        self.assertFalse(w.btn_export.isEnabled())
        self.assertEqual(w.tracks, [])

    def test_one_speaker_file_and_a_reference_can_align(self):
        w = self.window
        session, bob = self.file("session.wav"), self.file("bob.wav")
        w.set_session_reference(session)
        w.add_files([bob])
        seen = self.align([Suggestion(bob, 900.0, "matched", "Matched sound with session.wav "
                                      "(strong match)", "strong")],
                          lambda dialog: True, mode=SESSION_REFERENCE, reference=session)
        self.assertEqual(seen["constructed"], ([(bob, [None])], (session, [None])))
        self.assertIn("Reference: session.wav; session starts at 0:00", seen["basis"])
        self.assertEqual(w.tracks[0].offset, 900.0)
        self.assertEqual(w.tracks[0].offset_source, OffsetSource.MATCHED)
        self.assertEqual(w.table.item(0, main_window.COL_OFFSET).text(), "15:00")

    def test_tracks_from_one_file_have_nothing_to_align(self):
        self.stereo("room.wav")
        self.window.toggle_alignment()
        self.assertIsNone(self.window.align_worker)
        self.assertIn("already in step", self.info.call_args.args[2])

    def test_the_reference_cannot_also_be_a_speaker_file(self):
        w = self.window
        session = self.file("session.wav")
        (self.folder / "sub").mkdir()
        w.set_session_reference(session)
        w.add_files([self.folder / "sub" / ".." / "session.wav"])
        self.assertEqual(w.tracks, [])
        self.assertIn("is the session reference", self.info.call_args.args[2])

    def test_a_speaker_file_cannot_also_be_the_reference(self):
        w = self.window
        session, alice = self.file("session.wav"), self.file("alice.wav")
        w.set_session_reference(session)
        w.add_files([alice])
        (self.folder / "sub").mkdir()
        self.assertFalse(w.set_session_reference(self.folder / "sub" / ".." / "alice.wav"))
        self.assertEqual(w.session_reference.path, session)
        self.assertIn("Remove it from the track list first".lower(),
                      self.info.call_args.args[2].lower())

    def test_an_unreadable_reference_keeps_the_previous_one(self):
        w = self.window
        session = self.file("session.wav")
        w.set_session_reference(session)
        with patch("main_window.probe_streams", side_effect=OSError("not audio")):
            self.assertFalse(w.set_session_reference(self.file("notes.wav")))
        self.assertEqual(w.session_reference.path, session)
        self.assertIn("not audio", self.warning.call_args.args[2])

    def test_reference_controls_show_the_file_and_clear_removes_it(self):
        w = self.window
        w.show()
        self.assertFalse(w.btn_remove_reference.isVisible())
        w.set_session_reference(self.file("session.wav"))
        self.assertEqual(w.btn_reference.text(), "Replace…")
        self.assertTrue(w.btn_remove_reference.isVisible())
        self.assertIn("session.wav", w.reference_label.text())
        self.assertIn("1:00:00", w.reference_label.text())
        self.assertIn("Timing reference only", w.reference_label.text())
        w.clear_tracks()
        self.assertIsNone(w.session_reference)
        self.assertEqual(w.btn_reference.text(), "Add session reference…")
        self.assertFalse(w.btn_remove_reference.isVisible())

    def test_replacing_or_removing_the_reference_keeps_offsets(self):
        w = self.window
        w.add_files([self.file("bob.wav")])
        w._on_offset_edited(0, "5:00")
        w.set_session_reference(self.file("session.wav"))
        w.set_session_reference(self.file("other.wav"))
        self.assertEqual(w.session_reference.path.name, "other.wav")
        w.remove_session_reference()
        self.assertIsNone(w.session_reference)
        self.assertEqual((w.tracks[0].offset, w.tracks[0].offset_source),
                         (300.0, OffsetSource.MANUAL))

    def test_the_reference_never_reaches_transcription(self):
        w = self.window
        w.set_session_reference(self.file("session.wav"))
        w.add_files([self.file("bob.wav")])
        with patch("main_window.TranscribeWorker") as worker_class, \
                patch("main_window.llm.free_llm"):
            w.toggle_transcription()
        tracks, indices = worker_class.call_args.args
        self.assertEqual([t.path.name for t in tracks], ["bob.wav"])
        self.assertEqual(indices, [0])
        w.worker = None
        w._set_running(False)

    def test_controls_stay_locked_through_alignment_and_review(self):
        w = self.window
        alice, bob = self.file("alice.wav"), self.file("bob.wav")
        w.set_session_reference(self.file("session.wav"))
        w.add_files([alice, bob])
        seen = self.align([Suggestion(alice, 0.0, "matched", "m", "strong"),
                           Suggestion(bob, 4.0, "matched", "m", "strong")],
                          lambda dialog: False, mode=SESSION_REFERENCE)
        running = seen["while_running"]
        self.assertEqual(running["align_text"], "Cancel align")
        self.assertEqual(running["locked"], [True] * 5)
        self.assertTrue(running["transcribe"])        # no GPU or model involved
        self.assertTrue(running["offset_editable"])   # typing an offset stays allowed
        self.assertEqual(seen["locked_in_review"], [True] * 6)
        for button in (w.btn_add, w.btn_remove, w.btn_clear, w.btn_reference,
                       w.btn_remove_reference, w.btn_align):
            self.assertTrue(button.isEnabled())
        self.assertEqual(w.btn_align.text(), "Align…")

    def test_shared_controls_wait_for_both_jobs(self):
        w = self.window
        w.add_files([self.file("alice.wav"), self.file("bob.wav")])
        w.worker = Mock()
        w._set_running(True)
        w.align_worker = Mock()
        w._update_busy()
        w.worker = None
        w._set_running(False)                 # transcription done, Align still running
        self.assertFalse(w.btn_add.isEnabled())
        w.add_files([self.file("carol.wav")])
        self.assertEqual(len(w.tracks), 2)
        w.align_worker = None
        w._update_busy()
        self.assertTrue(w.btn_add.isEnabled())

    def test_strong_matches_start_ticked_and_apply_writes_every_row(self):
        w = self.window
        alice = self.file("alice.wav")
        w.add_files([alice])
        room = self.stereo("room.wav")
        carol = self.file("carol.wav")
        w.add_files([carol])
        dave = self.file("dave.wav")
        w.add_files([dave])
        rows = [Suggestion(alice, 0.0, "reference", "Reference (longest recording)"),
                Suggestion(room, 12.5, "matched", "strong", "strong"),
                Suggestion(carol, 3.0, "matched", "weak", "weak"),
                Suggestion(dave, 40.0, "guess", "Guess")]
        seen = self.align(rows, lambda dialog: True)
        self.assertEqual(list(seen["ticks"].values()), [True, True, False, False])
        offsets = [(t.path.name, t.offset, t.offset_source) for t in w.tracks]
        self.assertEqual(offsets, [
            ("alice.wav", 0.0, OffsetSource.MATCHED),   # the anchor: room hangs from it
            ("room.wav", 12.5, OffsetSource.MATCHED),
            ("room.wav", 12.5, OffsetSource.MATCHED),
            ("carol.wav", 0.0, OffsetSource.DEFAULT),
            ("dave.wav", 0.0, OffsetSource.DEFAULT),
        ])
        self.assertEqual(w.table.item(2, main_window.COL_OFFSET).text(), "0:12.5")
        self.assertIn("Applied 1 offset", w.statusBar().currentMessage())

    def test_a_ticked_guess_is_recorded_as_a_guess(self):
        w = self.window
        alice, bob = self.file("alice.wav"), self.file("bob.wav")
        w.add_files([alice, bob])

        def tick_all(dialog):
            for row in range(len(dialog.suggestions)):
                dialog.set_ticked(row, True)
            return True
        self.align([Suggestion(alice, 0.0, "reference", "r"),
                    Suggestion(bob, 40.0, "guess", "Guess")], tick_all)
        self.assertEqual((w.tracks[1].offset, w.tracks[1].offset_source),
                         (40.0, OffsetSource.GUESS))

    def test_a_manual_value_on_any_row_protects_the_whole_file(self):
        w = self.window
        alice = self.file("alice.wav")
        w.add_files([alice])
        room = self.stereo("room.wav")
        w._on_offset_edited(2, "0")            # a typed zero on the right channel
        rows = [Suggestion(alice, 0.0, "reference", "r"),
                Suggestion(room, 12.5, "matched", "Matched sound (strong match)", "strong")]
        seen = self.align(rows, lambda dialog: True)
        self.assertEqual(list(seen["ticks"].values()), [True, False])
        self.assertIn("You set the current value", seen["cells"][1][4])
        self.assertEqual([t.offset for t in w.tracks[1:]], [0.0, 0.0])

        def accept_room(dialog):
            dialog.set_ticked(1, True)
            return True
        self.align(rows, accept_room)
        self.assertEqual([(t.offset, t.offset_source) for t in w.tracks[1:]],
                         [(12.5, OffsetSource.MATCHED)] * 2)

    def test_an_offset_typed_during_alignment_shows_as_current_and_manual(self):
        w = self.window
        alice, bob = self.file("alice.wav"), self.file("bob.wav")
        w.add_files([alice, bob])
        result = AlignmentResult([Suggestion(alice, 0.0, "reference", "r"),
                                  Suggestion(bob, 4.0, "matched", "m", "strong")],
                                 AUTOMATIC_REFERENCE, alice)
        seen = self.align([], lambda dialog: True,
                          run=self.run_while(result, lambda: w._on_offset_edited(1, "1:00")))
        self.assertEqual(seen["cells"][1][2], "1:00")
        self.assertFalse(seen["ticks"]["bob.wav#1"])
        self.assertEqual((w.tracks[1].offset, w.tracks[1].offset_source),
                         (60.0, OffsetSource.MANUAL))

    def test_cancel_in_the_review_window_changes_nothing(self):
        w = self.window
        alice, bob = self.file("alice.wav"), self.file("bob.wav")
        w.add_files([alice, bob])
        self.align([Suggestion(alice, 0.0, "reference", "r"),
                    Suggestion(bob, 4.0, "matched", "m", "strong")], lambda dialog: False)
        self.assertEqual([t.offset for t in w.tracks], [0.0, 0.0])
        self.assertIn("No offsets were changed", w.statusBar().currentMessage())

    def test_cancelling_alignment_changes_nothing(self):
        w = self.window
        w.add_files([self.file("alice.wav"), self.file("bob.wav")])

        def until_stopped(worker):
            while not worker._stop:
                time.sleep(0.01)
            worker.cancelled = True
        QTimer.singleShot(0, w.toggle_alignment)   # press Cancel align
        seen = self.align([], run=until_stopped)
        self.assertEqual(seen["dialogs"], 0)
        self.assertEqual([t.offset for t in w.tracks], [0.0, 0.0])
        self.assertIn("cancelled", w.statusBar().currentMessage())
        self.assertEqual(w.btn_align.text(), "Align…")

    def test_a_failed_run_explains_and_changes_nothing(self):
        w = self.window
        w.set_session_reference(self.file("session.wav"))
        w.add_files([self.file("alice.wav")])
        message = "Couldn't use the session reference session.wav: no audio."

        def fail(worker):
            worker.error = message
        seen = self.align([], run=fail)
        self.assertEqual(seen["dialogs"], 0)
        self.assertEqual(self.warning.call_args.args[2], message)
        self.assertEqual(w.tracks[0].offset, 0.0)
        self.assertTrue(w.btn_reference.isEnabled())

    def test_rows_that_cannot_apply_have_no_checkbox(self):
        w = self.window
        session, alice, bob = (self.file(n) for n in ("session.wav", "alice.wav", "bob.wav"))
        w.set_session_reference(session)
        w.add_files([alice, bob])
        rows = [Suggestion(alice, -5.0, "matched", "Starts before the session reference by "
                           "0:05 (strong match with session.wav)", "strong"),
                Suggestion(bob, None, "error", "Couldn't read it: bad header")]

        def check(dialog):
            self.assertFalse(dialog.is_checkable(0))
            self.assertFalse(dialog.is_checkable(1))
            self.assertEqual(dialog.table.item(0, dialog.COL_SUGGESTED).text(), "-0:05")
            self.assertEqual(dialog.table.item(1, dialog.COL_SUGGESTED).text(), "—")
            for row in range(2):
                dialog.set_ticked(row, True)
            return True
        self.align(rows, check, mode=SESSION_REFERENCE, reference=session)
        self.assertEqual([t.offset for t in w.tracks], [0.0, 0.0])

    def test_an_offset_still_being_typed_counts_when_the_review_opens(self):
        w = self.window
        w.show()
        alice, bob = self.file("alice.wav"), self.file("bob.wav")
        w.add_files([alice, bob])
        self.type_into_offset_cell(1, "1:00", enter=False)   # Enter not pressed yet
        seen = self.align([Suggestion(alice, 0.0, "reference", "r"),
                           Suggestion(bob, 4.0, "matched", "m", "strong")], lambda dialog: True)
        self.assertEqual(seen["cells"][1][2], "1:00")
        self.assertFalse(seen["ticks"]["bob.wav#1"])
        self.assertEqual((w.tracks[1].offset, w.tracks[1].offset_source),
                         (60.0, OffsetSource.MANUAL))
        self.assertIsNone(w._offset_delegate.open_editor)

    def automatic(self, measured, anchor=0.0):
        """The result an automatic-mode run hands over, anchored at `anchor`."""
        return align.rebase(AlignmentResult([], AUTOMATIC_REFERENCE, measured[0].path,
                                            measured=measured), anchor)

    def test_what_align_read_is_kept_for_the_files_still_loaded(self):
        w = self.window
        alice, bob, carol = self.file("alice.wav"), self.file("bob.wav"), self.file("carol.wav")
        w.add_files([alice, bob])
        kept_alice = ((1, 10), "alice's fingerprint")
        read = {(alice, (None,)): kept_alice, (bob, (None,)): ((2, 20), "bob's")}

        def cancelled(worker):
            worker.fingerprints = dict(read)
            worker.cancelled = True
        seen = self.align([], run=cancelled)
        self.assertEqual(seen["known"], {})
        self.assertEqual(w._align_fingerprints, read)     # a cancelled run's too

        w.table.selectRow(1)
        w.remove_selected()
        w.add_files([carol])
        result = AlignmentResult([Suggestion(alice, 0.0, "reference", "r"),
                                  Suggestion(carol, 4.0, "matched", "m", "strong")],
                                 AUTOMATIC_REFERENCE, alice)
        seen = self.align([], result=result)
        self.assertEqual(seen["known"], {(alice, (None,)): kept_alice})
        self.assertEqual(w._align_fingerprints, {(alice, (None,)): kept_alice})

    def run_while(self, result, during):
        """A worker run that lets `during` happen on the main thread, then finishes."""
        release = threading.Event()

        def slow(worker):
            release.wait(5)
            worker.result = result

        def act():
            during()
            release.set()
        QTimer.singleShot(0, act)
        return slow

    def test_suggestions_follow_the_reference_offset_typed_during_alignment(self):
        w = self.window
        alice, bob = self.file("alice.wav"), self.file("bob.wav")
        w.add_files([alice, bob])
        result = self.automatic([Suggestion(alice, 0.0, "reference", "Reference"),
                                 Suggestion(bob, 4.0, "matched", "m", "strong")])
        seen = self.align([], lambda dialog: True,
                          run=self.run_while(result, lambda: w._on_offset_edited(0, "1:00")))
        self.assertEqual([row[3] for row in seen["cells"]], ["1:00", "1:04"])
        self.assertEqual([t.offset for t in w.tracks], [60.0, 64.0])
        self.assertEqual(w.tracks[0].offset_source, OffsetSource.MANUAL)

    def test_a_shift_past_a_typed_offset_leaves_everything_unticked(self):
        w = self.window
        alice, bob = self.file("alice.wav"), self.file("bob.wav")
        w.add_files([alice, bob])
        w._on_offset_edited(0, "0:20")
        # bob started 10 s before alice; then alice is retyped to 0:05, which
        # would put bob before 0:00, so every suggestion moves 5 s later.
        result = self.automatic([Suggestion(alice, 0.0, "reference", "Reference"),
                                 Suggestion(bob, -10.0, "matched", "m", "strong")],
                                anchor=20.0)
        notes = []
        seen = self.align([], lambda dialog: notes.append(dialog.note_label.text()) or True,
                          run=self.run_while(result, lambda: w._on_offset_edited(0, "0:05")))
        self.assertEqual([row[3] for row in seen["cells"]], ["0:10", "0:00"])
        self.assertEqual(list(seen["ticks"].values()), [False, False])
        self.assertIn("alice.wav's offset changed while Align ran", notes[0])
        self.assertIn("moved 0:05 later", notes[0])
        self.assertIn("Nothing is ticked yet: alice.wav already has an offset", notes[0])
        self.assertEqual([(t.offset, t.offset_source) for t in w.tracks],
                         [(5.0, OffsetSource.MANUAL), (0.0, OffsetSource.DEFAULT)])

    def test_a_shift_past_an_earlier_guess_leaves_everything_unticked(self):
        w = self.window
        alice, bob, carol = (self.file(n) for n in ("alice.wav", "bob.wav", "carol.wav"))
        w.add_files([alice, bob, carol])
        w.tracks[2].offset, w.tracks[2].offset_source = 40.0, OffsetSource.GUESS
        result = self.automatic([Suggestion(alice, 0.0, "reference", "Reference"),
                                 Suggestion(bob, -30.0, "matched", "m", "strong"),
                                 Suggestion(carol, 70.0, "guess", "Guess")])
        notes = []
        seen = self.align([], lambda dialog: notes.append(dialog.note_label.text()) or True,
                          result=result)
        self.assertEqual(list(seen["ticks"].values()), [False, False, False])
        self.assertIn("carol.wav already has an offset", notes[0])
        self.assertEqual([t.offset for t in w.tracks], [0.0, 0.0, 40.0])

    def test_a_new_reference_keeps_an_earlier_run_together(self):
        w = self.window
        alice, bob = self.file("alice.wav"), self.file("bob.wav")
        w.add_files([alice, bob])
        self.align([Suggestion(alice, 0.0, "reference", "Reference"),
                    Suggestion(bob, 10.0, "matched", "m", "strong")], lambda dialog: True)
        self.assertEqual([(t.offset, t.offset_source) for t in w.tracks],
                         [(0.0, OffsetSource.MATCHED), (10.0, OffsetSource.MATCHED)])
        # A longer recording comes in and becomes the reference; it places bob
        # clearly but alice only weakly. Ticking bob alone would put the two
        # 30 s apart where the first run measured 10.
        room = self.file("room.wav")
        w.add_files([room])
        notes = []
        seen = self.align([Suggestion(room, 0.0, "reference", "Reference"),
                           Suggestion(alice, 20.0, "matched", "m", "weak"),
                           Suggestion(bob, 30.0, "matched", "m", "strong")],
                          lambda dialog: notes.append(dialog.note_label.text()) or True)
        self.assertEqual(list(seen["ticks"].values()), [False, False, False])
        self.assertIn("alice.wav already has an offset", notes[0])
        self.assertEqual([t.offset for t in w.tracks], [0.0, 10.0, 0.0])

    def test_a_placed_file_that_starts_ticked_is_not_left_behind(self):
        w = self.window
        alice, bob, carol = (self.file(n) for n in ("alice.wav", "bob.wav", "carol.wav"))
        w.add_files([alice, bob, carol])
        w.tracks[1].offset, w.tracks[1].offset_source = 4.0, OffsetSource.MATCHED
        result = self.automatic([Suggestion(alice, 0.0, "reference", "Reference"),
                                 Suggestion(bob, -10.0, "matched", "m", "strong"),
                                 Suggestion(carol, 3.0, "matched", "m", "weak")])
        notes = []
        seen = self.align([], lambda dialog: notes.append(dialog.note_label.text()) or True,
                          result=result)
        self.assertEqual(list(seen["ticks"].values()), [True, True, False])
        self.assertIn("Rows left unticked keep their current offsets", notes[0])
        self.assertEqual([t.offset for t in w.tracks], [10.0, 0.0, 0.0])

    def test_a_shift_with_nothing_typed_keeps_the_usual_ticks(self):
        w = self.window
        alice, bob, carol = (self.file(n) for n in ("alice.wav", "bob.wav", "carol.wav"))
        w.add_files([alice, bob, carol])
        result = self.automatic([Suggestion(alice, 0.0, "reference", "Reference"),
                                 Suggestion(bob, -10.0, "matched", "m", "strong"),
                                 Suggestion(carol, 3.0, "matched", "m", "weak")])
        notes = []
        seen = self.align([], lambda dialog: notes.append(dialog.note_label.text()) or True,
                          result=result)
        self.assertEqual(list(seen["ticks"].values()), [True, True, False])
        self.assertIn("Rows left unticked keep their current offsets", notes[0])
        self.assertEqual([t.offset for t in w.tracks], [10.0, 0.0, 0.0])

    def test_the_review_waits_for_another_open_prompt(self):
        w = self.window
        w.add_files([self.file("alice.wav"), self.file("bob.wav")])
        prompt = QDialog(w)
        prompt.setModal(True)
        prompt.show()
        QTimer.singleShot(400, prompt.close)
        opened_over = []
        self.align([Suggestion(w.tracks[0].path, 0.0, "reference", "r")],
                   lambda dialog: opened_over.append(prompt.isVisible()) or False)
        self.assertEqual(opened_over, [False])

    def test_closing_while_the_review_waits_ends_it(self):
        w = self.window
        w.add_files([self.file("alice.wav"), self.file("bob.wav")])
        prompt = QDialog(w)
        prompt.setModal(True)
        prompt.show()
        QTimer.singleShot(100, lambda: (setattr(w, "_closing", True), prompt.close()))
        seen = self.align([Suggestion(w.tracks[0].path, 0.0, "reference", "r")])
        self.assertEqual(seen["dialogs"], 0)
        self.assertFalse(w._align_reviewing)

    def test_transcription_started_during_alignment_keeps_the_list_locked(self):
        w = self.window
        w.set_session_reference(self.file("session.wav"))
        alice, bob = self.file("alice.wav"), self.file("bob.wav")
        w.add_files([alice, bob])
        worker_class = self._patch("main_window.TranscribeWorker")
        self._patch("main_window.llm.free_llm")
        result = AlignmentResult([Suggestion(alice, 0.0, "matched", "m", "strong"),
                                  Suggestion(bob, 4.0, "matched", "m", "strong")],
                                 SESSION_REFERENCE, w.session_reference.path)
        self.align([], lambda dialog: True, mode=SESSION_REFERENCE,
                   run=self.run_while(result, w.toggle_transcription))
        tracks, indices = worker_class.call_args.args
        self.assertEqual([t.path.name for t in tracks], ["alice.wav", "bob.wav"])
        self.assertEqual(indices, [0, 1])
        self.assertEqual(w.tracks[1].offset, 4.0)
        for button in (w.btn_add, w.btn_remove, w.btn_clear):
            self.assertFalse(button.isEnabled())   # transcription is still running
        self.assertTrue(w.btn_reference.isEnabled())
        w.worker = None
        w._set_running(False)
        self.assertTrue(w.btn_add.isEnabled())

    def transcribed(self):
        """A session reference plus one transcribed speaker, Bob."""
        w = self.window
        w.set_session_reference(self.file("session.wav"))
        w.add_files([self.file("bob.wav")])
        bob = w.tracks[0]
        bob.segments.append(Segment("Bob", 1.0, 2.0, "Hello there."))
        bob.status = main_window.TrackStatus.DONE
        return bob

    def test_the_reference_never_reaches_ai_notes(self):
        w = self.window
        self.transcribed()
        with patch("main_window.llm.NotesWorker") as notes_class:
            w.toggle_ai_notes()
        self.assertEqual(notes_class.call_args.args[0].strip(), "[0:01] Bob: Hello there.")
        w.notes_worker = None

    def test_the_reference_never_reaches_exports_and_offsets_are_read_at_write(self):
        w = self.window
        bob = self.transcribed()
        target = self.folder / "out.json"

        def save_dialog(*_args):
            bob.offset = 5.0     # an offset that changed while the dialog was open
            return str(target), "JSON (*.json)"
        with patch("main_window.QFileDialog.getSaveFileName", side_effect=save_dialog):
            w.export_dialog()
        data = json.loads(target.read_text(encoding="utf-8"))
        self.assertEqual([t["file"] for t in data["tracks"]], [str(bob.path)])
        self.assertEqual(data["segments"][0]["start"], 6.0)

    def test_close_waits_for_a_running_alignment(self):
        worker = Mock()
        worker.isRunning.return_value = True
        self.window.align_worker = worker
        event = QCloseEvent()
        self.window.closeEvent(event)
        self.assertFalse(event.isAccepted())
        worker.stop.assert_called_once()
        worker.isRunning.return_value = False
        final_event = QCloseEvent()
        self.window.closeEvent(final_event)
        self.assertTrue(final_event.isAccepted())
        self.window.align_worker = None


class IncompleteExport(_WindowTest):
    """Export warns about every track that isn't fully transcribed, including
    the ones that produced nothing at all."""

    def setUp(self):
        super().setUp()
        self.folder = Path(self.temp.name)
        self.warning = self._patch("main_window.QMessageBox.warning",
                                   return_value=main_window.QMessageBox.StandardButton.Cancel)
        self.save_dialog = self._patch("main_window.QFileDialog.getSaveFileName",
                                       return_value=("", ""))
        self.add(Track(Path("alice.wav"), "Alice", status=TrackStatus.DONE,
                       segments=[Segment("Alice", 1.0, 2.0, "Hello there.")]))

    def warned_about(self):
        """The warning's text, or None if export went straight to the dialog."""
        if not self.warning.called:
            return None
        return self.warning.call_args.args[2]

    def test_a_track_with_no_transcript_is_warned_about(self):
        cases = [
            (TrackStatus.ERROR, "failed — nothing transcribed"),
            (TrackStatus.CANCELLED, "cancelled — nothing transcribed"),
            (TrackStatus.PENDING, "not transcribed"),
        ]
        for status, described in cases:
            with self.subTest(status=status):
                self.warning.reset_mock()
                self.save_dialog.reset_mock()
                del self.window.tracks[1:]
                self.window.tracks.append(Track(Path("bob.wav"), "Bob", status=status))
                self.window.export_dialog()
                self.assertIn(f"bob.wav — {described}", self.warned_about())
                self.assertNotIn("alice.wav", self.warned_about())
                self.save_dialog.assert_not_called()   # Cancel is the default

    def test_a_partial_track_is_still_warned_about(self):
        self.add(Track(Path("bob.wav"), "Bob", status=TrackStatus.ERROR,
                       segments=[Segment("Bob", 3.0, 4.0, "And then—")]))
        self.window.export_dialog()
        self.assertIn("bob.wav — failed partway — partial transcript", self.warned_about())

    def test_a_finished_silent_track_is_not_a_gap(self):
        self.add(Track(Path("bob.wav"), "Bob", status=TrackStatus.DONE))
        self.window.export_dialog()
        self.assertIsNone(self.warned_about())
        self.save_dialog.assert_called_once()

    def test_a_long_list_is_counted_in_full(self):
        self.add(*(Track(Path(f"guest{n}.wav"), f"Guest {n}") for n in range(7)))
        self.window.export_dialog()
        text = self.warned_about()
        self.assertIn("7 track(s) not fully transcribed", text)
        self.assertIn("guest4.wav", text)
        self.assertNotIn("guest5.wav", text)
        self.assertIn("…and 2 more", text)

    def test_continuing_exports_what_there_is_and_every_tracks_status(self):
        self.warning.return_value = main_window.QMessageBox.StandardButton.Yes
        self.add(Track(Path("bob.wav"), "Bob", status=TrackStatus.ERROR,
                       error="CUDA out of memory"))
        target = self.folder / "out.json"
        self.save_dialog.return_value = (str(target), "JSON (*.json)")
        self.window.export_dialog()
        data = json.loads(target.read_text(encoding="utf-8"))
        self.assertEqual([(t["speaker"], t["status"], t["error"]) for t in data["tracks"]],
                         [("Alice", "Done", ""), ("Bob", "Error", "CUDA out of memory")])
        self.assertEqual([s["text"] for s in data["segments"]], ["Hello there."])


class ExportFailures(_WindowTest):
    """A write that fails partway is reported as far as it got."""

    def setUp(self):
        super().setUp()
        self.folder = Path(self.temp.name)
        self.critical = self._patch("main_window.QMessageBox.critical")
        self.add(Track(Path("alice.wav"), "Alice", status=TrackStatus.DONE,
                       segments=[Segment("Alice", 1.0, 2.0, "Hello there.")]))

    def replace_failing_on(self, call):
        real = os.replace
        calls = []

        def replace(source, target):
            calls.append(target)
            if len(calls) == call:
                raise PermissionError(13, "Access is denied")
            return real(source, target)
        return patch("exporters.os.replace", side_effect=replace)

    def test_a_partial_all_formats_export_names_what_was_written(self):
        all_formats = main_window.EXPORT_FILTERS.split(";;")[-1]
        self._patch("main_window.QFileDialog.getSaveFileName",
                    return_value=(str(self.folder / "out.md"), all_formats))
        with self.replace_failing_on(2):
            self.window.export_dialog()
        title, message = self.critical.call_args.args[1:3]
        self.assertEqual(title, "Export incomplete")
        self.assertIn("Exported out.md, but could not write out.txt", message)
        self.assertEqual(self.window.statusBar().currentMessage(), "Partly exported: out.md")

    def test_a_failed_single_export_says_nothing_was_written(self):
        self._patch("main_window.QFileDialog.getSaveFileName",
                    return_value=(str(self.folder / "out.srt"), "SubRip subtitles (*.srt)"))
        with self.replace_failing_on(1):
            self.window.export_dialog()
        title, message = self.critical.call_args.args[1:3]
        self.assertEqual(title, "Export failed")
        self.assertIn("Nothing was exported", message)
        self.assertEqual(list(self.folder.glob("out*")), [])

    def test_a_failed_notes_export_keeps_the_earlier_file(self):
        target = self.folder / "notes-export.md"
        target.write_text("Last week's notes", encoding="utf-8")
        self.window.notes_edit.setPlainText("This week's notes")
        self._patch("main_window.QFileDialog.getSaveFileName",
                    return_value=(str(target), "Markdown (*.md)"))
        with self.replace_failing_on(1):
            self.window.export_notes_dialog()
        self.assertEqual(self.critical.call_args.args[1], "Export failed")
        self.assertEqual(target.read_text(encoding="utf-8"), "Last week's notes")


class WorkerLifecycle(unittest.TestCase):
    @unittest.skipUnless(sys.platform == "win32", "Windows windowed launcher")
    def test_pythonw_source_launch_records_startup_errors_and_timings(self):
        with tempfile.TemporaryDirectory() as folder:
            code = (
                "import sys; from pathlib import Path; sys.path.insert(0, 'app'); "
                "import main, runtime; "
                f"runtime.writable_dir = lambda: Path({folder!r}); "
                "main._setup_frozen_logging(); print('startup logging works', flush=True)"
            )
            result = subprocess.run([str(Path(sys.executable).with_name("pythonw.exe")), "-c", code],
                                    cwd=Path(__file__).resolve().parents[1], timeout=30)
            self.assertEqual(result.returncode, 0)
            log = (Path(folder) / "app.log").read_text(encoding="utf-8")
            self.assertIn("TransViber TransScriber started", log)
            self.assertIn("startup logging works", log)

    def test_interface_import_does_not_load_inference_libraries(self):
        code = (
            "import sys; sys.path.insert(0, 'app'); import main_window; "
            "heavy = {'torch', 'ctranslate2', 'numpy', 'faster_whisper', 'onnxruntime', 'av'}; "
            "assert not heavy.intersection(sys.modules), heavy.intersection(sys.modules)"
        )
        result = subprocess.run([sys.executable, "-c", code],
                                cwd=Path(__file__).resolve().parents[1],
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_stopping_stats_wakes_its_poll_wait_immediately(self):
        poller = StatsPoller()
        poller.stats.connect(lambda _text: None)
        with patch("stats.gpu_stats", return_value=(None, None)):
            poller.start()
            try:
                poller.stop()
                self.assertTrue(poller.wait(1000))
            finally:
                poller.stop()
                poller.wait()

    def test_cancelled_transcription_does_not_start_loading_a_model(self):
        worker = transcriber.TranscribeWorker([], [], "large-v3", "cpu", None)
        worker.stop()
        with patch.object(worker, "_load_model") as load:
            worker.run()
            load.assert_not_called()

    def test_a_language_the_model_lacks_fails_before_loading_or_retrying(self):
        worker = transcriber.TranscribeWorker([], [0, 1], "qwen3-asr-1.7b", "cuda", "no")
        failures, finished = [], []
        worker.track_failed.connect(lambda i, message: failures.append((i, message)))
        worker.finished_all.connect(finished.append)
        with patch("transcriber._get_model") as load:
            worker.run()
            load.assert_not_called()
        self.assertEqual([i for i, _ in failures], [0, 1])
        self.assertIn("can't transcribe", failures[0][1])
        self.assertEqual(finished, [False])
        self.assertEqual(worker.device, "cuda")  # no CPU retry was attempted

    def test_cancelled_notes_does_not_download_or_spawn_a_server(self):
        worker = llm.NotesWorker("Alice: hello")
        failures = []
        worker.failed.connect(failures.append)
        worker.stop()
        with patch("llm._download_gguf") as download, patch("llm._spawn_server") as spawn:
            worker.run()
        download.assert_not_called()
        spawn.assert_not_called()
        self.assertTrue(worker.cancelled)
        self.assertEqual(failures, [])

    def test_cancel_while_server_starts_kills_the_process(self):
        process = Mock()
        with self.assertRaises(llm._NotesCancelled):
            llm._await_ready(process, 1, cancelled=lambda: True)
        process.kill.assert_called_once()
        process.wait.assert_called_once()

    def test_releasing_whisper_cache_does_not_import_torch(self):
        old_cache = dict(transcriber._model_cache)
        self.addCleanup(transcriber._model_cache.update, old_cache)
        transcriber._model_cache.update(key=("whisper", "tiny", "cpu"), model=object())
        with patch.dict(sys.modules, {"torch": None}):
            transcriber.free_asr_models()
        self.assertIsNone(transcriber._model_cache["model"])
        self.assertIsNone(transcriber._model_cache["key"])

    def test_missing_cuda_driver_is_a_cpu_result(self):
        with patch("ctypes.WinDLL", side_effect=OSError("no driver"), create=True):
            self.assertFalse(transcriber.cuda_available())


if __name__ == "__main__":
    unittest.main()
