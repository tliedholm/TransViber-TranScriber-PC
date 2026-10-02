"""Startup, clearing, caching and shutdown regressions; no inference or GPU."""

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import tests  # noqa: F401

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QSettings
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import QApplication

import llm
import main_window
import transcriber
from models import Segment, Track
from stats import StatsPoller

_app = QApplication.instance() or QApplication([])


class WindowLifecycle(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        folder = Path(self.temp.name)
        self.notes_path = folder / "notes.md"
        self.settings = QSettings(str(folder / "settings.ini"), QSettings.Format.IniFormat)
        for target, kwargs in (
            ("main_window.QSettings", {"return_value": self.settings}),
            ("main_window.runtime.notes_path", {"return_value": self.notes_path}),
            ("main_window.StatsPoller.start", {}),
            ("main_window.DeviceProbe.start", {}),
        ):
            p = patch(target, **kwargs)
            p.start()
            self.addCleanup(p.stop)
        self.window = main_window.MainWindow()
        self.addCleanup(self.window.close)
        _app.processEvents()  # consume the deferred, mocked device probe
        self.window._on_device_detected(True)

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
            "heavy = {'torch', 'ctranslate2', 'numpy', 'faster_whisper', 'onnxruntime'}; "
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
