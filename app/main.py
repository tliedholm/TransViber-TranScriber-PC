"""TransViber TransScriber — entry point."""

import sys
from pathlib import Path

# Nothing heavy is imported at module level on purpose: _setup_frozen_logging()
# has to run before anything that can fail on import (PySide6, numpy, the CUDA
# setup, the app modules), or a startup crash in the windowed build happens
# while there is still no console and no log file to record it. Qt and
# MainWindow are imported inside main(), after logging is wired up.


def _selftest(argv: list[str]) -> int:
    """Headless smoke test: `--selftest audio.wav [model ...]` transcribes the
    file with each model (GPU first, CPU fallback as usual) and prints the
    segments; `--selftest --notes audio.wav` additionally exercises the
    notes engine.
    Used to verify the standalone build without clicking through the GUI.
    """
    from PySide6.QtCore import QCoreApplication

    QCoreApplication(sys.argv[:1])
    # --notes, or --notes=<model> to exercise one specific notes model —
    # the catalog carries models that exist to be compared, and comparing
    # them through the GUI is neither repeatable nor scriptable.
    notes_model = ""
    if argv and argv[0].startswith("--notes"):
        notes_model = argv[0].partition("=")[2] or "qwen3-4b"
        argv = argv[1:]
    if not argv:
        print("usage: main.py --selftest [--notes[=model]] <audio-file> [model ...]")
        return 2
    audio = Path(argv[0])
    if not audio.is_file():
        print(f"selftest: no such audio file: {audio}")
        return 2
    models = argv[1:] or ["distil-large-v3.5-ct2"]

    import transcriber
    from audio import tracks_for_file
    from transcriber import TranscribeWorker

    failures: list[str] = []
    device = "cuda" if transcriber.cuda_available() else "cpu"
    print(f"device: {device}")
    for model_name in models:
        print(f"\n=== {model_name} ===")
        tracks = tracks_for_file(audio)
        worker = TranscribeWorker(tracks, list(range(len(tracks))),
                                  model_name=model_name, device=device, language=None)
        worker.model_status.connect(lambda m: print(f"  status: {m}"))
        worker.segment_ready.connect(
            lambda i, s: print(f"  [{s.start:7.2f}-{s.end:7.2f}] {s.speaker}: {s.text.strip()}"))
        worker.track_failed.connect(
            lambda i, m: (print(f"  FAILED: {m}"), failures.append(f"{model_name}: {m}")))
        worker.track_note.connect(lambda i, m: print(f"  note: {m}"))
        worker.run()  # synchronously, in this thread
        transcriber.free_asr_models()

    if notes_model:
        import llm

        print(f"\n=== notes ({notes_model}) ===")
        worker = llm.NotesWorker("Alice: The dragon attacked us.\nBob: We ran away fast.",
                                 model_name=notes_model)
        worker.status.connect(lambda m: print(f"  status: {m}"))
        worker.chunk.connect(lambda t: print(t, end="", flush=True))
        worker.failed.connect(lambda m: (print(f"  FAILED: {m}"), failures.append(f"notes: {m}")))
        worker.run()
        llm.free_llm()
        print()

    print(f"\nselftest: {'FAIL — ' + '; '.join(failures) if failures else 'OK'}")
    return 1 if failures else 0


def _setup_frozen_logging() -> None:
    """Log both pythonw source launches and the windowed standalone build.

    With no console, tracebacks and startup timings would otherwise vanish.
    """
    import runtime

    if (not runtime.IS_FROZEN and sys.stderr is not None
            or sys.stderr is not None and sys.stderr.isatty()):
        return
    log_path = runtime.writable_dir() / "app.log"
    # Append across launches so a crash log survives the user relaunching the
    # app before reporting it; start fresh once the file gets unwieldy.
    try:
        keep = log_path.is_file() and log_path.stat().st_size < 1_000_000
    except OSError:
        keep = False
    try:
        log = open(log_path, "a" if keep else "w",
                   encoding="utf-8", errors="replace", buffering=1)
    except OSError:
        return
    sys.stdout = sys.stderr = log
    from datetime import datetime

    log.write(f"\n=== TransViber TransScriber started {datetime.now():%Y-%m-%d %H:%M:%S} ===\n")
    import faulthandler

    faulthandler.enable(log)


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "--selftest":
        return _selftest(sys.argv[2:])
    _setup_frozen_logging()
    from time import perf_counter

    started = perf_counter()

    def milestone(message: str) -> None:
        print(f"Startup +{perf_counter() - started:.3f}s: {message}", flush=True)

    milestone("loading interface")

    # Everything below can fail on import; it is loaded only now so that a
    # failure lands in the log file set up above.
    import ctypes

    from PySide6.QtCore import QTimer
    from PySide6.QtGui import QIcon
    from PySide6.QtWidgets import QApplication

    import runtime
    from main_window import MainWindow
    milestone("interface imported")

    # Give the process its own taskbar identity so Windows shows the app
    # icon instead of pythonw's.
    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("TransViber.TransScriber")

    app = QApplication(sys.argv)
    app.setApplicationName("TransViber TransScriber")
    app.setOrganizationName("TransViber")
    icon_path = runtime.asset_path("icon.png")
    if icon_path.exists():
        app.setWindowIcon(QIcon(str(icon_path)))
    window = MainWindow()
    window.show()
    QTimer.singleShot(0, lambda: milestone("window ready"))
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
