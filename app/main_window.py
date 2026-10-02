"""Main application window."""

from __future__ import annotations

import html
import time
from pathlib import Path

from PySide6.QtCore import QSettings, Qt, QTimer, Slot
from PySide6.QtGui import (
    QDragEnterEvent,
    QDropEvent,
    QGuiApplication,
    QTextCursor,
)
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSplitter,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from datetime import datetime

import exporters
import llm
import runtime
from audio import channel_suffixes, probe_streams, tracks_for_file
from merger import coalesce, merge_tracks
from models import AUDIO_EXTENSIONS, Track, TrackStatus
from stats import DeviceProbe, StatsPoller
from transcriber import (
    TranscribeWorker,
    engine_for,
    free_asr_models,
)

# Grouped by the user's preferred quality/speed tiers.
MODEL_GROUPS = [
    ("Max Detail (Large Tier)", [
        "qwen3-asr-1.7b", "large-v3", "large-v2",
    ]),
    ("Standard Speed (Medium Tier)", [
        "canary-1b-v2", "large-v3-turbo", "medium", "medium.en",
    ]),
    ("Quick Review / Rapid Capture (Small Tier)", [
        "distil-large-v3.5-ct2", "distil-large-v3", "qwen3-asr-0.6b",
        "parakeet-tdt-0.6b-v3", "parakeet-tdt-0.6b-v2",
    ]),
]

LANGUAGES = [
    ("Auto-detect", None), ("English", "en"), ("Swedish", "sv"), ("Spanish", "es"),
    ("French", "fr"), ("German", "de"), ("Italian", "it"), ("Portuguese", "pt"),
    ("Dutch", "nl"), ("Danish", "da"), ("Norwegian", "no"), ("Finnish", "fi"),
    ("Polish", "pl"), ("Russian", "ru"), ("Ukrainian", "uk"), ("Japanese", "ja"),
    ("Korean", "ko"), ("Chinese", "zh"), ("Arabic", "ar"), ("Hindi", "hi"),
    ("Turkish", "tr"),
]

COL_SPEAKER, COL_FILE, COL_DURATION, COL_STATUS, COL_PROGRESS = range(5)

FILE_FILTER = "Audio files (*" + " *".join(sorted(AUDIO_EXTENSIONS)) + ");;All files (*.*)"

EXPORT_FILTERS = (
    "Markdown (*.md);;Plain text (*.txt);;SubRip subtitles (*.srt);;"
    "WebVTT subtitles (*.vtt);;JSON (*.json);;All formats (*.md *.txt *.srt *.vtt *.json)"
)


def _fmt_duration(seconds: float) -> str:
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def _add_grouped_models(combo: QComboBox,
                        groups: list[tuple[str, list[str]]]) -> None:
    """Populate a combo with bold, non-selectable tier headings."""
    first_model_index = -1
    for group_number, (label, names) in enumerate(groups):
        if group_number:
            combo.insertSeparator(combo.count())
        combo.addItem(label)
        header = combo.model().item(combo.count() - 1)
        if header is not None:
            header.setEnabled(False)
            header.setSelectable(False)
            font = header.font()
            font.setBold(True)
            header.setFont(font)
        for name in names:
            combo.addItem(name)
            if first_model_index < 0:
                first_model_index = combo.count() - 1
    if first_model_index >= 0:
        combo.setCurrentIndex(first_model_index)


class _TranscriptPane:
    """Keeps one read-only view in step with the transcript by appending.

    Rebuilding the whole document is what a naive refresh does, and the cost
    of that grows with the session — on a four-hour recording the app would
    re-lay-out thousands of paragraphs every 400 ms while still transcribing,
    which is exactly the case this app exists for.

    Almost every refresh during transcription only lengthens the transcript,
    so almost every refresh is an append. Two things are not appends and are
    handled honestly rather than papered over: the *last* block keeps growing
    as its speaker carries on talking, so it is redrawn every time; and a
    segment from another track can sort in ahead of blocks already drawn, in
    which case the whole document is rebuilt. `blocks` are complete HTML
    paragraphs, already escaped by the caller.
    """

    def __init__(self, view, stick_to_bottom: bool) -> None:
        self._view = view
        self._stick_to_bottom = stick_to_bottom
        self._blocks: list[str] = []
        self._tail_start = 0

    def update(self, blocks: list[str]) -> None:
        """Show `blocks`, doing as little work as they allow.

        A hidden pane is skipped and left marked out of date: the notes
        mirror can be toggled off, and the tab that isn't showing has no
        reason to lay anything out. refresh() catches it up when it reappears.
        """
        if blocks == self._blocks or not self._view.isVisible():
            return
        scrollbar = self._view.verticalScrollBar()
        at_bottom = scrollbar.value() >= scrollbar.maximum() - 10
        position = scrollbar.value()

        if not blocks:
            self._view.clear()
            self._blocks, self._tail_start = [], 0
            return

        settled = self._blocks[:-1]
        if self._blocks and blocks[:len(settled)] == settled:
            cursor = self._view.textCursor()
            cursor.setPosition(self._tail_start)
            cursor.movePosition(QTextCursor.MoveOperation.End,
                                QTextCursor.MoveMode.KeepAnchor)
            cursor.removeSelectedText()
            self._append(cursor, blocks[len(settled):-1])
        else:
            self._view.clear()
            cursor = self._view.textCursor()
            cursor.movePosition(QTextCursor.MoveOperation.End)
            self._append(cursor, blocks[:-1])
        self._tail_start = cursor.position()
        self._append(cursor, blocks[-1:])
        self._blocks = list(blocks)

        if self._stick_to_bottom and at_bottom:
            scrollbar.setValue(scrollbar.maximum())
        elif not self._stick_to_bottom:
            scrollbar.setValue(position)

    def refresh(self, blocks: list[str]) -> None:
        """Catch up a pane that was hidden while the transcript changed.

        Called on every tab switch, so a pane that is already showing the
        right thing has to be left alone: throwing its document away and
        rebuilding it is exactly the work this class exists to avoid.
        """
        if self._blocks == blocks:
            return
        self.update(blocks)

    @staticmethod
    def _append(cursor, fragments: list[str]) -> None:
        for fragment in fragments:
            # insertHtml() of a <p> continues the current paragraph rather
            # than starting one, so every speaker block needs its own block
            # opened first — except into a document that is still empty.
            if not cursor.atStart() or cursor.block().length() > 1:
                cursor.insertBlock()
            cursor.insertHtml(fragment)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TransViber TransScriber")
        self.resize(1100, 750)
        self.setAcceptDrops(True)

        self.tracks: list[Track] = []
        self.worker: TranscribeWorker | None = None
        self.notes_worker: llm.NotesWorker | None = None
        self.has_cuda: bool | None = None
        self._closing = False
        self._notes_load_failed = False

        self._build_ui()
        self._transcript_blocks: list[str] = []
        self._panes = [
            _TranscriptPane(self.transcript_view, stick_to_bottom=True),
            _TranscriptPane(self.notes_transcript_view, stick_to_bottom=False),
        ]
        self.tabs.currentChanged.connect(self._refresh_visible_panes)
        self.check_show_transcript.toggled.connect(self._refresh_visible_panes)
        self._load_settings()
        self._refresh_timer = QTimer(self)
        self._refresh_timer.setInterval(400)
        self._refresh_timer.setSingleShot(True)
        self._refresh_timer.timeout.connect(self._render_transcript)
        self.device_probe = DeviceProbe(self)
        self.device_probe.detected.connect(self._on_device_detected)
        QTimer.singleShot(0, lambda: self.device_probe.start() if not self._closing else None)
        self._close_timer = QTimer(self)
        self._close_timer.setInterval(100)
        self._close_timer.timeout.connect(self.close)

    # ---------- UI construction ----------

    def _build_ui(self) -> None:
        self.tabs = QTabWidget()
        transcribe_tab = QWidget()
        root = QVBoxLayout(transcribe_tab)

        # Top control bar
        bar = QHBoxLayout()
        self.btn_add = QPushButton("Add Files…")
        self.btn_add.clicked.connect(self.add_files_dialog)
        self.btn_remove = QPushButton("Remove")
        self.btn_remove.clicked.connect(self.remove_selected)
        self.btn_clear = QPushButton("Clear")
        self.btn_clear.clicked.connect(self.clear_tracks)
        bar.addWidget(self.btn_add)
        bar.addWidget(self.btn_remove)
        bar.addWidget(self.btn_clear)
        bar.addSpacing(20)

        bar.addWidget(QLabel("Model:"))
        self.combo_model = QComboBox()
        _add_grouped_models(self.combo_model, MODEL_GROUPS)
        self.combo_model.setEditable(True)
        self.combo_model.setToolTip(
            "Whisper models: pick a size, or type any CTranslate2 repo from\n"
            "Hugging Face (e.g. distil-whisper/distil-large-v3.5-ct2).\n"
            "'.en' models are English-only and slightly more accurate for English.\n"
            "\n"
            "Qwen3-ASR (1.7b/0.6b): LLM-based — strong on names, jargon and\n"
            "punctuation; 30 languages plus 22 Chinese dialects (the forced\n"
            "aligner that supplies word timings covers 11); supports\n"
            "vocabulary hints. Apache-2.0.\n"
            "\n"
            "NVIDIA models (via onnx-asr): parakeet-tdt-0.6b-v2 (English) and\n"
            "-v3 (25 European languages) — the fastest here, and strong on\n"
            "published English WER; canary-1b-v2 leads published multilingual\n"
            "benchmarks. All CC-BY-4.0. None of those rankings has been\n"
            "measured on recordings like yours.\n"
            "\n"
            "The Language setting applies to Whisper and Qwen. NVIDIA models\n"
            "ignore it and auto-detect."
        )
        bar.addWidget(self.combo_model)
        bar.addWidget(self._make_star_button(self.combo_model, "starred_model"))

        bar.addWidget(QLabel("Language:"))
        self.combo_lang = QComboBox()
        for name, code in LANGUAGES:
            self.combo_lang.addItem(name, code)
        bar.addWidget(self.combo_lang)

        bar.addWidget(QLabel("Device:"))
        self.combo_device = QComboBox()
        self.combo_device.addItem("GPU (checking…)", "cuda")
        self.combo_device.addItem("CPU", "cpu")
        self.combo_device.setEnabled(False)
        bar.addWidget(self.combo_device)

        bar.addStretch()

        # Elapsed time of the running (or last finished) transcription.
        self.elapsed_label = QLabel("")
        self.elapsed_label.setToolTip("How long the current or last transcription run took,\n"
                                      "including model loading/downloading.")
        bar.addWidget(self.elapsed_label)
        self._run_started: float | None = None
        self._elapsed_timer = QTimer(self)
        self._elapsed_timer.setInterval(1000)
        self._elapsed_timer.timeout.connect(self._tick_elapsed)

        self.btn_transcribe = QPushButton("Transcribe")
        self.btn_transcribe.setDefault(True)
        self.btn_transcribe.setEnabled(False)
        self.btn_transcribe.clicked.connect(self.toggle_transcription)
        bar.addWidget(self.btn_transcribe)
        self.btn_export = QPushButton("Export…")
        self.btn_export.clicked.connect(self.export_dialog)
        self.btn_export.setEnabled(False)
        bar.addWidget(self.btn_export)
        root.addLayout(bar)

        # Vocabulary hints (Whisper/Qwen only): bias recognition toward unusual
        # names and terms — campaign characters, places, jargon. Lives in a
        # full-height panel on the left (below) so every entered term stays
        # visible. Hidden when a non-Whisper model (Parakeet/Canary) is chosen.
        self.hints_widget = QWidget()
        hints_col = QVBoxLayout(self.hints_widget)
        hints_col.setContentsMargins(0, 0, 6, 0)
        hints_col.addWidget(QLabel("Vocabulary hints:"))
        self.edit_hints = QPlainTextEdit()
        self.edit_hints.setPlaceholderText(
            "Names and terms to spell correctly — one per line or "
            "comma-separated, e.g.\nZephyrine\nThaldrin\nWaterdeep\nvecna\nowlbear"
        )
        self.edit_hints.setToolTip(
            "Words or phrases the audio is likely to contain, one per line or\n"
            "comma-separated. Whisper is nudged toward these spellings — great\n"
            "for D&D character names, places, and other words it would\n"
            "otherwise mangle. Remembered between sessions."
        )
        hints_col.addWidget(self.edit_hints)
        self.combo_model.currentTextChanged.connect(self._update_hints_visibility)

        # Track table + transcript view (stacked on the right)
        right_splitter = QSplitter(Qt.Orientation.Vertical)

        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["Speaker", "File", "Duration", "Status", "Progress"])
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(COL_SPEAKER, QHeaderView.ResizeMode.Interactive)
        header.setSectionResizeMode(COL_FILE, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(COL_DURATION, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(COL_STATUS, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(COL_PROGRESS, QHeaderView.ResizeMode.Fixed)
        self.table.setColumnWidth(COL_SPEAKER, 180)
        self.table.setColumnWidth(COL_PROGRESS, 140)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.verticalHeader().setVisible(False)
        self.table.itemChanged.connect(self._on_item_changed)
        self.table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self._show_table_menu)
        right_splitter.addWidget(self.table)

        self.transcript_view = QTextBrowser()
        self.transcript_view.setOpenExternalLinks(False)
        self.transcript_view.setPlaceholderText(
            "Drop audio tracks above (one file per speaker), set speaker names, "
            "then click Transcribe. The merged conversation appears here."
        )
        right_splitter.addWidget(self.transcript_view)
        right_splitter.setStretchFactor(0, 1)
        right_splitter.setStretchFactor(1, 2)

        # Hints on the left (full height), tracks + transcript on the right.
        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        main_splitter.addWidget(self.hints_widget)
        main_splitter.addWidget(right_splitter)
        main_splitter.setStretchFactor(0, 0)
        main_splitter.setStretchFactor(1, 1)
        main_splitter.setSizes([260, 840])
        root.addWidget(main_splitter)

        # Notes tab: instructions and notes sit side by side, with a read-only
        # transcript mirror spanning the full width underneath both.
        notes_tab = QSplitter(Qt.Orientation.Vertical)
        notes_top = QSplitter(Qt.Orientation.Horizontal)

        # AI instructions: extra guidance layered on top of the standard note
        # format. Full-height multi-line panel so long instructions stay
        # visible — mirrors the Vocabulary hints panel on the Transcribe tab.
        self.ai_instr_widget = QWidget()
        ai_instr_col = QVBoxLayout(self.ai_instr_widget)
        ai_instr_col.setContentsMargins(0, 0, 6, 0)
        ai_instr_col.addWidget(QLabel("AI instructions:"))
        self.edit_ai_instructions = QPlainTextEdit()
        self.edit_ai_instructions.setPlaceholderText(
            "Optional extra AI instructions, e.g.\nfocus on plot events and NPC "
            "names\nkeep the summary under 200 words\ncall out unresolved "
            "cliffhangers"
        )
        self.edit_ai_instructions.setToolTip(
            "Added on top of the standard note format (summary, key points,\n"
            "quotes, follow-ups, per-person session summaries).\n"
            "Remembered between sessions."
        )
        ai_instr_col.addWidget(self.edit_ai_instructions)
        notes_top.addWidget(self.ai_instr_widget)

        self.notes_transcript_view = QTextBrowser()
        self.notes_transcript_view.setPlaceholderText(
            "The merged transcript appears here once something is transcribed."
        )

        notes_right = QWidget()
        notes_layout = QVBoxLayout(notes_right)
        notes_layout.setContentsMargins(4, 0, 0, 0)
        notes_toolbar = QHBoxLayout()
        # View toggles: show/hide the instructions and transcript panes so the
        # notes editor can take the full width when they aren't needed.
        notes_toolbar.addWidget(QLabel("Show:"))
        self.check_show_instructions = QCheckBox("Instructions")
        self.check_show_instructions.setChecked(True)
        self.check_show_instructions.setToolTip(
            "Show or hide the AI-instructions panel. Hidden instructions are\n"
            "still applied to AI notes. Remembered between sessions."
        )
        self.check_show_instructions.toggled.connect(self.ai_instr_widget.setVisible)
        notes_toolbar.addWidget(self.check_show_instructions)
        self.check_show_transcript = QCheckBox("Transcript")
        self.check_show_transcript.setChecked(True)
        self.check_show_transcript.setToolTip(
            "Show or hide the merged-transcript panel.\n"
            "Remembered between sessions."
        )
        self.check_show_transcript.toggled.connect(self.notes_transcript_view.setVisible)
        notes_toolbar.addWidget(self.check_show_transcript)
        notes_toolbar.addSpacing(12)
        notes_toolbar.addWidget(QLabel("AI model:"))
        self.combo_notes_model = QComboBox()
        _add_grouped_models(self.combo_notes_model, llm.LLM_GROUPS)
        self.combo_notes_model.setEditable(True)
        self.combo_notes_model.setCurrentText(llm.DEFAULT_LLM)
        self.combo_notes_model.setToolTip(
            "Local model that writes the notes. One entry per model, at\n"
            "the best quant that comfortably fits 16 GB.\n"
            "qwen3-4b (default) — the proven quick option.\n"
            "qwen3-14b / mistral-nemo-12b — the large tier.\n"
            "llama-3.1-8b — mature, steady middle option.\n"
            "llama-3.2-3b / phi-4-mini — smaller and faster.\n"
            "\n"
            "'Benchmark Candidates' are newer models added so they can be\n"
            "compared, not because they are known to be better. Of these\n"
            "only gpt-oss-20b has actually been run here. gemma-4-12b\n"
            "supersedes gemma-3-12b, qwen3.5-9b supersedes qwen3-8b,\n"
            "ministral-3-8b supersedes ministral-8b (and is Apache-2.0\n"
            "rather than non-commercial), granite-4.2-3b supersedes\n"
            "granite-3.3-2b. gpt-oss-20b and qwen3.8-27b are tight on\n"
            "16 GB; lfm2.5 is under the LFM Open License, not Apache-2.0.\n"
            "\n"
            "Anything cut from this list is still available by name — add\n"
            "a '-q4' suffix for the smaller build of a model here\n"
            "(e.g. qwen3-14b-q4), or type gemma-3-12b, qwen3-8b,\n"
            "ministral-8b, granite-3.3-2b, ministral-3-14b or\n"
            "granite-4.2-8b directly.\n"
            "\n"
            "You can also type any Hugging Face GGUF repo id, or\n"
            "'repo_id::filename.gguf' to pick one exact file."
        )
        notes_toolbar.addWidget(self.combo_notes_model, 1)
        notes_toolbar.addWidget(
            self._make_star_button(self.combo_notes_model, "starred_notes_model"))
        notes_toolbar.addStretch()
        self.notes_progress = QProgressBar()
        self.notes_progress.setRange(0, 100)
        self.notes_progress.setValue(0)
        self.notes_progress.setFixedWidth(170)
        self.notes_progress.setVisible(False)
        notes_toolbar.addWidget(self.notes_progress)
        self.notes_elapsed_label = QLabel("")
        notes_toolbar.addWidget(self.notes_elapsed_label)
        self._notes_run_started: float | None = None
        self._notes_tokens_generated = 0
        self._notes_elapsed_timer = QTimer(self)
        self._notes_elapsed_timer.setInterval(1000)
        self._notes_elapsed_timer.timeout.connect(self._tick_notes_elapsed)
        self.btn_ai_notes = QPushButton("AI Notes")
        self.btn_ai_notes.setToolTip(
            "Have a local AI (Qwen3-4B, runs on your GPU) write structured notes\n"
            "on the transcript, appended below. Locked while transcribing —\n"
            "the two can't share the GPU's memory."
        )
        self.btn_ai_notes.clicked.connect(self.toggle_ai_notes)
        notes_toolbar.addWidget(self.btn_ai_notes)
        self.btn_export_notes = QPushButton("Export Notes…")
        self.btn_export_notes.clicked.connect(self.export_notes_dialog)
        notes_toolbar.addWidget(self.btn_export_notes)
        notes_layout.addLayout(notes_toolbar)
        self.notes_edit = QPlainTextEdit()
        self.notes_edit.setPlaceholderText(
            "Take notes on the transcript here — saved automatically between sessions."
        )
        notes_layout.addWidget(self.notes_edit)
        # Autosave a few seconds after typing stops, so notes survive crashes.
        self._notes_save_timer = QTimer(self)
        self._notes_save_timer.setInterval(3000)
        self._notes_save_timer.setSingleShot(True)
        self._notes_save_timer.timeout.connect(self._save_notes_text)
        self.notes_edit.textChanged.connect(self._notes_save_timer.start)
        notes_top.addWidget(notes_right)
        notes_top.setStretchFactor(0, 0)
        notes_top.setStretchFactor(1, 1)
        notes_top.setSizes([300, 800])

        notes_tab.addWidget(notes_top)
        notes_tab.addWidget(self.notes_transcript_view)
        notes_tab.setStretchFactor(0, 1)
        notes_tab.setStretchFactor(1, 1)
        notes_tab.setSizes([375, 375])

        self.tabs.addTab(transcribe_tab, "Transcribe")
        self.tabs.addTab(notes_tab, "Notes")
        self.setCentralWidget(self.tabs)

        self.stats_label = QLabel("")
        self.stats_label.setToolTip(
            "Whole-system utilization, refreshed every 2 s — not just this app.\n"
            "VRAM is where the models live; RAM includes OS caches, so it\n"
            "normally reads high."
        )
        self.statusBar().addPermanentWidget(self.stats_label)
        self.stats_poller = StatsPoller(self)
        self.stats_poller.stats.connect(self.stats_label.setText)
        self.stats_poller.start()

        self.check_dark = QCheckBox("Dark mode")
        self.check_dark.setChecked(
            QGuiApplication.styleHints().colorScheme() == Qt.ColorScheme.Dark)
        self.check_dark.toggled.connect(self._set_dark_mode)
        self.statusBar().addPermanentWidget(self.check_dark)

        self.device_label = QLabel("Checking GPU…")
        self.statusBar().addPermanentWidget(self.device_label)
        self.statusBar().showMessage("Ready")

    @Slot(bool)
    def _on_device_detected(self, available: bool) -> None:
        self.has_cuda = available
        self.combo_device.setItemText(0, "GPU (CUDA)" if available else "GPU (unavailable)")
        self.combo_device.model().item(0).setEnabled(available)
        if not available:
            self.combo_device.setCurrentIndex(1)
        self.device_label.setText(
            "GPU: CUDA available" if available else "GPU not available — using CPU")
        if not self._closing:
            self.combo_device.setEnabled(self.worker is None)
            self.btn_transcribe.setEnabled(self.notes_worker is None)

    @staticmethod
    def _format_elapsed(seconds: float) -> str:
        m, s = divmod(int(seconds), 60)
        h, m = divmod(m, 60)
        return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"

    def _tick_elapsed(self) -> None:
        if self._run_started is not None:
            self.elapsed_label.setText(self._format_elapsed(time.monotonic() - self._run_started))

    def _set_dark_mode(self, dark: bool) -> None:
        QGuiApplication.styleHints().setColorScheme(
            Qt.ColorScheme.Dark if dark else Qt.ColorScheme.Light)
        QSettings().setValue("dark", dark)

    def _make_star_button(self, combo: QComboBox, settings_key: str) -> QPushButton:
        """A ☆/★ toggle that pins the combo's current model as the startup default."""
        button = QPushButton("☆")
        button.setFlat(True)
        button.setFixedWidth(28)
        button.setToolTip(
            "Star the selected model to make it the default whenever the app\n"
            "starts. Click again to unstar (the app then remembers whichever\n"
            "model you used last)."
        )

        def refresh() -> None:
            starred = QSettings().value(settings_key, "", str)
            button.setText(
                "★" if starred and starred == combo.currentText().strip() else "☆")

        def toggle() -> None:
            settings = QSettings()
            current = combo.currentText().strip()
            if settings.value(settings_key, "", str) == current:
                settings.remove(settings_key)
            elif current:
                settings.setValue(settings_key, current)
            refresh()

        button.clicked.connect(toggle)
        combo.currentTextChanged.connect(lambda _text: refresh())
        refresh()
        return button

    def _update_hints_visibility(self) -> None:
        model = self.combo_model.currentText().strip()
        self.hints_widget.setVisible(engine_for(model) in ("whisper", "qwen"))

    def _hotwords_text(self) -> str:
        """Collapse the multi-line hints panel into one comma-separated string.

        Users can enter terms one per line or comma-separated; the ASR engines
        want a single hotword string, so newlines become ", " separators.
        """
        lines = (line.strip() for line in self.edit_hints.toPlainText().splitlines())
        return ", ".join(line for line in lines if line)

    # ---------- Drag & drop ----------

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if self.worker is None and any(
            Path(u.toLocalFile()).suffix.lower() in AUDIO_EXTENSIONS
            for u in event.mimeData().urls()
        ):
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent) -> None:
        paths = [
            Path(u.toLocalFile())
            for u in event.mimeData().urls()
            if Path(u.toLocalFile()).suffix.lower() in AUDIO_EXTENSIONS
        ]
        self.add_files(paths)

    # ---------- Track management ----------

    def add_files_dialog(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(self, "Add audio tracks", "", FILE_FILTER)
        self.add_files([Path(f) for f in files])

    def add_files(self, paths: list[Path]) -> None:
        existing = {t.path for t in self.tracks}
        for path in paths:
            if path in existing or not path.is_file():
                continue
            for track in tracks_for_file(path):
                self.tracks.append(track)
                self._append_row(track)
            existing.add(path)
        self.statusBar().showMessage(f"{len(self.tracks)} track(s) loaded")

    def _append_row(self, track: Track, row: int | None = None) -> None:
        if row is None:
            row = self.table.rowCount()
        self.table.blockSignals(True)
        self.table.insertRow(row)

        speaker_item = QTableWidgetItem(track.speaker)
        self.table.setItem(row, COL_SPEAKER, speaker_item)

        for col, text in [
            (COL_FILE, track.source_label),
            (COL_DURATION, _fmt_duration(track.duration)),
            (COL_STATUS, track.status.value),
        ]:
            item = QTableWidgetItem(text)
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            if col == COL_FILE:
                item.setToolTip(str(track.path))
            self.table.setItem(row, col, item)

        progress = QProgressBar()
        progress.setRange(0, 100)
        progress.setValue(0)
        self.table.setCellWidget(row, COL_PROGRESS, progress)
        self.table.blockSignals(False)

    def _on_item_changed(self, item: QTableWidgetItem) -> None:
        if item.column() != COL_SPEAKER:
            return
        row = item.row()
        if 0 <= row < len(self.tracks):
            track = self.tracks[row]
            new_name = item.text().strip() or track.speaker
            track.speaker = new_name
            for seg in track.segments:
                seg.speaker = new_name
            self._schedule_refresh()

    def _show_table_menu(self, pos) -> None:
        row = self.table.rowAt(pos.y())
        if row < 0 or self.worker is not None:
            return
        track = self.tracks[row]
        menu = QMenu(self)
        split = menu.addAction("Split into one speaker per channel")
        split.setEnabled(track.channel is None)
        if menu.exec(self.table.viewport().mapToGlobal(pos)) is split:
            self._split_track_channels(row)

    def _split_track_channels(self, row: int) -> None:
        track = self.tracks[row]
        try:
            streams = probe_streams(track.path)
            # `or 0` here would conflate stream 0 with "no stream index", and
            # the file may have changed since it was probed.
            index = track.stream_index if track.stream_index is not None else 0
            if index >= len(streams):
                raise IndexError(
                    f"{track.path.name} no longer has stream {index + 1} "
                    f"({len(streams)} audio stream(s) found)")
            channels = streams[index]["channels"]
        except Exception as exc:
            QMessageBox.warning(self, "Cannot split", str(exc))
            return
        if channels < 2:
            QMessageBox.information(self, "Mono track",
                                    "This track has only one channel — nothing to split.")
            return
        self.table.removeRow(row)
        del self.tracks[row]
        for ch, suffix in enumerate(channel_suffixes(channels)):
            new = Track(path=track.path, speaker=track.speaker + suffix,
                        duration=track.duration, stream_index=track.stream_index,
                        channel=ch)
            self.tracks.insert(row + ch, new)
            self._append_row(new, row + ch)
        self._schedule_refresh()

    def remove_selected(self) -> None:
        if self.worker is not None:
            return
        rows = sorted({i.row() for i in self.table.selectedIndexes()}, reverse=True)
        for row in rows:
            self.table.removeRow(row)
            del self.tracks[row]
        self._schedule_refresh()

    def clear_tracks(self) -> None:
        if self.worker is not None:
            return
        self.table.setRowCount(0)
        self.tracks.clear()
        self._refresh_timer.stop()
        self._render_transcript()
        self.btn_export.setEnabled(False)

    # ---------- Transcription ----------

    def toggle_transcription(self) -> None:
        if self._closing or self.notes_worker is not None or self.has_cuda is None:
            return
        if self.worker is not None:
            self.btn_transcribe.setEnabled(False)
            self.btn_transcribe.setText("Cancelling…")
            self.worker.stop()
            return

        indices = [i for i, t in enumerate(self.tracks) if t.status != TrackStatus.DONE]
        if not indices:
            indices = list(range(len(self.tracks)))  # everything done -> re-run all
        if not indices:
            QMessageBox.information(self, "No tracks", "Add at least one audio track first.")
            return

        for i in indices:
            track = self.tracks[i]
            track.segments.clear()
            track.status = TrackStatus.PENDING
            track.error = ""
            self._set_status(i, track.status.value)
            self._progress_bar(i).setValue(0)

        llm.free_llm()  # transcription and the notes LLM can't share VRAM

        self._run_started = time.monotonic()
        self.elapsed_label.setText("0:00")
        self._elapsed_timer.start()

        device = self.combo_device.currentData()
        self.worker = TranscribeWorker(
            self.tracks, indices,
            model_name=self.combo_model.currentText().strip(),
            device=device,
            language=self.combo_lang.currentData(),
            hotwords=self._hotwords_text(),
        )
        w = self.worker
        w.model_status.connect(self.statusBar().showMessage)
        w.track_started.connect(self._on_track_started)
        w.segment_ready.connect(self._on_segment)
        w.track_progress.connect(lambda i, p: self._progress_bar(i).setValue(p))
        w.track_done.connect(self._on_track_done)
        w.track_failed.connect(self._on_track_failed)
        w.finished_all.connect(self._on_finished)
        self._set_running(True)
        w.start()

    def _set_running(self, running: bool) -> None:
        for widget in (self.btn_add, self.btn_remove, self.btn_clear,
                       self.combo_model, self.combo_lang, self.combo_device,
                       self.edit_hints, self.btn_ai_notes):
            widget.setEnabled(not running)
        self.btn_export.setEnabled(not running and bool(merge_tracks(self.tracks)))
        self.btn_transcribe.setText("Cancel" if running else "Transcribe")
        self.btn_transcribe.setEnabled(True)

    def _progress_bar(self, row: int) -> QProgressBar:
        return self.table.cellWidget(row, COL_PROGRESS)

    def _set_status(self, row: int, text: str) -> None:
        self.table.blockSignals(True)
        self.table.item(row, COL_STATUS).setText(text)
        self.table.item(row, COL_STATUS).setToolTip("")
        self.table.blockSignals(False)

    @Slot(int)
    def _on_track_started(self, i: int) -> None:
        track = self.tracks[i]
        track.segments.clear()  # a CPU-fallback retry restarts the track from scratch
        track.status = TrackStatus.TRANSCRIBING
        self._set_status(i, TrackStatus.TRANSCRIBING.value)
        self._progress_bar(i).setValue(0)
        self.statusBar().showMessage(f"Transcribing {track.path.name}…")
        self._schedule_refresh()

    @Slot(int, object)
    def _on_segment(self, i: int, segment) -> None:
        self.tracks[i].segments.append(segment)
        self._schedule_refresh()

    @Slot(int, str)
    def _on_track_done(self, i: int, language: str) -> None:
        track = self.tracks[i]
        track.status = TrackStatus.DONE
        track.language = language
        label = TrackStatus.DONE.value + (f" ({language})" if language else "")
        self._set_status(i, label)

    @Slot(int, str)
    def _on_track_failed(self, i: int, message: str) -> None:
        track = self.tracks[i]
        track.status = TrackStatus.ERROR
        track.error = message
        self._set_status(i, TrackStatus.ERROR.value)
        self.table.item(i, COL_STATUS).setToolTip(message)

    @Slot(bool)
    def _on_finished(self, cancelled: bool) -> None:
        if self.worker is not None:
            self.worker.wait()
            self.worker = None
        if self._closing:
            return
        if cancelled:
            for i, track in enumerate(self.tracks):
                if track.status == TrackStatus.TRANSCRIBING:
                    track.status = TrackStatus.CANCELLED
                    self._set_status(i, TrackStatus.CANCELLED.value)
        self._elapsed_timer.stop()
        took = ""
        if self._run_started is not None:
            elapsed = self._format_elapsed(time.monotonic() - self._run_started)
            self.elapsed_label.setText(f"took {elapsed}")
            self._run_started = None
            took = f" in {elapsed}"
        errors = [t for t in self.tracks if t.status == TrackStatus.ERROR]
        if cancelled:
            msg = "Cancelled"
        elif errors:
            msg = f"Finished with {len(errors)} error(s)"
        else:
            msg = f"Transcription complete{took}"
        self.statusBar().showMessage(msg)
        self._set_running(False)
        self._render_transcript()
        if errors and not cancelled:
            detail = "\n\n".join(f"{t.source_label}:\n{t.error}" for t in errors[:5])
            QMessageBox.warning(self, "Some tracks failed", detail)

    # ---------- Transcript view ----------

    def _schedule_refresh(self) -> None:
        if not self._refresh_timer.isActive():
            self._refresh_timer.start()

    def _render_transcript(self) -> None:
        segments = merge_tracks(self.tracks)
        self._transcript_blocks = [
            f"<p><b>{html.escape(seg.speaker)}</b> "
            f"<span style='color:gray'>[{_fmt_duration(seg.start)}]</span>: "
            f"{html.escape(seg.text.strip())}</p>"
            for seg in coalesce(segments)
        ]
        for pane in self._panes:
            pane.update(self._transcript_blocks)

    def _refresh_visible_panes(self, *_signal_args) -> None:
        """Catch up a pane that changed while it was hidden."""
        for pane in self._panes:
            pane.refresh(self._transcript_blocks)

    # ---------- Export ----------

    def _confirm_incomplete_export(self) -> bool:
        """Warn before exporting when a track stopped early.

        A failed or cancelled track keeps the segments it produced before it
        stopped, so the transcript can look finished while missing the tail
        of one speaker. Nothing else in the output says so.
        """
        incomplete = [t for t in self.tracks
                      if t.status in (TrackStatus.ERROR, TrackStatus.CANCELLED)
                      and t.segments]
        if not incomplete:
            return True
        listed = "\n".join(f"  • {t.source_label} — {t.status.value.lower()}"
                           for t in incomplete[:5])
        if len(incomplete) > 5:
            listed += f"\n  • …and {len(incomplete) - 5} more"
        answer = QMessageBox.warning(
            self, "Incomplete transcript",
            f"{len(incomplete)} track(s) stopped before finishing:\n\n{listed}\n\n"
            "Their transcripts are partial — the rest of what those speakers "
            "said is missing. Export anyway?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        return answer == QMessageBox.StandardButton.Yes

    def _confirm_overwrite(self, paths: list[Path]) -> bool:
        """Confirm replacing files the save dialog never showed the user.

        QFileDialog confirms exactly one name — the one that was typed.
        Anything else an export would replace (the other four formats under
        "All formats", or a name that only gained its extension after the
        dialog closed) is overwritten in silence unless it is confirmed here.
        """
        if not paths:
            return True
        listed = "\n".join(f"  • {p.name}" for p in paths[:6])
        if len(paths) > 6:
            listed += f"\n  • …and {len(paths) - 6} more"
        answer = QMessageBox.warning(
            self, "Replace existing files?",
            f"{len(paths)} existing file(s) in that folder will be replaced:\n\n"
            f"{listed}\n\nReplace them?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        return answer == QMessageBox.StandardButton.Yes

    def export_dialog(self) -> None:
        segments = merge_tracks(self.tracks)
        if not segments:
            return
        if not self._confirm_incomplete_export():
            return
        default_name = str(Path.home() / "transcript.md")
        path_str, selected = QFileDialog.getSaveFileName(
            self, "Export transcript", default_name, EXPORT_FILTERS
        )
        if not path_str:
            return
        path = Path(path_str)
        model_name = self.combo_model.currentText()
        try:
            if selected.startswith("All formats"):
                targets = exporters.target_paths(path)
                if not self._confirm_overwrite(
                        [t for t in targets if t != path and t.exists()]):
                    return
                written = exporters.export_all(path, segments, self.tracks, model_name)
                self.statusBar().showMessage("Exported: " + ", ".join(p.name for p in written))
            else:
                if path.suffix.lower() not in exporters.EXPORTERS:
                    ext = selected.split("*")[1].rstrip(") ")
                    # Append rather than with_suffix(): ".5" of "Session 2.5"
                    # is part of the name, not an extension, and replacing it
                    # writes a file the user never named.
                    path = path.with_name(path.name + ext)
                    if path.exists() and not self._confirm_overwrite([path]):
                        return
                exporters.export(path, segments, self.tracks, model_name)
                self.statusBar().showMessage(f"Exported {path}")
        except OSError as exc:
            QMessageBox.critical(self, "Export failed", str(exc))

    # ---------- AI notes ----------

    def toggle_ai_notes(self) -> None:
        if self._closing or self.worker is not None:
            return
        if self.notes_worker is not None:
            self.btn_ai_notes.setEnabled(False)
            self.btn_ai_notes.setText("Cancelling…")
            self.notes_worker.stop()
            llm.free_llm()  # end a stalled stream instead of waiting for its read timeout
            return
        segments = merge_tracks(self.tracks)
        if not segments:
            QMessageBox.information(self, "Nothing to summarize",
                                    "Transcribe something first — the AI takes notes "
                                    "on the transcript.")
            return
        transcript = exporters.to_text(segments)
        self.notes_worker = llm.NotesWorker(
            transcript,
            self.edit_ai_instructions.toPlainText().strip(),
            model_name=self.combo_notes_model.currentText().strip() or llm.DEFAULT_LLM,
        )
        w = self.notes_worker
        w.status.connect(self.statusBar().showMessage)
        w.chunk.connect(self._append_notes_text)
        w.progress.connect(self._on_notes_progress)
        w.completed.connect(self._on_ai_notes_done)
        w.failed.connect(self._on_ai_notes_failed)

        header = f"## AI Notes — {datetime.now():%Y-%m-%d %H:%M}\n\n"
        if self.notes_edit.toPlainText().strip():
            header = "\n\n" + header
        self._append_notes_text(header)

        self.btn_ai_notes.setText("Cancel AI Notes")
        self.btn_transcribe.setEnabled(False)  # VRAM lock, both directions
        self.combo_notes_model.setEnabled(False)
        self.notes_progress.setValue(0)
        self.notes_progress.setFormat("%p% · 0 tok")
        self.notes_progress.setVisible(True)
        self._notes_tokens_generated = 0
        self._notes_run_started = time.monotonic()
        self.notes_elapsed_label.setText("0:00")
        self._notes_elapsed_timer.start()
        w.start()

    @Slot(int, int)
    def _on_notes_progress(self, pct: int, tokens: int) -> None:
        self._notes_tokens_generated = tokens
        self.notes_progress.setValue(pct)
        self.notes_progress.setFormat(f"%p% · {tokens:,} tok")

    def _tick_notes_elapsed(self) -> None:
        if self._notes_run_started is None:
            return
        seconds = time.monotonic() - self._notes_run_started
        tokens = self._notes_tokens_generated
        rate = f" · {tokens / seconds:.1f} tok/s" if seconds > 0 and tokens else ""
        self.notes_elapsed_label.setText(
            f"{self._format_elapsed(seconds)} · {tokens:,} tok{rate}")

    def _append_notes_text(self, text: str) -> None:
        cursor = self.notes_edit.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        cursor.insertText(text)
        self.notes_edit.setTextCursor(cursor)

    def _on_ai_notes_done(self) -> None:
        cancelled = False
        if self.notes_worker is not None:
            self.notes_worker.wait()
            cancelled = self.notes_worker.cancelled
            self.notes_worker = None
        llm.free_llm()
        if self._closing:
            return
        self._notes_elapsed_timer.stop()
        stats = ""
        if self._notes_run_started is not None:
            seconds = time.monotonic() - self._notes_run_started
            elapsed = self._format_elapsed(seconds)
            tokens = self._notes_tokens_generated
            rate = f" · {tokens / seconds:.1f} tok/s avg" if seconds > 0 and tokens else ""
            stats = f" in {elapsed} · {tokens:,} tok{rate}"
            self._notes_run_started = None
        self.notes_elapsed_label.clear()  # live readout only runs during generation
        self.notes_progress.setVisible(False)
        self.btn_ai_notes.setText("AI Notes")
        self.btn_ai_notes.setEnabled(True)
        self.btn_transcribe.setEnabled(True)
        self.combo_notes_model.setEnabled(True)
        # Kept in the status bar until the next status change, so the run's
        # numbers can be noted down. A cancelled run says so: on the
        # map-reduce path "done" would be describing unmerged part notes.
        self.statusBar().showMessage(
            f"AI notes {'cancelled' if cancelled else 'done'}{stats}")

    def _on_ai_notes_failed(self, message: str) -> None:
        self._on_ai_notes_done()
        if self._closing:
            return
        self.statusBar().showMessage("AI notes failed")
        QMessageBox.critical(self, "AI notes failed", message)

    def export_notes_dialog(self) -> None:
        text = self.notes_edit.toPlainText()
        if not text.strip():
            QMessageBox.information(self, "No notes", "There are no notes to export yet.")
            return
        path_str, _ = QFileDialog.getSaveFileName(
            self, "Export notes", str(Path.home() / "notes.md"),
            "Markdown (*.md);;Plain text (*.txt)")
        if not path_str:
            return
        try:
            Path(path_str).write_text(text, encoding="utf-8")
            self.statusBar().showMessage(f"Notes exported to {path_str}")
        except OSError as exc:
            QMessageBox.critical(self, "Export failed", str(exc))

    # ---------- Settings persistence ----------

    def _save_notes_text(self) -> None:
        """Write the Notes tab to disk, atomically.

        Failures appear in the status bar: a modal error after every edit
        would be unusable. The text stays in the editor and is retried on
        the next edit and on close. A failed initial read disables autosave
        so an empty editor cannot overwrite the saved document.
        """
        if self._notes_load_failed:
            return  # never overwrite a document we failed to read
        try:
            runtime.write_atomic(runtime.notes_path(),
                                 self.notes_edit.toPlainText())
        except OSError as exc:
            self.statusBar().showMessage(f"Notes could not be saved: {exc}")

    def _read_notes_text(self) -> str:
        """The saved notes, migrating them out of the registry on first run."""
        path = runtime.notes_path()
        try:
            return path.read_text(encoding="utf-8")
        except FileNotFoundError:
            pass
        except (OSError, UnicodeError) as exc:
            self._notes_load_failed = True
            self.statusBar().showMessage(
                f"Saved notes could not be read; autosave disabled to protect the file: {exc}")
            return ""
        settings = QSettings()
        legacy = settings.value("notes", "", str)
        if legacy:
            try:
                runtime.write_atomic(path, legacy)
                settings.remove("notes")
            except OSError:
                pass  # keep the registry copy rather than losing it
        return legacy

    def _load_settings(self) -> None:
        settings = QSettings()
        self.edit_hints.setPlainText(settings.value("hints", "", str))
        self.notes_edit.blockSignals(True)
        try:
            self.notes_edit.setPlainText(self._read_notes_text())
        finally:
            self.notes_edit.blockSignals(False)
        self.edit_ai_instructions.setPlainText(settings.value("ai_instructions", "", str))
        self.check_show_instructions.setChecked(
            settings.value("notes_show_instructions", True, bool))
        self.check_show_transcript.setChecked(
            settings.value("notes_show_transcript", True, bool))
        # A starred model beats the last-used one at startup.
        notes_model = (settings.value("starred_notes_model", "", str)
                       or settings.value("notes_model", "", str))
        if notes_model:
            self.combo_notes_model.setCurrentText(notes_model)
        model = (settings.value("starred_model", "", str)
                 or settings.value("model", "", str))
        if model:
            self.combo_model.setCurrentText(model)
        lang_index = self.combo_lang.findData(settings.value("language", None))
        if lang_index >= 0:
            self.combo_lang.setCurrentIndex(lang_index)
        dark = settings.value("dark", None)
        if dark is not None:  # user chose explicitly before; otherwise follow the OS
            dark = dark in (True, "true")
            self.check_dark.setChecked(dark)
            self._set_dark_mode(dark)
        self._update_hints_visibility()

    def _save_settings(self) -> None:
        # The notes themselves are a file, not a registry value — see
        # _save_notes_text.
        self._save_notes_text()
        settings = QSettings()
        settings.setValue("hints", self.edit_hints.toPlainText())
        settings.setValue("ai_instructions", self.edit_ai_instructions.toPlainText())
        settings.setValue("notes_show_instructions",
                          self.check_show_instructions.isChecked())
        settings.setValue("notes_show_transcript",
                          self.check_show_transcript.isChecked())
        settings.setValue("notes_model", self.combo_notes_model.currentText().strip())
        settings.setValue("model", self.combo_model.currentText().strip())
        settings.setValue("language", self.combo_lang.currentData())

    # ---------- Shutdown ----------

    def closeEvent(self, event) -> None:
        if not self._closing:
            self._closing = True
            self._notes_save_timer.stop()
            self._save_settings()
            self.centralWidget().setEnabled(False)
            self.statusBar().showMessage("Closing — waiting for background work to stop…")
            if self.worker is not None:
                self.worker.stop()
            if self.notes_worker is not None:
                self.notes_worker.stop()
            self.stats_poller.stop()
        # Never destroy a running QThread. Keep the event loop alive for its
        # queued signals; a model download/load may not be interruptible yet.
        # Kill any notes server that was started during cancellation too.
        llm.free_llm()
        threads = (self.worker, self.notes_worker, self.stats_poller, self.device_probe)
        if any(thread is not None and thread.isRunning() for thread in threads):
            event.ignore()
            if not self._close_timer.isActive():
                self._close_timer.start()
            return
        self._close_timer.stop()
        self._save_settings()  # include any final notes chunks queued during cancellation
        free_asr_models()
        event.accept()
