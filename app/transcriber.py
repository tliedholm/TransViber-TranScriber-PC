"""Background transcription worker using faster-whisper (CTranslate2)."""

from __future__ import annotations

from contextlib import contextmanager

import cuda_setup  # noqa: F401  (must run before ctranslate2 loads)

from PySide6.QtCore import QThread, Signal

from audio import AudioFileError
from models import Segment, Track

# A pause this long between words splits a Whisper segment in two. Whisper can
# lump utterances separated by silence into one segment, which would sort a
# speaker's later remark at the time of their earlier one and break the
# conversation interleaving.
SPLIT_GAP_SECONDS = 1.0

# Friendly UI names for CTranslate2 Whisper repositories that Faster-Whisper
# does not include in its built-in short-name catalog.
WHISPER_ALIASES = {
    "distil-large-v3.5-ct2": "distil-whisper/distil-large-v3.5-ct2",
}

# NVIDIA models (and other non-Whisper models) run through onnx-asr instead
# of faster-whisper. UI names map to onnx-asr model ids.
ONNX_ALIASES = {
    "parakeet-tdt-0.6b-v3": "nemo-parakeet-tdt-0.6b-v3",
    "parakeet-tdt-0.6b-v2": "nemo-parakeet-tdt-0.6b-v2",
    "parakeet": "nemo-parakeet-tdt-0.6b-v3",
    "canary-1b-v2": "nemo-canary-1b-v2",
    "canary": "nemo-canary-1b-v2",
}


# Qwen3-ASR (Apache-2.0, 30 languages plus 22 Chinese dialects) runs through
# the official qwen-asr package on CUDA; word timestamps come from its
# companion forced-aligner model. Ported from the Mac edition's MLX
# integration.
QWEN_REPOS = {
    "qwen3-asr-1.7b": "Qwen/Qwen3-ASR-1.7B",
    "qwen3-asr-0.6b": "Qwen/Qwen3-ASR-0.6B",
    "qwen3-asr": "Qwen/Qwen3-ASR-1.7B",
    "qwen": "Qwen/Qwen3-ASR-1.7B",
}

QWEN_ALIGNER_REPO = "Qwen/Qwen3-ForcedAligner-0.6B"
QWEN_LANGUAGE_PROBE_SECONDS = 30

# Qwen expects full language names, not ISO codes. This is also the list of
# Language choices Qwen can take, so it holds only names qwen-asr accepts:
# anything outside its SUPPORTED_LANGUAGES raises ValueError (0.0.6 has no
# Norwegian or Ukrainian, which Whisper has), and only once the model has
# loaded. See language_supported.
QWEN_LANGUAGE_NAMES = {
    "en": "English", "sv": "Swedish", "es": "Spanish", "fr": "French",
    "de": "German", "it": "Italian", "pt": "Portuguese", "nl": "Dutch",
    "da": "Danish", "fi": "Finnish", "pl": "Polish", "ru": "Russian",
    "ja": "Japanese", "ko": "Korean", "zh": "Chinese", "ar": "Arabic",
    "hi": "Hindi", "tr": "Turkish",
}

# Scripts each Qwen language can be written in (unlisted languages use Latin).
# Forcing a language only biases Qwen's decoder — on a VAD region that is
# really just a breath or mic bump it can still emit Chinese filler such as
# "喂。", and with a forced language qwen-asr returns that output verbatim
# (parse_asr_output skips all checks). Segments written mostly in a script
# the locked language can't use are therefore noise hallucinations.
QWEN_LANGUAGE_SCRIPTS = {
    "Chinese": {"cjk"},
    "Cantonese": {"cjk"},
    "Japanese": {"cjk", "kana"},
    "Korean": {"hangul", "cjk"},
    "Arabic": {"arabic"},
    "Persian": {"arabic"},
    "Russian": {"cyrillic"},
    "Ukrainian": {"cyrillic"},
    "Macedonian": {"cyrillic"},
    "Greek": {"greek"},
    "Thai": {"thai"},
    "Hindi": {"devanagari"},
}

_SCRIPT_RANGES = (
    (0x0041, 0x024F, "latin"),   # ASCII through Latin Extended-B
    (0x0370, 0x03FF, "greek"),
    (0x0400, 0x052F, "cyrillic"),
    (0x0600, 0x077F, "arabic"),
    (0x0900, 0x097F, "devanagari"),
    (0x0E00, 0x0E7F, "thai"),
    (0x1E00, 0x1EFF, "latin"),   # Latin Extended Additional (Vietnamese)
    (0x1F00, 0x1FFF, "greek"),
    (0x3040, 0x30FF, "kana"),
    (0x31F0, 0x31FF, "kana"),
    (0x3400, 0x4DBF, "cjk"),
    (0x4E00, 0x9FFF, "cjk"),
    (0xAC00, 0xD7AF, "hangul"),
)


def _foreign_script(text: str, language: str) -> bool:
    """True when text is written mostly in a script `language` can't use."""
    allowed = QWEN_LANGUAGE_SCRIPTS.get(language, {"latin"})
    matching = foreign = 0
    for ch in text:
        if not ch.isalpha():
            continue
        cp = ord(ch)
        for lo, hi, script in _SCRIPT_RANGES:
            if lo <= cp <= hi:
                if script in allowed:
                    matching += 1
                else:
                    foreign += 1
                break
    return foreign > matching


def engine_for(model_name: str) -> str:
    n = model_name.lower()
    if n in QWEN_REPOS or n.startswith("qwen"):
        return "qwen"
    # Route anything Parakeet/Canary-shaped to onnx-asr, even misspellings of
    # the exact id — its "unknown model" error is clearer than faster-whisper
    # failing to find a Hugging Face repo of that name.
    if n in ONNX_ALIASES or n.startswith(
        ("nemo-", "parakeet", "canary", "gigaam", "alphacep/", "t-tech/")
    ):
        return "onnx"
    return "whisper"


def language_supported(model_name: str, language: str | None) -> bool:
    """Whether `model_name` can be told to transcribe `language`, a Language
    dropdown code (None is auto-detect). Only Qwen has a closed list: Whisper
    knows every language the dropdown offers, and the NVIDIA models ignore
    the setting."""
    return (language is None or engine_for(model_name) != "qwen"
            or language in QWEN_LANGUAGE_NAMES)


def _qwen_char_spans(text: str, words) -> list[tuple[int, int]] | None:
    """Locate each aligner word inside the full transcript text.

    The aligner emits bare words while result.text carries punctuation and
    casing; slicing the text between word positions keeps that formatting in
    the emitted segments. Returns None when a word can't be found (fall back
    to joining bare words).
    """
    spans = []
    pos = 0
    for word in words:
        idx = text.find(word.text, pos)
        if idx < 0:
            return None
        spans.append((idx, idx + len(word.text)))
        pos = idx + len(word.text)
    return spans


class _QwenModel:
    """Qwen3-ASR model plus the forced aligner that provides word timestamps."""

    def __init__(self, repo: str, device: str):
        import torch
        from qwen_asr import Qwen3ASRModel

        if device == "cuda":
            dtype, device_map = torch.bfloat16, "cuda:0"
        else:
            dtype, device_map = torch.float32, "cpu"
        self.model = Qwen3ASRModel.from_pretrained(
            repo,
            dtype=dtype,
            device_map=device_map,
            forced_aligner=QWEN_ALIGNER_REPO,
            forced_aligner_kwargs=dict(dtype=dtype, device_map=device_map),
        )


# The model is loaded lazily and cached so repeated runs (and multiple
# tracks) reuse it. Only one model is kept to limit VRAM use.
_model_cache: dict = {"key": None, "model": None}


def free_asr_models() -> None:
    """Release the cached ASR model and its VRAM (before loading the notes LLM)."""
    if _model_cache["model"] is None and _model_cache["key"] is None:
        return
    _model_cache["key"] = None
    _model_cache["model"] = None
    import gc

    gc.collect()
    try:
        import sys

        torch = sys.modules.get("torch")
        if torch is not None and torch.cuda.is_initialized():
            torch.cuda.empty_cache()
    except Exception:
        pass


def split_on_word_gaps(fw_segment, speaker: str, max_gap: float = SPLIT_GAP_SECONDS) -> list[Segment]:
    """Split a faster-whisper segment at silence gaps between its words."""
    words = fw_segment.words
    if not words:
        return [Segment(speaker, fw_segment.start, fw_segment.end, fw_segment.text)]
    pieces: list[Segment] = []
    current = [words[0]]
    for word in words[1:]:
        if word.start - current[-1].end > max_gap:
            pieces.append(Segment(speaker, current[0].start, current[-1].end,
                                  "".join(w.word for w in current)))
            current = [word]
        else:
            current.append(word)
    pieces.append(Segment(speaker, current[0].start, current[-1].end,
                          "".join(w.word for w in current)))
    return pieces


def cuda_available() -> bool:
    """Query the driver without importing an inference engine (or PyTorch).

    Model loading still checks backend compatibility and falls back to CPU.
    """
    import ctypes
    import sys

    try:
        driver = (ctypes.WinDLL("nvcuda.dll") if sys.platform == "win32"
                  else ctypes.CDLL("libcuda.so.1"))
        driver.cuInit.argtypes = [ctypes.c_uint]
        driver.cuInit.restype = ctypes.c_int
        driver.cuDeviceGetCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
        driver.cuDeviceGetCount.restype = ctypes.c_int
        if driver.cuInit(0) != 0:
            return False
        count = ctypes.c_int()
        return driver.cuDeviceGetCount(ctypes.byref(count)) == 0 and count.value > 0
    except (OSError, AttributeError):
        return False


@contextmanager
def hf_download_progress(status_cb, model_name: str):
    """Report Hugging Face download progress percentages via status_cb.

    All model files are fetched through huggingface_hub, whose byte progress
    runs through tqdm progress-bar classes. A reporting subclass is swapped
    into the three places a bar class is picked up from: the hub's default
    (used by onnx-asr's hf_hub_download), the snapshot-download binding, and
    faster-whisper's deliberately disabled class. The first byte-unit bar in
    flight is treated as the primary one so parallel bookkeeping bars (e.g.
    xet reconstruction) don't make the percentage jump around.
    """
    import importlib
    import io

    try:
        import huggingface_hub._snapshot_download as snap_mod

        # huggingface_hub.utils.tqdm the *module* is shadowed by the tqdm
        # *class* re-exported from utils/__init__ — importlib gets the module.
        tqdm_mod = importlib.import_module("huggingface_hub.utils.tqdm")
        xet_mod = importlib.import_module("huggingface_hub.utils._xet_progress_reporting")
        base = tqdm_mod.tqdm
    except Exception:
        yield
        return

    state = {"last": -1, "primary": None}

    class ReportingTqdm(base):
        def __init__(self, *args, **kwargs):
            # The app runs under pythonw (no console); render into a buffer.
            kwargs["file"] = io.StringIO()
            kwargs.pop("disable", None)
            super().__init__(*args, disable=False, **kwargs)
            if getattr(self, "unit", "") == "B" and state["primary"] is None:
                state["primary"] = self

        def update(self, n=1):
            result = super().update(n)
            try:
                if state["primary"] is self and self.total:
                    pct = min(100, int(self.n / self.total * 100))
                    if pct != state["last"]:
                        state["last"] = pct
                        if self.total >= 1e9:
                            size_text = f"{self.total / 1e9:.1f} GB"
                        else:
                            size_text = f"{self.total / 1e6:.0f} MB"
                        status_cb(f"Downloading model '{model_name}' — {pct}% of {size_text}…")
            except Exception:
                pass
            return result

        def close(self):
            if state["primary"] is self:
                state["primary"] = None  # let the next download bar report
            super().close()

    patches = [(tqdm_mod, "tqdm"), (snap_mod, "hf_tqdm"), (xet_mod, "tqdm")]
    try:
        import faster_whisper.utils as fw_utils

        patches.append((fw_utils, "disabled_tqdm"))
    except Exception:
        pass

    saved = []
    for mod, attr in patches:
        if hasattr(mod, attr):
            saved.append((mod, attr, getattr(mod, attr)))
            setattr(mod, attr, ReportingTqdm)
    try:
        yield
    finally:
        for mod, attr, value in saved:
            setattr(mod, attr, value)


def _get_model(model_name: str, device: str, status_cb) -> object:
    engine = engine_for(model_name)
    key = (engine, model_name, device)
    if _model_cache["key"] != key:
        free_asr_models()
        status_cb(f"Loading model '{model_name}' ({device}) — first use downloads it…")
        with hf_download_progress(status_cb, model_name):
            if engine == "qwen":
                repo = QWEN_REPOS.get(model_name.lower(), model_name)
                _model_cache["model"] = _QwenModel(repo, device)
            elif engine == "onnx":
                import onnx_asr

                providers = (
                    ["CUDAExecutionProvider", "CPUExecutionProvider"]
                    if device == "cuda" else ["CPUExecutionProvider"]
                )
                onnx_name = ONNX_ALIASES.get(model_name.lower(), model_name)
                model = onnx_asr.load_model(onnx_name, providers=providers)
                _model_cache["model"] = model.with_vad(onnx_asr.load_vad("silero"))
            else:
                from faster_whisper import WhisperModel

                import runtime

                compute_type = "float16" if device == "cuda" else "int8"
                whisper_name = WHISPER_ALIASES.get(model_name.lower(), model_name)
                # The standalone build ships one model inside the package; use
                # it directly instead of downloading.
                bundled = runtime.bundled_model_dir(model_name.lower())
                if bundled is not None:
                    whisper_name = str(bundled)
                _model_cache["model"] = WhisperModel(whisper_name, device=device,
                                                     compute_type=compute_type)
        _model_cache["key"] = key
    return _model_cache["model"]


class TranscribeWorker(QThread):
    model_status = Signal(str)
    track_started = Signal(int)
    segment_ready = Signal(int, object)  # (track index, Segment)
    track_progress = Signal(int, int)  # (track index, percent)
    track_done = Signal(int, str)  # (track index, detected language)
    track_failed = Signal(int, str)  # (track index, error message)
    track_note = Signal(int, str)  # (track index, what its timestamps ran into)
    finished_all = Signal(bool)  # cancelled?

    def __init__(self, tracks: list[Track], indices: list[int], model_name: str,
                 device: str, language: str | None, hotwords: str = "", parent=None):
        super().__init__(parent)
        self.tracks = tracks
        self.indices = indices
        self.model_name = model_name
        self.device = device
        self.language = language
        self.hotwords = hotwords.strip()
        self._stop = False

    def stop(self) -> None:
        self._stop = True

    def _load_model(self) -> object | None:
        """Load the model on the requested device, spilling over to CPU if the GPU fails."""
        try:
            return _get_model(self.model_name, self.device, self.model_status.emit)
        except Exception as exc:
            if self.device == "cpu":
                raise
            self.model_status.emit(f"GPU model load failed ({exc}) — falling back to CPU…")
            self.device = "cpu"
            return _get_model(self.model_name, self.device, self.model_status.emit)

    def _decode(self, i: int):
        """Track i's audio, 16 kHz mono on its file's own clock.

        Every engine decodes here, Whisper included: faster-whisper's own
        decoder ignores timestamps (see audio.py), so a gap or a late start
        in the file would move every word after it.
        """
        from audio import Timing, decode_track_audio

        track = self.tracks[i]
        timing = Timing()
        audio = decode_track_audio(track.path, track.stream_index, track.channel, timing)
        note = timing.summary()
        if note:
            self.track_note.emit(i, note)
        return audio

    def _transcribe_track(self, model, i: int) -> bool:
        """Transcribe one track. Returns True if cancelled mid-track."""
        engine = engine_for(self.model_name)
        if engine == "onnx":
            return self._transcribe_track_onnx(model, i)
        if engine == "qwen":
            return self._transcribe_track_qwen(model, i)
        track = self.tracks[i]
        self.track_started.emit(i)
        source = self._decode(i)
        if self._stop:
            return True
        segments, info = model.transcribe(
            source,
            language=self.language,
            beam_size=5,
            vad_filter=True,
            word_timestamps=True,
            condition_on_previous_text=False,
            hotwords=self.hotwords or None,
        )
        duration = info.duration or track.duration or 0.0
        for seg in segments:
            if self._stop:
                return True
            for piece in split_on_word_gaps(seg, track.speaker):
                self.segment_ready.emit(i, piece)
            if duration > 0:
                self.track_progress.emit(i, min(100, int(seg.end / duration * 100)))
        self.track_progress.emit(i, 100)
        self.track_done.emit(i, info.language or "")
        return False

    def _transcribe_track_qwen(self, model: "_QwenModel", i: int) -> bool:
        """Transcribe one track with Qwen3-ASR (qwen-asr package, CUDA).

        The audio is cut into speech regions with silero VAD and only those
        regions are fed to the model (in batches), each result offset by its
        region's known position. Qwen must never see long silences: on sparse
        audio it hallucinates repeated text into the gaps and the forced
        aligner then smears it across the silence, corrupting timestamps.
        Vocabulary hints go in as decoder context.
        """
        from faster_whisper.vad import VadOptions, get_speech_timestamps

        track = self.tracks[i]
        self.track_started.emit(i)
        audio = self._decode(i)
        if self._stop:
            return True

        # Speech regions: silences under 2 s stay inside a region (harmless),
        # longer ones become region boundaries; regions are capped below the
        # forced aligner's 5-minute window.
        regions = get_speech_timestamps(
            audio,
            vad_options=VadOptions(max_speech_duration_s=240, speech_pad_ms=400),
        )
        if not regions:
            self.track_progress.emit(i, 100)
            self.track_done.emit(i, "")
            return False

        # run() has already refused a language Qwen can't take, so a missing
        # entry here is a bug, not a cue to auto-detect.
        language = QWEN_LANGUAGE_NAMES[self.language] if self.language else None
        detected = language or ""

        # Auto-detecting every short VAD region independently is unreliable:
        # breaths and fillers such as "mm" can be labelled Chinese and then
        # transcribed as Chinese filler words. Detect once from the clearest
        # available sample and lock that language for the whole speaker track.
        if language is None:
            import numpy as np

            remaining = QWEN_LANGUAGE_PROBE_SECONDS * 16000
            probe_parts = []
            for region in sorted(
                regions, key=lambda r: r["end"] - r["start"], reverse=True
            ):
                if remaining <= 0:
                    break
                part = audio[region["start"]:region["end"]]
                if len(part) < 16000 and probe_parts:
                    continue
                part = part[:remaining]
                if len(part):
                    probe_parts.append(part)
                    remaining -= len(part)

            if probe_parts:
                self.model_status.emit("Qwen: detecting language from the speaker track...")
                probe_audio = np.concatenate(probe_parts)
                probe_results = model.model.transcribe(
                    (probe_audio, 16000),
                    context=self.hotwords or "",
                    language=None,
                    return_time_stamps=False,
                )
                if self._stop:
                    return True
                probe_language = (probe_results[0].language or "").split(",", 1)[0].strip()
                if probe_language in model.model.get_supported_languages():
                    language = probe_language
                    detected = probe_language
                    self.model_status.emit(f"Qwen: detected {probe_language}; transcribing...")

        batch_size = 8
        for start_idx in range(0, len(regions), batch_size):
            if self._stop:
                return True
            batch = regions[start_idx:start_idx + batch_size]
            inputs = [(audio[r["start"]:r["end"]], 16000) for r in batch]
            results = model.model.transcribe(
                inputs,
                context=self.hotwords or "",
                language=language,
                return_time_stamps=True,
            )
            if self._stop:
                return True
            # Qwen3ASRModel.transcribe is documented to return one result per
            # input, and zip() would silently drop regions if it ever didn't —
            # audio transcribed and then thrown away, with nothing to show for
            # it. Cheap to assert, invisible to miss.
            if len(results) != len(batch):
                raise RuntimeError(
                    f"Qwen returned {len(results)} result(s) for {len(batch)} "
                    f"audio region(s); refusing to drop the difference")
            for region, result in zip(batch, results):
                offset = region["start"] / 16000
                chunk_dur = (region["end"] - region["start"]) / 16000
                self._emit_qwen_segments(i, track.speaker, result, offset,
                                         chunk_dur, language)
                if not detected and result.language:
                    detected = result.language
            self.track_progress.emit(
                i, min(100, int(batch[-1]["end"] / len(audio) * 100)))

        self.track_progress.emit(i, 100)
        self.track_done.emit(i, detected)
        return False

    def _emit_qwen_segments(self, i: int, speaker: str, result, offset: float,
                            chunk_dur: float, language: str | None = None) -> None:
        """Turn one region's Qwen result into segments at the track position offset.

        `language` is the locked track language (user-picked or probed);
        segments in a script that language can't use are hallucinated noise
        and get dropped.
        """
        words = result.time_stamps or []
        text = (result.text or "").strip()
        if not words and text:
            if language and _foreign_script(text, language):
                return
            # Aligner produced nothing but there is text: keep it, placed on
            # the region rather than dropped.
            self.segment_ready.emit(i, Segment(speaker, offset, offset + chunk_dur, text))
            return
        if not words:
            return
        # Group words into segments at silence gaps; take each segment's
        # text as a slice of the punctuated transcript when possible.
        spans = _qwen_char_spans(text, words)
        groups = []
        start = 0
        for k in range(1, len(words)):
            if words[k].start_time - words[k - 1].end_time > SPLIT_GAP_SECONDS:
                groups.append((start, k - 1))
                start = k
        groups.append((start, len(words) - 1))
        for a, b in groups:
            if spans is not None:
                end_char = spans[b + 1][0] if b + 1 < len(words) else len(text)
                seg_text = text[spans[a][0]:end_char].strip()
            else:
                seg_text = " ".join(w.text for w in words[a:b + 1])
            if language and _foreign_script(seg_text, language):
                continue
            self.segment_ready.emit(i, Segment(
                speaker, offset + words[a].start_time,
                offset + words[b].end_time, seg_text))

    def _transcribe_track_onnx(self, model, i: int) -> bool:
        """Transcribe one track with an onnx-asr model (Parakeet etc.)."""
        track = self.tracks[i]
        self.track_started.emit(i)
        audio = self._decode(i)
        if self._stop:
            return True
        duration = len(audio) / 16000
        for res in model.recognize(audio, sample_rate=16000):
            if self._stop:
                return True
            if not res.text.strip():
                continue
            self.segment_ready.emit(i, Segment(track.speaker, res.start, res.end, res.text))
            if duration > 0:
                self.track_progress.emit(i, min(100, int(res.end / duration * 100)))
        self.track_progress.emit(i, 100)
        self.track_done.emit(i, "")
        return False

    def run(self) -> None:
        if self._stop:
            self.finished_all.emit(True)
            return
        # Before any model loads, and outside the GPU-to-CPU retry below: no
        # device can transcribe a language the model doesn't have.
        if not language_supported(self.model_name, self.language):
            for i in self.indices:
                self.track_failed.emit(
                    i, f"{self.model_name} can't transcribe language "
                       f"'{self.language}'. Choose another language or "
                       f"Auto-detect, or a Whisper model.")
            self.finished_all.emit(False)
            return
        try:
            model = self._load_model()
        except Exception as exc:  # model download/load failure on every device tried
            for i in self.indices:
                self.track_failed.emit(i, f"Model load failed: {exc}")
            self.finished_all.emit(False)
            return

        cancelled = False
        for i in self.indices:
            if self._stop:
                cancelled = True
                break
            try:
                cancelled = self._transcribe_track(model, i)
            except AudioFileError as exc:  # the file itself: the CPU would read it no better
                self.track_failed.emit(i, str(exc))
            except Exception as exc:
                if self._stop:
                    cancelled = True
                    break
                if self.device != "cpu":
                    # GPU trouble (e.g. out of VRAM) — spill this and later tracks to CPU.
                    self.model_status.emit(
                        f"GPU error on {self.tracks[i].path.name} ({exc}) — retrying on CPU…"
                    )
                    self.device = "cpu"
                    model = None  # release the local GPU reference before reloading
                    try:
                        model = _get_model(self.model_name, self.device, self.model_status.emit)
                        cancelled = self._transcribe_track(model, i)
                    except Exception as exc2:
                        self.track_failed.emit(i, str(exc2))
                else:
                    self.track_failed.emit(i, str(exc))

        self.finished_all.emit(cancelled)
