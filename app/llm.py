"""Local LLM that writes notes on a transcript (llama.cpp server on CUDA).

Inference runs through the official llama-server binary (in ../llama-cpp/)
with GGUF models — much faster on Windows than the previous
transformers+bitsandbytes backend. The app manages the server process:
loading a model starts it, free_llm() kills it, and the ASR models and the
notes LLM never hold VRAM at the same time. Generation streams
token-by-token into the UI via the `chunk` signal.
"""

from __future__ import annotations

import json
import socket
import subprocess
import time
from pathlib import Path

from PySide6.QtCore import QThread, Signal

import runtime

LLAMA_SERVER = runtime.bundle_dir() / "llama-cpp" / "llama-server.exe"

# Notes models: verified GGUF repos + exact filenames (Q4_K_M unless the
# name says otherwise). The box is
# editable; a typed value can be "repo_id" (first *Q4_K_M.gguf is used) or
# "repo_id::filename.gguf".
# Each "-q8" entry is the 8-bit (Q8_0) quant of the sibling above it: better
# quality than Q4_K_M at ~2x the download and VRAM. Q8 is only offered for the
# small/medium models — a Q8 of the 12-14B large-tier models is ~13-16 GB of
# weights and won't fit 16 GB VRAM alongside the context, so Q4 stays best there.
LLM_REPOS = {
    "qwen3-4b": ("unsloth/Qwen3-4B-Instruct-2507-GGUF", "Qwen3-4B-Instruct-2507-Q4_K_M.gguf"),
    "qwen3-4b-q8": ("unsloth/Qwen3-4B-Instruct-2507-GGUF", "Qwen3-4B-Instruct-2507-Q8_0.gguf"),
    "qwen3-8b": ("unsloth/Qwen3-8B-GGUF", "Qwen3-8B-Q4_K_M.gguf"),
    "qwen3-8b-q8": ("unsloth/Qwen3-8B-GGUF", "Qwen3-8B-Q8_0.gguf"),
    "qwen3-14b": ("unsloth/Qwen3-14B-GGUF", "Qwen3-14B-Q4_K_M.gguf"),
    "gemma-3-12b": ("unsloth/gemma-3-12b-it-GGUF", "gemma-3-12b-it-Q4_K_M.gguf"),
    "llama-3.1-8b": ("bartowski/Meta-Llama-3.1-8B-Instruct-GGUF", "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"),
    "llama-3.1-8b-q8": ("bartowski/Meta-Llama-3.1-8B-Instruct-GGUF", "Meta-Llama-3.1-8B-Instruct-Q8_0.gguf"),
    "llama-3.2-3b": ("unsloth/Llama-3.2-3B-Instruct-GGUF", "Llama-3.2-3B-Instruct-Q4_K_M.gguf"),
    "llama-3.2-3b-q8": ("unsloth/Llama-3.2-3B-Instruct-GGUF", "Llama-3.2-3B-Instruct-Q8_0.gguf"),
    # unsloth's Nemo GGUF is a pre-head-dim-fix conversion current llama.cpp
    # rejects (attn_q shape mismatch); bartowski's is a working conversion.
    "mistral-nemo-12b": ("bartowski/Mistral-Nemo-Instruct-2407-GGUF", "Mistral-Nemo-Instruct-2407-Q4_K_M.gguf"),
    "ministral-8b": ("bartowski/Ministral-8B-Instruct-2410-GGUF", "Ministral-8B-Instruct-2410-Q4_K_M.gguf"),
    "ministral-8b-q8": ("bartowski/Ministral-8B-Instruct-2410-GGUF", "Ministral-8B-Instruct-2410-Q8_0.gguf"),
    "phi-4-mini": ("unsloth/Phi-4-mini-instruct-GGUF", "Phi-4-mini-instruct-Q4_K_M.gguf"),
    "phi-4-mini-q8": ("unsloth/Phi-4-mini-instruct-GGUF", "Phi-4-mini-instruct.Q8_0.gguf"),
    "granite-3.3-2b": ("bartowski/ibm-granite_granite-3.3-2b-instruct-GGUF",
                       "ibm-granite_granite-3.3-2b-instruct-Q4_K_M.gguf"),
    "granite-3.3-2b-q8": ("bartowski/ibm-granite_granite-3.3-2b-instruct-GGUF",
                          "ibm-granite_granite-3.3-2b-instruct-Q8_0.gguf"),
}

LLM_GROUPS = [
    ("Max Detail (Large Tier)", [
        "qwen3-14b", "gemma-3-12b", "mistral-nemo-12b",
    ]),
    ("Standard Speed (Medium Tier)", [
        "qwen3-8b", "qwen3-8b-q8", "llama-3.1-8b", "llama-3.1-8b-q8",
        "ministral-8b", "ministral-8b-q8",
    ]),
    ("Quick Review (Small Tier)", [
        "qwen3-4b", "qwen3-4b-q8", "llama-3.2-3b", "llama-3.2-3b-q8",
        "phi-4-mini", "phi-4-mini-q8", "granite-3.3-2b", "granite-3.3-2b-q8",
    ]),
]
DEFAULT_LLM = "qwen3-4b"

# Qwen3-8B/14B are hybrid thinking models; notes need the answer, not the
# visible reasoning trace.
_NO_THINKING_MODELS = {"qwen3-8b", "qwen3-8b-q8", "qwen3-14b"}

# Transcripts longer than this many tokens are processed in parts, then the
# per-part notes are merged in a final pass (map-reduce).
MAX_INPUT_TOKENS = 20000
PART_TOKENS = 14000
MAX_NEW_TOKENS = 2500
CONTEXT_SIZE = MAX_INPUT_TOKENS + MAX_NEW_TOKENS + 500

# Starting guess for characters-per-token, used only to pick a first split
# point before this transcript's real ratio is known, and as the fallback if
# /tokenize can't be reached. It is never trusted on its own: it is about
# right for plain English prose and wrong for anything denser, and a part
# sized on a wrong ratio overflows the server's context *silently* —
# llama-server truncates the prompt rather than refusing it. Every prompt
# built here is verified against /tokenize before it is sent.
CHARS_PER_TOKEN = 3.5

# Held back inside MAX_INPUT_TOKENS for what /tokenize doesn't see: the chat
# template's role headers and control tokens wrapped around the prompt.
TEMPLATE_OVERHEAD_TOKENS = 256

# Floor under the prompt budget. Long user instructions ride along with every
# part and eat into MAX_INPUT_TOKENS; past this point the instructions are the
# problem, and shrinking parts further would just produce hundreds of them.
MIN_PROMPT_TOKENS = 2000

DEFAULT_INSTRUCTIONS = (
    "Write concise, well-organized Markdown notes on this conversation "
    "transcript: a short summary, key points and decisions, memorable "
    "moments or quotes with speaker attribution, and open questions or "
    "follow-ups. Use the speakers' names. Finish with a 'Per-person summary' "
    "section: for each speaker (and any character they play), a full-session "
    "summary of what they did, said, decided, and what happened to them."
)

_llm_cache: dict = {"key": None, "process": None, "port": None}

# The install folder may be read-only (Program Files); log somewhere writable.
_SERVER_LOG = (runtime.writable_dir() / "server.log" if runtime.IS_FROZEN
               else LLAMA_SERVER.parent / "server.log")


# One job object for the life of the process; every llama-server we spawn is
# assigned to it. Never closed on purpose — closing it is what kills the
# children, and that is exactly what should happen when this process ends.
_job_handle = None


def _kill_with_parent(process: subprocess.Popen) -> None:
    """Tie the server's lifetime to this process's (Windows job object).

    closeEvent stops the server on the way out, but only on the graceful
    path. A crash, an unhandled exception in the Qt event loop, or End Task
    from Task Manager leaves llama-server running and holding several GB of
    VRAM until someone notices — and the app's whole VRAM discipline (ASR
    models and the notes LLM never resident at the same time) assumes it
    dies with us.

    A job object with JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE makes Windows do it:
    when this process ends, however it ends, its handles close, the job's
    last handle goes with them, and everything in the job is terminated.

    Best effort — if any step fails the app is exactly as it was before,
    relying on the graceful path.
    """
    global _job_handle
    import ctypes
    from ctypes import wintypes

    class _BasicLimits(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_int64),
            ("PerJobUserTimeLimit", ctypes.c_int64),
            ("LimitFlags", wintypes.DWORD),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ctypes.c_size_t),
            ("PriorityClass", wintypes.DWORD),
            ("SchedulingClass", wintypes.DWORD),
        ]

    class _IoCounters(ctypes.Structure):
        _fields_ = [(name, ctypes.c_uint64) for name in (
            "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
            "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]

    class _ExtendedLimits(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", _BasicLimits),
            ("IoInfo", _IoCounters),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]

    KILL_ON_JOB_CLOSE = 0x2000
    EXTENDED_LIMIT_INFORMATION = 9  # JobObjectExtendedLimitInformation
    PROCESS_SET_QUOTA, PROCESS_TERMINATE = 0x0100, 0x0001

    try:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        # Handles are pointer-sized; without these the default c_int return
        # truncates them on 64-bit and every later call fails.
        kernel32.CreateJobObjectW.restype = wintypes.HANDLE
        kernel32.CreateJobObjectW.argtypes = [wintypes.LPVOID, wintypes.LPCWSTR]
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel32.SetInformationJobObject.argtypes = [
            wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID, wintypes.DWORD]
        kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]

        if _job_handle is None:
            handle = kernel32.CreateJobObjectW(None, None)
            if not handle:
                return
            limits = _ExtendedLimits()
            limits.BasicLimitInformation.LimitFlags = KILL_ON_JOB_CLOSE
            if not kernel32.SetInformationJobObject(
                    handle, EXTENDED_LIMIT_INFORMATION,
                    ctypes.byref(limits), ctypes.sizeof(limits)):
                kernel32.CloseHandle(handle)
                return
            _job_handle = handle

        child = kernel32.OpenProcess(
            PROCESS_SET_QUOTA | PROCESS_TERMINATE, False, process.pid)
        if not child:
            return
        try:
            kernel32.AssignProcessToJobObject(_job_handle, child)
        finally:
            kernel32.CloseHandle(child)
    except Exception:
        pass


def free_llm() -> None:
    """Stop the llama-server process, releasing its VRAM (C-level allocation
    — killing the process is the only reliable way to get it all back)."""
    process = _llm_cache["process"]
    _llm_cache["key"] = None
    _llm_cache["process"] = None
    _llm_cache["port"] = None
    if process is None:
        return
    try:
        process.terminate()
        process.wait(timeout=10)
    except Exception:
        try:
            process.kill()
        except Exception:
            pass


def _resolve_gguf(model_name: str) -> tuple[str, str | None]:
    name = model_name.lower().strip()
    if name in LLM_REPOS:
        return LLM_REPOS[name]
    raw = model_name.strip()
    if "::" in raw:
        repo, _, filename = raw.partition("::")
        return repo, filename
    return raw, None


def _find_free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _load(model_name: str, status_cb) -> int:
    """Ensure llama-server is running with the requested model; return its port."""
    import requests

    from transcriber import free_asr_models, hf_download_progress

    repo, filename = _resolve_gguf(model_name)
    key = (repo, filename)
    process = _llm_cache["process"]
    if _llm_cache["key"] == key and process is not None and process.poll() is None:
        return _llm_cache["port"]

    if not LLAMA_SERVER.exists():
        raise RuntimeError(f"llama-server.exe not found at {LLAMA_SERVER}")

    free_llm()  # switching models: stop the old server first
    free_asr_models()  # the LLM needs the VRAM the ASR model holds

    status_cb(f"Loading notes model '{model_name}' — first use downloads it…")
    with hf_download_progress(status_cb, model_name):
        from huggingface_hub import hf_hub_download, list_repo_files

        if filename is None:
            q4 = [f for f in list_repo_files(repo) if f.lower().endswith("q4_k_m.gguf")]
            if not q4:
                raise RuntimeError(f"No Q4_K_M .gguf found in '{repo}'. "
                                   "Use 'repo_id::filename.gguf' to pick a file.")
            filename = q4[0]
        model_path = hf_hub_download(repo_id=repo, filename=filename)

    status_cb(f"Starting notes engine with '{model_name}'…")
    port = _find_free_port()
    log = open(_SERVER_LOG, "w", encoding="utf-8", errors="replace")
    try:
        process = subprocess.Popen(
            [str(LLAMA_SERVER), "-m", model_path,
             "--host", "127.0.0.1", "--port", str(port),
             "-ngl", "999", "-c", str(CONTEXT_SIZE),
             "-fa", "on", "--cache-type-k", "q8_0", "--cache-type-v", "q8_0",
             "--no-webui"],
            stdout=log, stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
    finally:
        # The child got its own inherited copy of the handle; ours has no
        # further use and would otherwise leak one per model load.
        log.close()
    _kill_with_parent(process)
    deadline = time.monotonic() + 300
    while time.monotonic() < deadline:
        if process.poll() is not None:
            tail = ""
            try:
                tail = "\n".join(
                    _SERVER_LOG.read_text(encoding="utf-8", errors="replace").splitlines()[-8:])
            except Exception:
                pass
            raise RuntimeError(f"Notes engine failed to start:\n{tail}")
        try:
            if requests.get(f"http://127.0.0.1:{port}/health", timeout=2).status_code == 200:
                break
        except requests.RequestException:
            pass
        time.sleep(0.5)
    else:
        free_llm()
        raise RuntimeError("Notes engine did not become ready within 5 minutes.")

    _llm_cache["key"] = key
    _llm_cache["process"] = process
    _llm_cache["port"] = port
    return port


class NotesWorker(QThread):
    status = Signal(str)
    chunk = Signal(str)  # streamed notes text
    progress = Signal(int, int)  # (0-100 across all passes, tokens generated)
    completed = Signal()
    failed = Signal(str)

    def __init__(self, transcript: str, instructions: str = "",
                 model_name: str = DEFAULT_LLM, parent=None):
        super().__init__(parent)
        self.transcript = transcript
        # User instructions add to the standard format instead of replacing it.
        self.instructions = DEFAULT_INSTRUCTIONS
        if instructions.strip():
            self.instructions += f"\n\nAdditional instructions: {instructions.strip()}"
        self.model_name = model_name
        self._stop = False
        # Read by the window after `completed`: a cancelled run still emits
        # `completed`, and reporting it as "AI notes done" misdescribes what
        # the user is looking at.
        self.cancelled = False
        self._passes_total = 1
        self._passes_done = 0
        self._tokens_total = 0

    def stop(self) -> None:
        self._stop = True

    # ---------- generation ----------

    def _count_tokens(self, port: int, text: str) -> int:
        import requests

        response = requests.post(f"http://127.0.0.1:{port}/tokenize",
                                 json={"content": text}, timeout=120)
        response.raise_for_status()
        return len(response.json().get("tokens", []))

    def _generate(self, port: int, user_prompt: str, stream: bool) -> str:
        import requests

        messages = [
            {"role": "system",
             "content": "You take clear, structured notes on conversation transcripts."},
            {"role": "user", "content": user_prompt},
        ]
        payload = {
            "messages": messages,
            "max_tokens": MAX_NEW_TOKENS,
            "temperature": 0.7,
            "top_p": 0.8,
            "stream": True,
        }
        if self.model_name.lower().strip() in _NO_THINKING_MODELS:
            payload["chat_template_kwargs"] = {"enable_thinking": False}

        tokens = 0
        tokens_before_pass = self._tokens_total
        pieces: list[str] = []
        response = requests.post(
            f"http://127.0.0.1:{port}/v1/chat/completions",
            json=payload, stream=True, timeout=(30, 600))
        try:
            response.raise_for_status()
            # Decode ourselves: llama-server sends UTF-8 but no charset header,
            # and requests would otherwise fall back to Latin-1 (mojibake).
            for raw in response.iter_lines():
                if self._stop:
                    break
                line = raw.decode("utf-8", errors="replace") if raw else ""
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                delta = json.loads(data)["choices"][0].get("delta", {})
                text = delta.get("content") or ""
                if not text:
                    continue
                pieces.append(text)
                tokens += 1  # one SSE delta ≈ one token
                self._tokens_total = tokens_before_pass + tokens
                if stream:
                    self.chunk.emit(text)
                fraction = min(1.0, tokens / MAX_NEW_TOKENS)
                self.progress.emit(
                    min(99, int((self._passes_done + fraction) / self._passes_total * 100)),
                    self._tokens_total)
        finally:
            response.close()
        result = "".join(pieces)
        # SSE deltas are only an estimate; retokenize the finished pass so
        # the count retained beside the timer is exact.
        if result and not self._stop:
            try:
                self._tokens_total = tokens_before_pass + self._count_tokens(port, result)
            except Exception:
                pass
        self._passes_done += 1
        return result

    # ---------- fitting prompts to the context ----------

    def _try_count_tokens(self, port: int, text: str) -> int | None:
        """Real token count, or None when the server can't be asked.

        Sizing has to degrade rather than fail: one flaky /tokenize call
        shouldn't abort a notes run that may already be twenty minutes in.
        Callers fall back to the measured character ratio.
        """
        try:
            return self._count_tokens(port, text)
        except Exception:
            return None

    def _prompt_budget(self, port: int) -> int:
        """Tokens available for prompt *content* beside the instructions.

        MAX_INPUT_TOKENS covers the whole prompt, and the instructions ride
        along with every single part — they are user-editable and can be
        long, so they are measured rather than assumed small.
        """
        instructions = self._try_count_tokens(port, self.instructions)
        if instructions is None:
            instructions = int(len(self.instructions) / CHARS_PER_TOKEN)
        return max(MIN_PROMPT_TOKENS,
                   MAX_INPUT_TOKENS - instructions - TEMPLATE_OVERHEAD_TOKENS)

    def _chars_per_token(self, total_tokens: int) -> float:
        """This transcript's own character-to-token ratio.

        run() tokenizes the whole transcript before deciding anything, so the
        true ratio costs nothing extra — and it is the only honest basis for
        a character-sized split, since the ratio depends on the language, the
        speakers' names, and how many timestamps the text carries. Clamped so
        a nonsensical measurement can't produce gigantic parts.
        """
        if total_tokens <= 0 or not self.transcript:
            return CHARS_PER_TOKEN
        return max(1.0, min(len(self.transcript) / total_tokens, 8.0))

    def _pack(self, port: int, items: list[str], joiner: str, budget: int,
              chars_per_token: float, split_item=None) -> list[list[str]]:
        """Group consecutive items into batches that really fit `budget` tokens.

        The character ratio proposes a boundary and /tokenize confirms it: a
        measured ratio is still an average, and a prompt that lands over the
        budget is truncated by llama-server without complaint, so nothing
        here relies on the estimate alone. Groups are only ever trimmed, and
        whatever is trimmed off starts the next group. `split_item` cuts an
        item that is over budget on its own; without it such an item is
        passed through as its own group.
        """
        budget_chars = max(1, int(budget * chars_per_token))
        items = list(items)
        groups: list[list[str]] = []
        index = 0
        while index < len(items):
            take, chars = 1, len(items[index]) + len(joiner)
            while index + take < len(items):
                size = len(items[index + take]) + len(joiner)
                if chars + size > budget_chars:
                    break
                chars += size
                take += 1
            # Bounded: every pass either shrinks the group or shortens the
            # single item it is stuck on, so this cannot spin.
            for _ in range(10):
                tokens = self._try_count_tokens(
                    port, joiner.join(items[index:index + take]))
                if tokens is None or tokens <= budget:
                    break
                if take > 1:
                    take = max(1, min(take - 1, int(take * budget / tokens)))
                    continue
                if split_item is None:
                    break
                pieces = split_item(
                    items[index],
                    max(1, int(len(items[index]) * budget / tokens * 0.9)))
                if len(pieces) < 2:
                    break
                items[index:index + 1] = pieces
            groups.append(items[index:index + take])
            index += take
        return groups

    def _split_transcript(self, port: int, budget: int,
                          chars_per_token: float) -> list[str]:
        """Split the transcript into parts that fit `budget` tokens each.

        Cuts fall on line boundaries so a speaker's block stays whole
        (merger.MAX_COALESCE_CHARS exists to keep those blocks small enough
        for exactly this — see the note on that constant). A line longer than
        a whole part is cut down anyway rather than passed through: nothing
        here should depend on an invariant maintained in another module.
        """
        limit = max(1, int(budget * chars_per_token))
        lines = [piece for line in self.transcript.splitlines()
                 for piece in _split_long_line(line, limit)]
        groups = self._pack(port, lines, "\n", budget, chars_per_token,
                            split_item=_split_long_line)
        return ["\n".join(group) for group in groups]

    def _trim_to_budget(self, port: int, text: str, budget: int,
                        chars_per_token: float) -> str:
        """Last-resort cut so a prompt never exceeds what the server can read.

        Only reachable if one notes block is over budget by itself. Cutting
        visibly — and saying so in the output — beats letting llama-server
        drop part of the prompt without telling anyone.
        """
        tokens = self._try_count_tokens(port, text)
        if tokens is None:
            tokens = max(1, int(len(text) / chars_per_token))
        if tokens <= budget:
            return text
        keep = max(1, int(len(text) * budget / tokens * 0.9))
        return text[:keep] + "\n\n*(notes truncated to fit the model's context)*"

    # ---------- map-reduce over a long transcript ----------

    def _combine_prompt(self, combined: str) -> str:
        return (f"{self.instructions}\n\nBelow are notes taken on consecutive "
                f"parts of one long conversation. Merge them into a single "
                f"coherent set of notes, deduplicated and in order:\n\n{combined}")

    def _combine(self, port: int, part_notes: list[str], budget: int,
                 chars_per_token: float) -> None:
        """Merge the per-part notes, reducing in rounds when they don't fit.

        Each part contributes up to MAX_NEW_TOKENS of notes, so enough parts
        overflow the combining prompt just as surely as an oversized part
        overflows its own — nine full parts exceed MAX_INPUT_TOKENS between
        them. When that happens the notes are merged in groups that fit, and
        the group summaries are merged in turn, until one prompt holds
        everything.
        """
        blocks = [f"### Notes on part {k}\n{notes}"
                  for k, notes in enumerate(part_notes, 1)]
        while True:
            groups = self._pack(port, blocks, "\n\n", budget, chars_per_token)
            if len(groups) == 1:
                self.status.emit("Combining notes…")
                combined = self._trim_to_budget(
                    port, "\n\n".join(groups[0]), budget, chars_per_token)
                self._generate(port, self._combine_prompt(combined), stream=True)
                return
            if len(groups) >= len(blocks) > 1:
                # Every block is over budget on its own. Pair them up anyway:
                # summarizing is what brings the size down, and pairing
                # guarantees each round has strictly fewer blocks than the last.
                groups = [blocks[i:i + 2] for i in range(0, len(blocks), 2)]
            self._passes_total += len(groups)
            reduced = []
            for k, group in enumerate(groups, 1):
                if self._stop:
                    return
                self.status.emit(f"Combining notes — group {k} of {len(groups)}…")
                text = self._trim_to_budget(
                    port, "\n\n".join(group), budget, chars_per_token)
                reduced.append(self._generate(port, self._combine_prompt(text),
                                              stream=False))
            blocks = [f"### Notes on section {k}\n{notes}"
                      for k, notes in enumerate(reduced, 1)]

    def _emit_part_notes(self, part_notes: list[str], part_count: int) -> None:
        """Salvage the per-part notes when a map-reduce run is cancelled.

        Parts are generated with stream=False, so until the combining pass
        starts nothing at all has reached the editor. Without this, cancelling
        a five-part run during part five discards four finished summaries —
        twenty minutes of GPU time the user sat and waited for.
        """
        done = [notes.strip() for notes in part_notes if notes.strip()]
        if not done:
            return
        self.chunk.emit(
            f"\n\n**Partial — cancelled before the parts were combined.** "
            f"{len(done)} of {part_count} transcript part(s) were summarized; "
            f"their notes follow as written, unmerged.\n")
        for k, notes in enumerate(done, 1):
            self.chunk.emit(f"\n### Notes on part {k}\n\n{notes}\n")

    def run(self) -> None:
        try:
            port = _load(self.model_name, self.status.emit)
            total_tokens = self._try_count_tokens(port, self.transcript)
            if total_tokens is None:
                # /tokenize is out. Sizing falls back to the character
                # estimate everywhere below; generation itself may well be
                # fine, and failing the run outright would be worse.
                total_tokens = int(len(self.transcript) / CHARS_PER_TOKEN)
            budget = self._prompt_budget(port)

            if total_tokens <= budget:
                self.status.emit("Writing notes…")
                self._generate(
                    port,
                    f"{self.instructions}\n\nTranscript:\n\n{self.transcript}",
                    stream=True)
            else:
                # Long transcript: notes per part, then a combining pass.
                chars_per_token = self._chars_per_token(total_tokens)
                parts = self._split_transcript(
                    port, min(PART_TOKENS, budget), chars_per_token)
                self._passes_total = len(parts) + 1
                part_notes = []
                for k, part in enumerate(parts, 1):
                    if self._stop:
                        break
                    self.status.emit(f"Reading transcript — part {k} of {len(parts)}…")
                    notes = self._generate(
                        port,
                        f"{self.instructions}\n\nThis is part {k} of {len(parts)} "
                        f"of the transcript.\n\nTranscript part:\n\n{part}",
                        stream=False)
                    part_notes.append(notes)
                if not self._stop:
                    self._combine(port, part_notes, budget, chars_per_token)
                if self._stop:
                    # Covers a cancel during the parts (nothing emitted yet)
                    # and during the combining pass (a merge that stops
                    # partway through the session).
                    self._emit_part_notes(part_notes, len(parts))

            if self._stop:
                self.cancelled = True
                self.chunk.emit("\n\n*(cancelled)*")
            self.progress.emit(100, self._tokens_total)
            self.completed.emit()
        except Exception as exc:
            self.failed.emit(str(exc))


def _split_long_line(line: str, limit: int) -> list[str]:
    """Break one over-long line into pieces of at most `limit` characters.

    Cuts at a space in the back half of the piece when there is one, so words
    and sentences survive; falls back to a mid-word cut rather than returning
    a piece that is still too big.
    """
    if len(line) <= limit:
        return [line]
    pieces, start = [], 0
    while start < len(line):
        end = min(start + limit, len(line))
        if end < len(line):
            space = line.rfind(" ", start + limit // 2, end)
            if space > start:
                end = space
        pieces.append(line[start:end])
        start = end
        while start < len(line) and line[start] == " ":
            start += 1
    return [piece for piece in pieces if piece] or [line]
