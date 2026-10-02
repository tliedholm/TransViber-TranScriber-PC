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
import re
import socket
import subprocess
import time
from dataclasses import dataclass, replace
from pathlib import Path

from PySide6.QtCore import QThread, Signal

import runtime

LLAMA_SERVER = runtime.bundle_dir() / "llama-cpp" / "llama-server.exe"


@dataclass(frozen=True)
class NotesModel:
    """One notes model: where the weights are, and how to talk to it.

    Generation policy travels with the *resolved* model — repo, filename and
    revision — not with the name the user typed. A typed
    "repo_id::file.gguf" that names the same weights as a catalog alias has
    to get the same thinking/sampling treatment, or the two produce
    different notes and the comparison between them means nothing.
    """

    repo: str
    filename: str | None = None
    # Pinned so a comparison run months apart is the same weights. Only the
    # benchmark candidates are pinned; leaving the rest on main lets them
    # pick up re-converted GGUF fixes (the Nemo head-dim story below is
    # exactly that happening).
    revision: str | None = None
    # Chat-template switches, e.g. {"enable_thinking": False}. Passed to
    # llama-server as chat_template_kwargs; a key the template doesn't use is
    # ignored, so naming both spellings of one switch is safe.
    template_kwargs: dict | None = None
    # Harmony-style models (gpt-oss) budget reasoning instead of toggling it.
    reasoning_effort: str | None = None
    # Extra llama-server flags this model needs at launch. Some models reason
    # with no template switch to stop them, and the server's own flags are
    # then the only lever — a launch-time decision, not a per-request one.
    server_args: tuple[str, ...] | None = None
    # Context window to allocate, for a model measured to have room for more
    # than DEFAULT_CONTEXT_SIZE. A bigger window means fewer map-reduce parts
    # and so fewer lossy merges, and it is bought with KV cache — which costs
    # wildly different amounts depending on how many layers hold full
    # attention. Measured on this machine (RTX 5070 Ti, 16,303 MiB), whole
    # device at peak including a full-window prefill:
    #     lfm2.5-8b-a1b   23,000 -> 11,886 MiB   65,536 -> 12,275 MiB
    #     gpt-oss-20b     23,000 -> 14,544 MiB   40,960 -> 14,761 MiB
    #     gemma-4-12b Q6  23,000 -> 12,985 MiB   49,152 -> 13,249 MiB
    # **Per entry, never per pattern.** The headroom belongs to the quant,
    # not the model: gemma-4-12b at Q6_K has 3 GiB spare where the same model
    # at Q8_0 has almost none, so _IDENTITY_POLICIES must not set this.
    # An unmeasured model keeps the default — guessing it is how a notes run
    # turns into a CPU-offload crawl.
    context: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    repeat_penalty: float | None = None
    # Mixture-of-experts: how many expert layers to push to the CPU if the
    # model won't fit VRAM. 0 means dense — retry by dropping GPU layers.
    cpu_moe_layers: int = 0
    note: str = ""


# Notes models: verified GGUF repos + exact filenames. The box is editable; a
# typed value can be "repo_id" (first unsplit *Q4_K_M.gguf is used) or
# "repo_id::filename.gguf".
#
# One entry per role, at the best quant that comfortably fits. The old "large
# models must stay Q4" rule was wrong: at the context this app actually
# allocates (23,000 tokens, Q8_0 K/V caches, Flash Attention) a 12B at Q6_K
# budgets ~12-13 GiB on a 16 GB card. Only Q8_0 of a 12B is genuinely tight,
# which is why gemma-4-12b is Q6 and not Q8.
#
# The naming convention inverted when the duplicates were cut: a bare name is
# now the quant that entry ships, and a **-q4 suffix is the smaller fallback**
# of the same model — for a machine with less VRAM, or simply the file already
# in the cache. Fallbacks and superseded models stay resolvable by their short
# alias but are deliberately kept out of LLM_GROUPS; the dropdown is the thing
# that was too long, not the catalog.
LLM_REPOS: dict[str, NotesModel] = {
    # --- established models, one per role ----------------------------------
    "qwen3-4b": NotesModel(
        "unsloth/Qwen3-4B-Instruct-2507-GGUF", "Qwen3-4B-Instruct-2507-Q8_0.gguf",
        revision="a06e946bb6b655725eafa393f4a9745d460374c9",
        note="Instruct-2507: answers directly; no thinking mode to switch off. "
             "Startup default, and the notes model with the longest working "
             "history here — Qwen3.5 4B has to beat it before it replaces it."),
    "qwen3-4b-q4": NotesModel(
        "unsloth/Qwen3-4B-Instruct-2507-GGUF", "Qwen3-4B-Instruct-2507-Q4_K_M.gguf",
        revision="a06e946bb6b655725eafa393f4a9745d460374c9",
        note="Smaller fallback for qwen3-4b."),
    "qwen3-14b": NotesModel(
        "unsloth/Qwen3-14B-GGUF", "Qwen3-14B-Q5_K_M.gguf",
        revision="a04a82c4739b3ef5fa6da7d10261db2c67dd1985",
        note="Q5_K_M: 10.5 GB, fits at this context. Q6_K would be borderline. "
             "No newer 14B dense Qwen exists, so this stays as the large-tier "
             "control the challengers are measured against."),
    "qwen3-14b-q4": NotesModel(
        "unsloth/Qwen3-14B-GGUF", "Qwen3-14B-Q4_K_M.gguf",
        revision="a04a82c4739b3ef5fa6da7d10261db2c67dd1985",
        note="Smaller fallback for qwen3-14b."),
    # unsloth's Nemo GGUF is a pre-head-dim-fix conversion current llama.cpp
    # rejects (attn_q shape mismatch); bartowski's is a working conversion.
    "mistral-nemo-12b": NotesModel(
        "bartowski/Mistral-Nemo-Instruct-2407-GGUF",
        "Mistral-Nemo-Instruct-2407-Q6_K.gguf",
        note="Q6_K: 10.1 GB, the tightest of the promoted quants (~13 GiB "
             "budgeted). Direct-answer narrative style the Qwens don't have."),
    "mistral-nemo-12b-q4": NotesModel(
        "bartowski/Mistral-Nemo-Instruct-2407-GGUF",
        "Mistral-Nemo-Instruct-2407-Q4_K_M.gguf",
        note="Smaller fallback for mistral-nemo-12b."),
    "llama-3.1-8b": NotesModel("bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
                               "Meta-Llama-3.1-8B-Instruct-Q8_0.gguf"),
    "llama-3.1-8b-q4": NotesModel("bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
                                  "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
                                  note="Smaller fallback for llama-3.1-8b."),
    "llama-3.2-3b": NotesModel("unsloth/Llama-3.2-3B-Instruct-GGUF",
                               "Llama-3.2-3B-Instruct-Q8_0.gguf"),
    "llama-3.2-3b-q4": NotesModel("unsloth/Llama-3.2-3B-Instruct-GGUF",
                                  "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
                                  note="Smaller fallback for llama-3.2-3b."),
    "phi-4-mini": NotesModel("unsloth/Phi-4-mini-instruct-GGUF",
                             "Phi-4-mini-instruct.Q8_0.gguf"),
    "phi-4-mini-q4": NotesModel("unsloth/Phi-4-mini-instruct-GGUF",
                                "Phi-4-mini-instruct-Q4_K_M.gguf",
                                note="Smaller fallback for phi-4-mini."),

    # --- benchmark challengers ---------------------------------------------
    # Filenames and revisions verified against Hugging Face metadata on
    # 2026-09-20. Only gpt-oss-20b has actually been loaded on this machine;
    # the rest are here to be measured, not because they are known to be
    # better on this kind of transcript.
    "gemma-4-12b": NotesModel(
        "unsloth/gemma-4-12b-it-GGUF", "gemma-4-12b-it-Q6_K.gguf",
        revision="fc034cfff751157913579611efad8462ac1be606",
        note="9.8 GB. Q6 rather than Q8 on purpose — Q8 of a 12B is tight on "
             "16 GB here. Gemma terms apply upstream even through a community "
             "GGUF. Replaces gemma-3-12b if it wins."),
    "gemma-4-12b-qat": NotesModel(
        "google/gemma-4-12B-it-qat-q4_0-gguf", "gemma-4-12b-it-qat-q4_0.gguf",
        revision="29d097773436b69ff9feafd636ab4cf873786537",
        note="Google's own quantization-aware Q4_0; the paired test for gemma-4-12b."),
    "gemma-4-12b-q8": NotesModel(
        "unsloth/gemma-4-12b-it-GGUF", "gemma-4-12b-it-Q8_0.gguf",
        revision="fc034cfff751157913579611efad8462ac1be606",
        note="12.7 GB — tight on 16 GB at this context. A fidelity ceiling, "
             "not a default."),
    "qwen3.5-9b": NotesModel(
        "unsloth/Qwen3.5-9B-GGUF", "Qwen3.5-9B-Q8_0.gguf",
        revision="3885219b6810b007914f3a7950a8d1b469d598a5",
        note="9.5 GB. Replaces qwen3-8b if it wins."),
    "qwen3.5-9b-q6": NotesModel(
        "unsloth/Qwen3.5-9B-GGUF", "Qwen3.5-9B-Q6_K.gguf",
        revision="3885219b6810b007914f3a7950a8d1b469d598a5",
        note="Speed/size comparison for qwen3.5-9b."),
    "qwen3.5-4b": NotesModel(
        "unsloth/Qwen3.5-4B-GGUF", "Qwen3.5-4B-Q8_0.gguf",
        revision="e87f176479d0855a907a41277aca2f8ee7a09523",
        note="Has to beat qwen3-4b before it can become the default."),
    "gpt-oss-20b": NotesModel(
        "ggml-org/gpt-oss-20b-GGUF", "gpt-oss-20b-MXFP4.gguf",
        revision="ef9b12f2ff56c69cf32153a02784e7a3c88bf524",
        cpu_moe_layers=8,
        context=40960,
        note="Native MXFP4 experts — re-quantizing them to Q8/F16 wastes "
             "memory and recovers nothing. 12.1 GB; tight on 16 GB, but "
             "measured here at ~180 tok/s fully resident, no CPU offload. "
             "The tightest entry given a larger window: 14,761 MiB at peak "
             "leaves ~1.5 GiB, so this is the one to drop back to the "
             "default if anything else on the machine wants VRAM."),
    "lfm2.5-8b-a1b": NotesModel(
        "LiquidAI/LFM2.5-8B-A1B-GGUF", "LFM2.5-8B-A1B-Q8_0.gguf",
        revision="49c14831707011e64d70b2ebd8462ba08d608434",
        cpu_moe_layers=8,
        context=65536,
        note="LFM Open License v1.0, NOT Apache-2.0 — restricted above $10M "
             "revenue. Vendor suggests temperature 0.2 / repeat penalty 1.05; "
             "left on the app defaults until the two are compared. Reasons by "
             "default and has no template switch to stop it — see the lfm2 "
             "rule in _IDENTITY_POLICIES. Measured ~290 tok/s decode here."),
    "lfm2.5-8b-a1b-q6": NotesModel(
        "LiquidAI/LFM2.5-8B-A1B-GGUF", "LFM2.5-8B-A1B-Q6_K.gguf",
        revision="49c14831707011e64d70b2ebd8462ba08d608434",
        cpu_moe_layers=8,
        context=65536,
        note="LFM Open License v1.0, as above. Same reasoning behaviour. "
             "Q6_K is smaller than the Q8_0 measured at this window, so it "
             "has strictly more room."),
    "qwen3.8-27b": NotesModel(
        "unsloth/Qwen3.8-27B-GGUF", "Qwen3.8-27B-UD-IQ3_S.gguf",
        revision="4ca720788d1e01f1bff70c033e0d0028fd02e502",
        note="Capacity experiment at an aggressive quant on an untested newer "
             "checkpoint. Expect to need a newer llama.cpp than the bundled one."),
    "ministral-3-8b": NotesModel(
        "mistralai/Ministral-3-8B-Instruct-2512-GGUF",
        "Ministral-3-8B-Instruct-2512-Q8_0.gguf",
        note="Replaces ministral-8b: Apache-2.0 instead of the Mistral "
             "Research License, which was non-commercial."),
    "granite-4.2-3b": NotesModel("ibm-granite/granite-4.2-3b-GGUF",
                                 "granite-4.2-3b-Q8_0.gguf",
                                 note="Replaces granite-3.3-2b."),

    # --- superseded and reserve: resolvable, kept out of the dropdown ------
    # Not deleted. Each is a working model whose replacement above has never
    # been loaded on this machine, so these are the fallback if one of the
    # challengers turns out not to load, not dead weight.
    "gemma-3-12b": NotesModel("unsloth/gemma-3-12b-it-GGUF",
                              "gemma-3-12b-it-Q6_K.gguf",
                              note="Superseded by gemma-4-12b. Gemma terms."),
    "gemma-3-12b-q4": NotesModel("unsloth/gemma-3-12b-it-GGUF",
                                 "gemma-3-12b-it-Q4_K_M.gguf",
                                 note="The gemma-3 build with a working local "
                                      "run behind it (~3,465 tok/s prefill, "
                                      "67 tok/s decode)."),
    "qwen3-8b": NotesModel("unsloth/Qwen3-8B-GGUF", "Qwen3-8B-Q8_0.gguf",
                           note="Superseded by qwen3.5-9b."),
    "qwen3-8b-q4": NotesModel("unsloth/Qwen3-8B-GGUF", "Qwen3-8B-Q4_K_M.gguf",
                              note="Smaller fallback for qwen3-8b."),
    "ministral-8b": NotesModel("bartowski/Ministral-8B-Instruct-2410-GGUF",
                               "Ministral-8B-Instruct-2410-Q8_0.gguf",
                               note="Superseded by ministral-3-8b. Mistral "
                                    "Research License: personal use only."),
    "ministral-8b-q4": NotesModel("bartowski/Ministral-8B-Instruct-2410-GGUF",
                                  "Ministral-8B-Instruct-2410-Q4_K_M.gguf",
                                  note="Smaller fallback for ministral-8b."),
    "granite-3.3-2b": NotesModel("bartowski/ibm-granite_granite-3.3-2b-instruct-GGUF",
                                 "ibm-granite_granite-3.3-2b-instruct-Q8_0.gguf",
                                 note="Superseded by granite-4.2-3b."),
    "granite-3.3-2b-q4": NotesModel("bartowski/ibm-granite_granite-3.3-2b-instruct-GGUF",
                                    "ibm-granite_granite-3.3-2b-instruct-Q4_K_M.gguf",
                                    note="Smaller fallback for granite-3.3-2b."),
    "ministral-3-14b": NotesModel("mistralai/Ministral-3-14B-Instruct-2512-GGUF",
                                  "Ministral-3-14B-Instruct-2512-Q5_K_M.gguf"),
    "granite-4.2-8b": NotesModel("ibm-granite/granite-4.2-8b-GGUF",
                                 "granite-4.2-8b-Q8_0.gguf"),
}

LLM_GROUPS = [
    ("Max Detail (Large Tier)", [
        "qwen3-14b", "mistral-nemo-12b",
    ]),
    ("Standard Speed (Medium Tier)", [
        "llama-3.1-8b",
    ]),
    ("Quick Review (Small Tier)", [
        "qwen3-4b", "llama-3.2-3b", "phi-4-mini",
    ]),
    ("Untested — Benchmark Candidates", [
        "gemma-4-12b", "qwen3.5-9b", "qwen3.5-4b", "gpt-oss-20b",
        "lfm2.5-8b-a1b", "qwen3.8-27b", "ministral-3-8b", "granite-4.2-3b",
    ]),
]
DEFAULT_LLM = "qwen3-4b"

# Generation policy by model *identity*, matched against "repo/filename"
# lowercased, first match wins. This is what makes a typed
# "repo::file.gguf" behave like the catalog alias for the same weights, and
# what gives a model the app has never seen a sane default when its family
# is recognizable. A catalog entry's own fields always win over these.
#
# Hybrid-thinking models are switched off rather than left to default: notes
# need the answer, and a visible reasoning trace both spends the completion
# budget and lands in the user's notes.
_IDENTITY_POLICIES: tuple[tuple[str, dict], ...] = (
    # gpt-oss speaks Harmony: reasoning is budgeted, never switched off.
    # "low" is not "off" — it still spends completion tokens, which is what
    # REASONING_EXTRA_TOKENS below pays for.
    (r"gpt-?oss", {"reasoning_effort": "low"}),
    # LFM2.5 reasons by default and its template offers no way to stop it:
    # `preserve_thinking` only decides whether *history* keeps its traces,
    # and the generation prompt is a bare "<|im_start|>assistant\n". So this
    # is the one model whose thinking has to be settled at the server.
    # Measured here on LFM2.5-8B-A1B Q8_0 at temperature 0, same prompt:
    #   default                        609 tokens ->   584 chars of notes
    #   --reasoning off                609 tokens ->   584 chars  (identical)
    #   chat_template_kwargs           609 tokens ->   584 chars  (identical)
    #   --reasoning-budget 0           286 tokens -> 1,113 chars
    # Only the budget flag works; the other two are silent no-ops. It ends
    # the trace by emitting an empty <think></think> into the content, which
    # _LeadingThinkFilter below strips.
    (r"lfm2", {"server_args": ("--reasoning-budget", "0")}),
    # Qwen3-*-Instruct-* answer directly and have no switch to set; this has
    # to be tested before the general Qwen3 rule below.
    (r"qwen3[.\d]*-\d+b-instruct", {}),
    (r"qwen3[.\d]*-", {"template_kwargs": {"enable_thinking": False}}),
    # Gemma 4's template already defaults enable_thinking to false, so this
    # is insurance rather than a fix — but the default belongs to whoever
    # converted the GGUF, and the rule above is to switch thinking off
    # rather than inherit it.
    (r"gemma-4", {"template_kwargs": {"enable_thinking": False}}),
    # Granite 4's template spells the switch "thinking"; older Qwen-style
    # "enable_thinking" is harmless alongside it and covers either template.
    (r"granite-4", {"template_kwargs": {"enable_thinking": False,
                                        "thinking": False}}),
    (r"ministral-3", {"temperature": 0.1}),
)

# Transcripts longer than the prompt budget are processed in parts, then the
# per-part notes are merged in a final pass (map-reduce).
#
# The *window* is per-model (NotesModel.context); everything carved out of it
# is the same for every model. 23,000 is what fits the tightest entry in the
# catalog, so it is the floor rather than the ambition: a model with measured
# headroom asks for more and takes fewer, larger parts, which means fewer
# lossy merges. Raising this default instead would hand the same bill to the
# dense models, whose every layer holds full attention and whose KV cache
# therefore grows several times faster than the MoE and sliding-window
# entries — see the table on NotesModel.context.
MAX_NEW_TOKENS = 2500
DEFAULT_CONTEXT_SIZE = 23000
# Slack between the prompt budget and the window, absorbing the difference
# between what /tokenize counts and what the server ends up holding.
CONTEXT_HEADROOM = 500
# One part of a map-reduce run, as a share of the prompt budget. The rest is
# room for the instructions, which ride along with every single part.
PART_FRACTION = 0.7


def _input_tokens(context: int) -> int:
    """Prompt budget inside a window of `context` tokens."""
    return context - MAX_NEW_TOKENS - CONTEXT_HEADROOM


def _part_tokens(context: int) -> int:
    """How much transcript goes in one part of a map-reduce run."""
    return int(_input_tokens(context) * PART_FRACTION)


# The default model's budgets. Kept as names because they are what the tests
# and the run report talk about; a model with its own context derives its own.
CONTEXT_SIZE = DEFAULT_CONTEXT_SIZE
MAX_INPUT_TOKENS = _input_tokens(CONTEXT_SIZE)
PART_TOKENS = _part_tokens(CONTEXT_SIZE)

# Extra completion tokens granted to a model that reasons before answering
# (gpt-oss at any reasoning_effort). Its hidden reasoning is discarded by
# this app but still spends the completion allowance, so a flat 2,500-token
# cap can leave a hard transcript with almost no visible notes. Taken out of
# the prompt budget rather than added to the context, so CONTEXT_SIZE — and
# therefore the KV cache the server allocates — does not change.
REASONING_EXTRA_TOKENS = 1500

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

_llm_cache: dict = {"request": None, "process": None, "port": None,
                    "model": None}

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
    _llm_cache["request"] = None
    _llm_cache["process"] = None
    _llm_cache["port"] = None
    _llm_cache["model"] = None
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


def _resolve_model(model_name: str) -> NotesModel:
    """Turn what the user typed into a model: alias, repo::file, or bare repo."""
    name = model_name.lower().strip()
    if name in LLM_REPOS:
        return LLM_REPOS[name]
    raw = model_name.strip()
    if "::" in raw:
        repo, _, filename = raw.partition("::")
        return NotesModel(repo.strip(), filename.strip() or None)
    return NotesModel(raw)


def _apply_identity_policy(model: NotesModel) -> NotesModel:
    """Fill in unset generation policy from the resolved repo and filename.

    The catalog entry always wins; this only supplies what it left unset.
    Keying off the resolved identity rather than the typed alias is the
    point: "unsloth/Qwen3-14B-GGUF::Qwen3-14B-Q5_K_M.gguf" is the same model
    as the qwen3-14b entry and has to get the same thinking policy, or a
    quant comparison silently compares thinking against non-thinking.
    """
    identity = f"{model.repo}/{model.filename or ''}".lower()
    for pattern, policy in _IDENTITY_POLICIES:
        if re.search(pattern, identity):
            unset = {field: value for field, value in policy.items()
                     if getattr(model, field) is None}
            return replace(model, **unset) if unset else model
    return model


# "Model-00001-of-00003.gguf" and friends. llama.cpp loads such a set from
# the first shard and expects the rest beside it, so all of them have to come
# down — hf_hub_download fetches exactly one file per call, which used to
# mean a split model downloaded a third of itself and then failed to load.
_SHARD_RE = re.compile(r"^(?P<stem>.+)-(?P<index>\d{5})-of-(?P<total>\d{5})\.gguf$",
                       re.IGNORECASE)


def _shard_set(filename: str) -> list[str]:
    """Every file in a split GGUF set, in order, or [filename] if unsplit."""
    match = _SHARD_RE.match(Path(filename).name)
    if not match:
        return [filename]
    prefix = filename[:len(filename) - len(Path(filename).name)]
    stem, total = match.group("stem"), int(match.group("total"))
    return [f"{prefix}{stem}-{n:05d}-of-{total:05d}.gguf"
            for n in range(1, total + 1)]


def _pick_repo_file(repo: str, revision: str | None) -> str:
    """Choose a file from a bare repo id: the Q4_K_M build, unsplit if it exists."""
    from huggingface_hub import list_repo_files

    files = list_repo_files(repo, revision=revision)
    unsplit = [f for f in files if f.lower().endswith("q4_k_m.gguf")]
    if unsplit:
        return unsplit[0]
    # A split Q4_K_M carries the shard suffix after the quant name, so the
    # endswith test above cannot see it. Naming the first shard is enough —
    # _shard_set expands it back to the whole set.
    first_shards = []
    for f in files:
        match = _SHARD_RE.match(Path(f).name)
        if match and match.group("index") == "00001" and "q4_k_m" in f.lower():
            first_shards.append(f)
    if not first_shards:
        raise RuntimeError(f"No Q4_K_M .gguf found in '{repo}'. "
                           "Use 'repo_id::filename.gguf' to pick a file.")
    return first_shards[0]


def _download_gguf(model: NotesModel, status_cb) -> tuple[Path, NotesModel]:
    """Fetch the weights; return the file to load and the fully resolved model."""
    from transcriber import hf_download_progress

    with hf_download_progress(status_cb, model.repo):
        from huggingface_hub import hf_hub_download

        filename = model.filename or _pick_repo_file(model.repo, model.revision)
        model = _apply_identity_policy(replace(model, filename=filename))
        paths = [hf_hub_download(repo_id=model.repo, filename=shard,
                                 revision=model.revision)
                 for shard in _shard_set(filename)]
    return Path(paths[0]), model


# Launch policy. Named constants rather than inline strings because these are
# what a memory failure gets diagnosed against, and because the fallback
# ladder below has to be able to override them.
GPU_LAYERS = 999    # -ngl: put everything on the GPU
SERVER_SLOTS = 1    # one slot: a run should not depend on the server's
                    # automatic slot count, which differs between builds


def _server_command(model_path: Path, port: int, extra: list[str],
                    context: int = DEFAULT_CONTEXT_SIZE) -> list[str]:
    return [str(LLAMA_SERVER), "-m", str(model_path),
            "--host", "127.0.0.1", "--port", str(port),
            "-ngl", str(GPU_LAYERS), "-c", str(context),
            "--parallel", str(SERVER_SLOTS),
            "-fa", "on", "--cache-type-k", "q8_0", "--cache-type-v", "q8_0",
            "--no-webui", *extra]


def _log_text() -> str:
    try:
        return _SERVER_LOG.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _log_tail(lines: int = 12) -> str:
    return "\n".join(_log_text().splitlines()[-lines:])


_OUT_OF_MEMORY = ("out of memory", "failed to allocate", "cudamalloc",
                  "unable to allocate", "insufficient memory")


def _looks_like_oom(log: str) -> bool:
    lowered = log.lower()
    return any(marker in lowered for marker in _OUT_OF_MEMORY)


def _fallback_ladder(model: NotesModel) -> list[tuple[list[str], str]]:
    """What to retry, in order, when the model won't fit as launched.

    `-ngl 999` is a request, not a guarantee, and the server's own --fit
    logic only adjusts arguments that were left *unset* — which -ngl and -c
    are not. Without this a too-large model just fails the notes run with a
    CUDA allocation error and the user has no way to ask for less. A
    mixture-of-experts model gives up its experts first: they are the bulk of
    the weights and only a fraction of them is touched per token. A dense
    model can only shed whole layers, which costs far more speed.
    """
    if model.cpu_moe_layers:
        return [
            (["--n-cpu-moe", str(model.cpu_moe_layers)],
             f"moving {model.cpu_moe_layers} expert layers to the CPU"),
            (["--cpu-moe"], "moving every expert layer to the CPU"),
        ]
    return [(["-ngl", "24"], "keeping 24 layers on the GPU"),
            (["-ngl", "12"], "keeping 12 layers on the GPU")]


def _launch_attempts(model: NotesModel) -> list[tuple[list[str], str]]:
    """Every launch to try, in order, with what to say about each.

    The model's own flags belong to the weights, not to one attempt, so they
    ride along every rung of the fallback ladder. Dropping them on a retry
    would mean a model that only just failed to fit came back up reasoning
    again — the notes would be short for a different reason each time.
    """
    base = list(model.server_args or ())
    return [(base, "")] + [(base + extra, description)
                           for extra, description in _fallback_ladder(model)]


def _runtime_report(log: str) -> str:
    """One line describing what the server actually allocated.

    Reading it back costs nothing and is the only record of whether the model
    really stayed resident: llama.cpp reports a partial offload as an
    ordinary progress line, so a model quietly running half on the CPU looks
    exactly like one that fit — until the notes take twenty minutes.
    """
    bits: list[str] = []
    version = re.search(r"^version: (\S+ \(\S+\))", log, re.MULTILINE)
    if version:
        bits.append(f"llama.cpp {version.group(1)}")
    offload = re.search(r"offloaded (\d+)/(\d+) layers to GPU", log)
    if offload:
        on_gpu, total = offload.groups()
        bits.append(f"{on_gpu}/{total} layers on GPU"
                    + ("" if on_gpu == total else " — PARTIAL, expect it to crawl"))
    for label, pattern in (("weights", r"model buffer size *= *([\d.]+) MiB"),
                           ("KV", r"KV buffer size *= *([\d.]+) MiB"),
                           ("compute", r"compute buffer size *= *([\d.]+) MiB")):
        sizes = [float(value) for value in re.findall(pattern, log)]
        if sizes:
            bits.append(f"{label} {sum(sizes) / 1024:.2f} GiB")
    return "; ".join(bits)


def _find_free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _spawn_server(model_path: Path, extra: list[str],
                  context: int = DEFAULT_CONTEXT_SIZE) -> tuple[subprocess.Popen, int]:
    port = _find_free_port()
    log = open(_SERVER_LOG, "w", encoding="utf-8", errors="replace")
    try:
        process = subprocess.Popen(
            _server_command(model_path, port, extra, context),
            stdout=log, stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
    finally:
        # The child got its own inherited copy of the handle; ours has no
        # further use and would otherwise leak one per model load.
        log.close()
    _kill_with_parent(process)
    return process, port


class _NotesCancelled(Exception):
    """Cooperative cancellation while downloading/starting the notes engine."""


def _check_cancelled(cancelled) -> None:
    if cancelled is not None and cancelled():
        raise _NotesCancelled()


def _await_ready(process: subprocess.Popen, port: int, timeout: float = 300,
                 cancelled=None) -> bool:
    """Wait for /health; False if the server died on the way up."""
    import requests

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            _check_cancelled(cancelled)
        except _NotesCancelled:
            process.kill()
            process.wait()
            raise
        if process.poll() is not None:
            return False
        try:
            if requests.get(f"http://127.0.0.1:{port}/health",
                            timeout=2).status_code == 200:
                return True
        except requests.RequestException:
            pass
        time.sleep(0.5)
    try:
        process.kill()
    except Exception:
        pass
    raise RuntimeError("Notes engine did not become ready within 5 minutes.")


def _load(model_name: str, status_cb, cancelled=None) -> tuple[int, NotesModel]:
    """Ensure llama-server is running with the requested model.

    Returns its port and the *resolved* model — repo, filename, revision and
    the generation policy that belongs to those weights. Callers generate
    against that, never against the name they passed in.
    """
    from transcriber import free_asr_models

    _check_cancelled(cancelled)

    # Cached on the typed text: the resolved filename of a bare repo id isn't
    # known without asking Hugging Face, and restarting a loaded server to
    # find out would cost far more than the occasional duplicate load.
    request = model_name.lower().strip()
    process = _llm_cache["process"]
    if (request == _llm_cache["request"]
            and process is not None and process.poll() is None):
        return _llm_cache["port"], _llm_cache["model"]

    if not LLAMA_SERVER.exists():
        raise RuntimeError(f"llama-server.exe not found at {LLAMA_SERVER}")

    free_llm()  # switching models: stop the old server first
    free_asr_models()  # the LLM needs the VRAM the ASR model holds

    status_cb(f"Loading notes model '{model_name}' — first use downloads it…")
    model = _apply_identity_policy(_resolve_model(model_name))
    model_path, model = _download_gguf(model, status_cb)
    _check_cancelled(cancelled)

    context = model.context or DEFAULT_CONTEXT_SIZE
    status_cb(f"Starting notes engine with '{model_name}' at {context:,} tokens…")
    attempts = _launch_attempts(model)
    for number, (extra, description) in enumerate(attempts):
        _check_cancelled(cancelled)
        if description:
            status_cb(f"Not enough VRAM — retrying '{model_name}', {description}…")
        process, port = _spawn_server(model_path, extra, context)
        if _await_ready(process, port, cancelled=cancelled):
            break
        log = _log_text()
        if number == len(attempts) - 1 or not _looks_like_oom(log):
            raise RuntimeError(f"Notes engine failed to start:\n{_log_tail()}")

    report = _runtime_report(_log_text())
    if report:
        status_cb(f"Notes engine ready — {report}")

    _llm_cache["request"] = request
    _llm_cache["process"] = process
    _llm_cache["port"] = port
    _llm_cache["model"] = model
    return port, model


_EMPTY_THINK = "<think></think>"


class _LeadingThinkFilter:
    """Drop an empty ``<think></think>`` from the start of a content stream.

    A model held to ``--reasoning-budget 0`` obeys by opening and closing the
    trace at once, and an *empty* trace is not extracted into
    ``reasoning_content`` the way a real one is — it arrives as ordinary
    content, split across deltas ("\\n", "<think>", "</think>"). Left alone
    it opens every set of notes with a stray tag.

    Only a leading and genuinely empty block is dropped. A block with
    anything in it means the server is not extracting traces at all, and
    hiding that would turn a misconfiguration into a mystery.
    """

    def __init__(self) -> None:
        self._buffer = ""
        # "seeking": still deciding. "stripping": tag dropped, now eating the
        # newlines behind it, which arrive as their own deltas. "passing":
        # out of the way for the rest of the stream.
        self._state = "seeking"

    def feed(self, text: str) -> str:
        if self._state == "passing":
            return text
        if self._state == "stripping":
            text = text.lstrip("\n")
            if text:
                self._state = "passing"
            return text
        self._buffer += text
        compact = "".join(self._buffer.split())
        if compact.startswith(_EMPTY_THINK):
            end = self._buffer.index("</think>") + len("</think>")
            held, self._buffer = self._buffer[end:], ""
            held = held.lstrip("\n")
            self._state = "passing" if held else "stripping"
            return held
        if _EMPTY_THINK.startswith(compact):
            return ""  # still might turn into one; hold it back a delta
        self._state = "passing"
        held, self._buffer = self._buffer, ""
        return held

    def flush(self) -> str:
        """Whatever is still held when the stream ends."""
        self._state = "passing"
        held, self._buffer = self._buffer, ""
        return held


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
        # Filled in by run() from _load: the resolved weights and the
        # generation policy that goes with them, never the typed name.
        self.model: NotesModel | None = None
        self._stop = False
        # Read by the window after `completed`: a cancelled run still emits
        # `completed`, and reporting it as "AI notes done" misdescribes what
        # the user is looking at.
        self.cancelled = False
        self._passes_total = 1
        self._passes_done = 0
        self._tokens_total = 0
        # Reasoning tokens the model spent and this app threw away, and the
        # passes that hit the completion cap. Both are invisible otherwise:
        # notes that stop mid-sentence look like the model had nothing more
        # to say.
        self._reasoning_tokens = 0
        self._truncated_passes = 0
        # Prompt length as the *server* counted it, template and all. The
        # app's own sizing only ever sees the text.
        self._last_prompt_tokens = 0
        self._peak_prompt_tokens = 0

    def stop(self) -> None:
        self._stop = True

    # ---------- per-model budgets ----------

    def _reasoning_budget(self) -> int:
        """Completion tokens set aside for reasoning this app never shows."""
        if self.model is not None and self.model.reasoning_effort:
            return REASONING_EXTRA_TOKENS
        return 0

    def _max_new_tokens(self) -> int:
        return MAX_NEW_TOKENS + self._reasoning_budget()

    def _context_size(self) -> int:
        """The window the server was started with, for this model.

        Read from the resolved model rather than a constant: run() sets
        self.model from _load, which is the same object _load sized the
        server's `-c` from, so the two cannot drift apart.
        """
        if self.model is not None and self.model.context:
            return self.model.context
        return DEFAULT_CONTEXT_SIZE

    def _max_input_tokens(self) -> int:
        return _input_tokens(self._context_size())

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
        model = self.model or NotesModel("")
        payload = {
            "messages": messages,
            "max_tokens": self._max_new_tokens(),
            "temperature": 0.7 if model.temperature is None else model.temperature,
            "top_p": 0.8 if model.top_p is None else model.top_p,
            "stream": True,
            # The prompt this app measures is the text; what the server
            # reads is that text wrapped in the model's chat template. Asking
            # for usage is the only way to see the difference, and the
            # difference is what TEMPLATE_OVERHEAD_TOKENS is guessing at.
            "stream_options": {"include_usage": True},
        }
        if model.repeat_penalty is not None:
            payload["repeat_penalty"] = model.repeat_penalty
        template_kwargs = dict(model.template_kwargs or {})
        if model.reasoning_effort:
            # Harmony models take the effort through the chat template; the
            # top-level field is what the OpenAI-shaped API calls it. Sending
            # both covers either spelling, and llama-server ignores a field
            # it doesn't know.
            template_kwargs["reasoning_effort"] = model.reasoning_effort
            payload["reasoning_effort"] = model.reasoning_effort
        if template_kwargs:
            payload["chat_template_kwargs"] = template_kwargs

        tokens = 0
        reasoning_tokens = 0
        finish_reason = None
        tokens_before_pass = self._tokens_total
        max_new = self._max_new_tokens()
        pieces: list[str] = []
        think_filter = _LeadingThinkFilter()
        try:
            response = requests.post(
                f"http://127.0.0.1:{port}/v1/chat/completions",
                json=payload, stream=True, timeout=(30, 600))
        except requests.RequestException:
            if self._stop:
                return ""
            raise
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
                event = json.loads(data)
                usage = event.get("usage") or {}
                if usage.get("prompt_tokens"):
                    self._last_prompt_tokens = usage["prompt_tokens"]
                if not event.get("choices"):
                    continue  # the usage-only event that ends the stream
                choice = event["choices"][0]
                finish_reason = choice.get("finish_reason") or finish_reason
                delta = choice.get("delta", {})
                # A reasoning model's hidden trace is discarded here, but it
                # was still generated and still spent the completion budget —
                # count it so a short set of notes can be explained.
                if delta.get("reasoning_content"):
                    reasoning_tokens += 1
                text = delta.get("content") or ""
                if not text:
                    continue
                tokens += 1  # one SSE delta ≈ one token
                self._tokens_total = tokens_before_pass + tokens
                # Counted before filtering: a delta the user never sees was
                # still generated and still spent the completion budget.
                visible = think_filter.feed(text)
                if visible:
                    pieces.append(visible)
                    if stream:
                        self.chunk.emit(visible)
                fraction = min(1.0, (tokens + reasoning_tokens) / max_new)
                self.progress.emit(
                    min(99, int((self._passes_done + fraction) / self._passes_total * 100)),
                    self._tokens_total)
        except requests.RequestException:
            if not self._stop:
                raise
        finally:
            response.close()
        held = think_filter.flush()
        if held:
            pieces.append(held)
            if stream:
                self.chunk.emit(held)
        result = "".join(pieces)
        # SSE deltas are only an estimate; retokenize the finished pass so
        # the count retained beside the timer is exact.
        if result and not self._stop:
            try:
                self._tokens_total = tokens_before_pass + self._count_tokens(port, result)
            except Exception:
                pass
        self._reasoning_tokens += reasoning_tokens
        self._peak_prompt_tokens = max(self._peak_prompt_tokens,
                                       self._last_prompt_tokens)
        if finish_reason == "length" and not self._stop:
            self._truncated_passes += 1
            self._report_truncation(reasoning_tokens, stream)
        self._passes_done += 1
        return result

    def _report_truncation(self, reasoning_tokens: int, stream: bool) -> None:
        """Say so when the model was cut off at the completion cap.

        llama-server reports this as finish_reason "length" and otherwise
        says nothing: the notes simply stop, usually mid-sentence and always
        before the final per-person summary. On a reasoning model the cause
        is normally the hidden trace having eaten the allowance, which is
        worth naming because the fix is a different model, not a retry.
        """
        spent = (f" after spending ~{reasoning_tokens} tokens on reasoning "
                 f"this app discards" if reasoning_tokens else "")
        self.status.emit(f"Notes hit the {self._max_new_tokens()}-token limit{spent}.")
        if stream:
            self.chunk.emit(
                f"\n\n*(cut off at the {self._max_new_tokens()}-token "
                f"completion limit{spent} — try a shorter instruction set or "
                f"another model)*\n")

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

        The model's own input budget covers the whole prompt, and the
        instructions ride along with every single part — they are
        user-editable and can be long, so they are measured rather than
        assumed small. A reasoning model's extra completion allowance is
        taken out of here too: the server allocated its cache for the
        window, and wanting a longer answer does not change what it
        allocated.
        """
        instructions = self._try_count_tokens(port, self.instructions)
        if instructions is None:
            instructions = int(len(self.instructions) / CHARS_PER_TOKEN)
        return max(MIN_PROMPT_TOKENS,
                   self._max_input_tokens() - instructions
                   - TEMPLATE_OVERHEAD_TOKENS - self._reasoning_budget())

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
            port, self.model = _load(self.model_name, self.status.emit,
                                     cancelled=lambda: self._stop)
            _check_cancelled(lambda: self._stop)
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
                    port, min(_part_tokens(self._context_size()), budget),
                    chars_per_token)
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
            self._report_run()
            self.progress.emit(100, self._tokens_total)
            self.completed.emit()
        except _NotesCancelled:
            self.cancelled = True
            self.chunk.emit("\n\n*(cancelled)*")
            self.completed.emit()
        except Exception as exc:
            if self._stop:
                self.cancelled = True
                self.chunk.emit("\n\n*(cancelled)*")
                self.completed.emit()
            else:
                self.failed.emit(str(exc))


    def _report_run(self) -> None:
        """Record what the run cost beyond the notes the user can see.

        Reasoning tokens are generated, charged against the completion cap
        and then thrown away by this app, so a model that spent half its
        allowance thinking produces half the notes for the same wait with
        nothing on screen to say why. A pass that was cut off is just as
        invisible: the only other evidence is notes that stop mid-sentence.
        """
        parts = []
        if self._peak_prompt_tokens:
            parts.append(f"largest prompt {self._peak_prompt_tokens} tokens "
                         f"of {self._context_size()}")
        if self._reasoning_tokens:
            parts.append(f"~{self._reasoning_tokens} reasoning tokens discarded")
        if self._truncated_passes:
            parts.append(f"{self._truncated_passes} pass(es) cut off at the "
                         f"{self._max_new_tokens()}-token limit")
        if parts:
            self.status.emit("Notes finished — " + "; ".join(parts) + ".")
        if self._peak_prompt_tokens > self._context_size() - MAX_NEW_TOKENS:
            # The text was sized to fit; the template pushed it over. That
            # means llama-server silently dropped the front of a prompt, so
            # say so rather than leaving it to be noticed in the notes.
            self.status.emit(
                f"Warning: a prompt reached {self._peak_prompt_tokens} tokens — "
                f"raise TEMPLATE_OVERHEAD_TOKENS for this model.")


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
