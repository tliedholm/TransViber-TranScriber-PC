"""Notes-model resolution, generation policy, and prompt sizing.

No server is started: the sizing code takes its token counts through
``_try_count_tokens``, so a stubbed counter exercises the real packing,
splitting and reduction logic offline.
"""

from __future__ import annotations

import unittest
from pathlib import Path

import tests  # noqa: F401  (puts app/ on sys.path)

import llm
from llm import (
    CONTEXT_SIZE,
    MAX_INPUT_TOKENS,
    MAX_NEW_TOKENS,
    NotesModel,
    NotesWorker,
    _apply_identity_policy,
    _fallback_ladder,
    _launch_attempts,
    _LeadingThinkFilter,
    _resolve_model,
    _runtime_report,
    _shard_set,
    _split_long_line,
)


def policy(name: str) -> NotesModel:
    return _apply_identity_policy(_resolve_model(name))


class Catalog(unittest.TestCase):
    def test_every_dropdown_entry_exists(self):
        for group, names in llm.LLM_GROUPS:
            for name in names:
                with self.subTest(group=group, name=name):
                    self.assertIn(name, llm.LLM_REPOS)

    def test_the_default_is_in_the_catalog_and_the_dropdown(self):
        self.assertIn(llm.DEFAULT_LLM, llm.LLM_REPOS)
        listed = [n for _, names in llm.LLM_GROUPS for n in names]
        self.assertIn(llm.DEFAULT_LLM, listed)

    def test_no_entry_is_listed_in_two_groups(self):
        listed = [n for _, names in llm.LLM_GROUPS for n in names]
        self.assertEqual(len(listed), len(set(listed)))

    def test_the_dropdown_offers_one_entry_per_model(self):
        """No two dropdown entries may be two quants of the same weights.

        The dropdown was 34 entries, over a third of them a second quant of
        the model above. Keeping it one-per-role is the whole point of the
        -q4 fallbacks living outside LLM_GROUPS.
        """
        listed = [n for _, names in llm.LLM_GROUPS for n in names]
        repos = [llm.LLM_REPOS[n].repo for n in listed]
        self.assertEqual(len(repos), len(set(repos)), sorted(repos))

    def test_no_fallback_or_superseded_entry_is_in_the_dropdown(self):
        listed = [n for _, names in llm.LLM_GROUPS for n in names]
        self.assertEqual([n for n in listed if n.endswith("-q4")], [])

    def test_every_fallback_points_at_its_base_model(self):
        """A -q4 entry is a smaller build of the same weights, not another model."""
        for name, model in llm.LLM_REPOS.items():
            if not name.endswith("-q4"):
                continue
            base = llm.LLM_REPOS.get(name[:-3])
            with self.subTest(name=name):
                self.assertIsNotNone(base, f"{name} has no base entry")
                self.assertEqual(model.repo, base.repo)
                self.assertNotEqual(model.filename, base.filename)
                self.assertIn("q4", model.filename.lower())

    def test_every_catalog_entry_names_an_exact_file(self):
        for name, model in llm.LLM_REPOS.items():
            with self.subTest(name=name):
                self.assertTrue(model.filename)
                self.assertTrue(model.filename.lower().endswith(".gguf"))

    def test_the_context_budget_adds_up(self):
        self.assertEqual(CONTEXT_SIZE, MAX_INPUT_TOKENS + MAX_NEW_TOKENS + 500)


class Resolution(unittest.TestCase):
    def test_an_alias_resolves_to_its_exact_file(self):
        model = _resolve_model("qwen3-14b")
        self.assertEqual(model.repo, "unsloth/Qwen3-14B-GGUF")
        self.assertEqual(model.filename, "Qwen3-14B-Q5_K_M.gguf")

    def test_an_alias_is_case_insensitive(self):
        self.assertEqual(_resolve_model("QWEN3-14B"), _resolve_model("qwen3-14b"))

    def test_a_typed_repo_and_file_is_taken_literally(self):
        model = _resolve_model("owner/Repo-GGUF::Some-Model-Q5_K_M.gguf")
        self.assertEqual(model.repo, "owner/Repo-GGUF")
        self.assertEqual(model.filename, "Some-Model-Q5_K_M.gguf")

    def test_a_bare_repo_leaves_the_file_to_be_discovered(self):
        self.assertIsNone(_resolve_model("owner/Repo-GGUF").filename)


class IdentityPolicy(unittest.TestCase):
    """Policy has to follow the weights, not the name that was typed."""

    def test_thinking_is_off_for_hybrid_qwen_models(self):
        for name in ("qwen3-8b", "qwen3-14b", "qwen3.5-9b", "qwen3.5-4b",
                     "qwen3.8-27b"):
            with self.subTest(name=name):
                self.assertEqual(policy(name).template_kwargs,
                                 {"enable_thinking": False})

    def test_a_typed_path_gets_the_same_policy_as_the_alias(self):
        typed = policy("unsloth/Qwen3-14B-GGUF::Qwen3-14B-Q5_K_M.gguf")
        self.assertEqual(typed.template_kwargs, policy("qwen3-14b").template_kwargs)

    def test_a_bare_repo_is_recognised_too(self):
        self.assertEqual(policy("unsloth/Qwen3.5-9B-GGUF").template_kwargs,
                         {"enable_thinking": False})

    def test_direct_answer_qwen_gets_no_thinking_switch(self):
        """Instruct-2507 has no thinking mode; setting one is meaningless."""
        self.assertIsNone(policy("qwen3-4b").template_kwargs)
        self.assertIsNone(policy("qwen3-4b-q4").template_kwargs)

    def test_gpt_oss_budgets_reasoning_instead_of_disabling_it(self):
        model = policy("gpt-oss-20b")
        self.assertEqual(model.reasoning_effort, "low")
        self.assertIsNone(model.template_kwargs)

    def test_granite_4_covers_both_spellings_of_the_switch(self):
        self.assertEqual(policy("granite-4.2-8b").template_kwargs,
                         {"enable_thinking": False, "thinking": False})

    def test_granite_3_has_no_switch(self):
        self.assertIsNone(policy("granite-3.3-2b").template_kwargs)

    def test_ministral_3_uses_its_vendor_temperature(self):
        self.assertEqual(policy("ministral-3-8b").temperature, 0.1)
        self.assertIsNone(policy("ministral-8b").temperature)

    def test_lfm2_is_quietened_at_the_server_not_the_template(self):
        """Its template has no thinking switch; only the server flag works.

        Measured on LFM2.5-8B-A1B Q8_0 at temperature 0: `--reasoning off`
        and an `enable_thinking` kwarg both produced byte-identical output
        to the default, while `--reasoning-budget 0` cut the pass from 609
        tokens to 286 and roughly doubled the notes.
        """
        for name in ("lfm2.5-8b-a1b", "lfm2.5-8b-a1b-q6"):
            with self.subTest(name=name):
                model = policy(name)
                self.assertEqual(model.server_args, ("--reasoning-budget", "0"))
                self.assertIsNone(model.template_kwargs)
                self.assertIsNone(model.reasoning_effort)

    def test_a_typed_lfm2_path_is_quietened_too(self):
        typed = policy("LiquidAI/LFM2.5-8B-A1B-GGUF::LFM2.5-8B-A1B-Q8_0.gguf")
        self.assertEqual(typed.server_args, ("--reasoning-budget", "0"))

    def test_gemma_4_switches_thinking_off_and_gemma_3_is_untouched(self):
        self.assertEqual(policy("gemma-4-12b").template_kwargs,
                         {"enable_thinking": False})
        self.assertEqual(policy("gemma-4-12b-qat").template_kwargs,
                         {"enable_thinking": False})
        self.assertIsNone(policy("gemma-3-12b").template_kwargs)

    def test_only_lfm2_needs_server_flags(self):
        """Every other model is settled per request, which is cheaper."""
        for _, names in llm.LLM_GROUPS:
            for name in names:
                with self.subTest(name=name):
                    if "lfm2" in name:
                        continue
                    self.assertIsNone(policy(name).server_args)

    def test_an_unknown_model_gets_no_policy_at_all(self):
        model = policy("owner/Whatever-GGUF::Whatever-Q4_K_M.gguf")
        self.assertIsNone(model.template_kwargs)
        self.assertIsNone(model.reasoning_effort)
        self.assertIsNone(model.temperature)
        self.assertIsNone(model.server_args)

    def test_a_catalog_entry_overrides_the_pattern(self):
        explicit = NotesModel("unsloth/Qwen3-14B-GGUF", "Qwen3-14B-Q4_K_M.gguf",
                              template_kwargs={"enable_thinking": True})
        self.assertEqual(_apply_identity_policy(explicit).template_kwargs,
                         {"enable_thinking": True})


class LaunchAttempts(unittest.TestCase):
    """A model's own flags have to survive every VRAM retry."""

    def test_a_plain_model_starts_with_no_extra_flags(self):
        attempts = _launch_attempts(NotesModel("owner/Repo", "file.gguf"))
        self.assertEqual(attempts[0], ([], ""))

    def test_server_args_ride_along_every_rung_of_the_ladder(self):
        model = policy("lfm2.5-8b-a1b")
        attempts = _launch_attempts(model)
        self.assertGreater(len(attempts), 1)  # MoE: it has a ladder
        for extra, _ in attempts:
            with self.subTest(extra=extra):
                self.assertEqual(extra[:2], ["--reasoning-budget", "0"])

    def test_the_ladder_still_carries_its_own_flags(self):
        model = policy("lfm2.5-8b-a1b")
        laddered = [extra for extra, _ in _launch_attempts(model)[1:]]
        self.assertIn(["--reasoning-budget", "0", "--n-cpu-moe", "8"], laddered)
        self.assertIn(["--reasoning-budget", "0", "--cpu-moe"], laddered)

    def test_one_attempt_cannot_corrupt_the_next(self):
        """Each rung needs its own list, not a shared one that got appended to."""
        model = policy("lfm2.5-8b-a1b")
        attempts = _launch_attempts(model)
        attempts[0][0].append("--poison")
        self.assertNotIn("--poison", attempts[1][0])


class PerModelContext(unittest.TestCase):
    """The window is per-model; what is carved out of it is not."""

    def test_the_defaults_are_the_numbers_they_have_always_been(self):
        """Deriving them must not quietly move anyone's budget."""
        self.assertEqual(llm.DEFAULT_CONTEXT_SIZE, 23000)
        self.assertEqual(MAX_INPUT_TOKENS, 20000)
        self.assertEqual(llm.PART_TOKENS, 14000)
        self.assertEqual(CONTEXT_SIZE, MAX_INPUT_TOKENS + MAX_NEW_TOKENS + 500)

    def test_a_model_without_one_gets_the_default(self):
        worker = StubbedWorker("")
        worker.model = NotesModel("owner/Repo", "f.gguf")
        self.assertEqual(worker._context_size(), llm.DEFAULT_CONTEXT_SIZE)
        self.assertEqual(worker._max_input_tokens(), MAX_INPUT_TOKENS)

    def test_no_model_at_all_still_answers(self):
        """_report_run can run after a failed load, with self.model unset."""
        self.assertEqual(StubbedWorker("")._context_size(),
                         llm.DEFAULT_CONTEXT_SIZE)

    def test_a_bigger_window_buys_prompt_room_not_answer_room(self):
        worker = StubbedWorker("")
        worker.model = NotesModel("owner/Repo", "f.gguf", context=65536)
        self.assertEqual(worker._context_size(), 65536)
        self.assertEqual(worker._max_input_tokens(),
                         65536 - MAX_NEW_TOKENS - llm.CONTEXT_HEADROOM)
        self.assertEqual(worker._max_new_tokens(), MAX_NEW_TOKENS)
        self.assertGreater(worker._prompt_budget(0), MAX_INPUT_TOKENS)

    def test_parts_scale_with_the_window(self):
        self.assertEqual(llm._part_tokens(llm.DEFAULT_CONTEXT_SIZE), 14000)
        self.assertGreater(llm._part_tokens(65536),
                           llm._part_tokens(llm.DEFAULT_CONTEXT_SIZE))

    def test_a_part_always_leaves_room_for_the_instructions(self):
        for context in (llm.DEFAULT_CONTEXT_SIZE, 40960, 65536):
            with self.subTest(context=context):
                self.assertLess(llm._part_tokens(context),
                                llm._input_tokens(context))

    def test_the_server_is_started_at_the_models_window(self):
        command = llm._server_command(Path("m.gguf"), 8080, [], 65536)
        self.assertIn("65536", command)
        self.assertEqual(command[command.index("-c") + 1], "65536")

    def test_the_server_defaults_to_the_default_window(self):
        command = llm._server_command(Path("m.gguf"), 8080, [])
        self.assertEqual(command[command.index("-c") + 1],
                         str(llm.DEFAULT_CONTEXT_SIZE))

    def test_context_is_never_set_by_a_pattern(self):
        """Headroom belongs to the quant, so it cannot ride on the family."""
        for _, policy_fields in llm._IDENTITY_POLICIES:
            self.assertNotIn("context", policy_fields)

    def test_every_enlarged_entry_is_one_that_was_measured(self):
        """Only these three were measured; nothing else may claim a window."""
        measured = {"lfm2.5-8b-a1b": 65536, "lfm2.5-8b-a1b-q6": 65536,
                    "gpt-oss-20b": 40960}
        for name, entry in llm.LLM_REPOS.items():
            with self.subTest(name=name):
                if entry.context is None:
                    continue
                self.assertEqual(entry.context, measured.get(name))

    def test_an_enlarged_window_is_actually_bigger(self):
        for name, entry in llm.LLM_REPOS.items():
            if entry.context is not None:
                with self.subTest(name=name):
                    self.assertGreater(entry.context, llm.DEFAULT_CONTEXT_SIZE)


class LeadingThink(unittest.TestCase):
    """--reasoning-budget 0 leaves an empty <think></think> in the content."""

    @staticmethod
    def run_filter(deltas):
        f = _LeadingThinkFilter()
        return "".join(f.feed(d) for d in deltas) + f.flush()

    def test_the_empty_block_is_dropped_across_deltas(self):
        """Exactly how llama-server streams it, token by token."""
        self.assertEqual(
            self.run_filter(["\n", "<think>", "</think>", "\n", "##", " Notes"]),
            "## Notes")

    def test_it_is_dropped_when_it_arrives_in_one_piece(self):
        self.assertEqual(
            self.run_filter(["\n<think></think>\n## Notes"]), "## Notes")

    def test_ordinary_notes_pass_through_untouched(self):
        for deltas in (["## Notes", " and more"],
                       ["\n", "## Notes"],
                       ["<", "b", ">bold"]):
            with self.subTest(deltas=deltas):
                self.assertEqual(self.run_filter(deltas), "".join(deltas))

    def test_a_non_empty_trace_is_left_visible(self):
        """It means the server stopped extracting — worth seeing, not hiding."""
        text = "<think>weighing it up</think>\n## Notes"
        self.assertEqual(self.run_filter([text]), text)

    def test_a_later_think_tag_is_not_touched(self):
        text = "## Notes\nAna said <think></think> out loud"
        self.assertEqual(self.run_filter([text]), text)

    def test_a_stream_that_ends_mid_tag_is_not_swallowed(self):
        """A pass cut off inside the tag still has to surface its text."""
        self.assertEqual(self.run_filter(["\n", "<think>"]), "\n<think>")
        self.assertEqual(self.run_filter(["\n"]), "\n")

    def test_an_empty_stream_stays_empty(self):
        self.assertEqual(self.run_filter([]), "")


class Shards(unittest.TestCase):
    def test_an_unsplit_file_is_its_own_set(self):
        self.assertEqual(_shard_set("gemma-4-12b-it-Q6_K.gguf"),
                         ["gemma-4-12b-it-Q6_K.gguf"])

    def test_a_split_file_expands_to_every_shard(self):
        self.assertEqual(
            _shard_set("Model-Q4_K_M-00001-of-00003.gguf"),
            ["Model-Q4_K_M-00001-of-00003.gguf",
             "Model-Q4_K_M-00002-of-00003.gguf",
             "Model-Q4_K_M-00003-of-00003.gguf"])

    def test_a_shard_in_a_subfolder_keeps_its_folder(self):
        self.assertEqual(_shard_set("Q4_K_M/m-00001-of-00002.gguf"),
                         ["Q4_K_M/m-00001-of-00002.gguf",
                          "Q4_K_M/m-00002-of-00002.gguf"])


class FallbackLadder(unittest.TestCase):
    def test_an_moe_model_sheds_experts_first(self):
        steps = _fallback_ladder(llm.LLM_REPOS["gpt-oss-20b"])
        self.assertIn("--n-cpu-moe", steps[0][0])
        self.assertIn("--cpu-moe", steps[-1][0])

    def test_a_dense_model_sheds_layers(self):
        steps = _fallback_ladder(llm.LLM_REPOS["qwen3-14b"])
        self.assertTrue(all("-ngl" in args for args, _ in steps))

    def test_every_step_explains_itself(self):
        for model in (llm.LLM_REPOS["gpt-oss-20b"], llm.LLM_REPOS["qwen3-14b"]):
            for _args, description in _fallback_ladder(model):
                self.assertTrue(description)


class RuntimeReport(unittest.TestCase):
    LOG = """version: 10034 (505b1ed15)
load_tensors: offloaded 41/41 layers to GPU
load_tensors:        CUDA0 model buffer size =  9216.00 MiB
llama_kv_cache:      CUDA0 KV buffer size =   300.00 MiB
llama_context:      CUDA0 compute buffer size =   512.00 MiB
"""

    def test_reports_the_binary_and_the_allocation(self):
        report = _runtime_report(self.LOG)
        self.assertIn("10034 (505b1ed15)", report)
        self.assertIn("41/41 layers on GPU", report)
        self.assertIn("weights 9.00 GiB", report)

    def test_a_partial_offload_is_called_out(self):
        report = _runtime_report(self.LOG.replace("41/41", "20/41"))
        self.assertIn("PARTIAL", report)

    def test_an_empty_log_reports_nothing_rather_than_guessing(self):
        self.assertEqual(_runtime_report(""), "")


class SplitLongLine(unittest.TestCase):
    def test_a_short_line_is_left_alone(self):
        self.assertEqual(_split_long_line("short", 100), ["short"])

    def test_pieces_respect_the_limit(self):
        line = "word " * 500
        for piece in _split_long_line(line, 100):
            self.assertLessEqual(len(piece), 100)

    def test_it_prefers_to_cut_at_a_space(self):
        pieces = _split_long_line("aaa bbb ccc ddd eee", 12)
        self.assertTrue(all(not p.startswith(" ") for p in pieces))
        self.assertEqual("".join(p if p.endswith(" ") else p + " "
                                 for p in pieces).split(),
                         "aaa bbb ccc ddd eee".split())

    def test_an_unbreakable_line_is_still_cut(self):
        pieces = _split_long_line("x" * 250, 100)
        self.assertEqual(len(pieces), 3)
        self.assertLessEqual(max(len(p) for p in pieces), 100)


class StubbedWorker(NotesWorker):
    """A worker whose token counts come from a ratio instead of a server."""

    CHARS_PER_TOKEN = 4.0

    def _try_count_tokens(self, port, text):
        return max(1, round(len(text) / self.CHARS_PER_TOKEN))


class Sizing(unittest.TestCase):
    def setUp(self):
        self.worker = StubbedWorker("", model_name="qwen3-4b")

    def test_instructions_come_out_of_the_prompt_budget(self):
        budget = self.worker._prompt_budget(0)
        self.assertLess(budget, MAX_INPUT_TOKENS)
        self.assertGreater(budget, llm.MIN_PROMPT_TOKENS)

    def test_a_reasoning_model_trades_prompt_room_for_answer_room(self):
        """Its extra completion allowance must not enlarge the context.

        Both models are given the same window on purpose: since context
        became per-model, comparing two catalog entries would measure the
        window difference instead of the thing this protects.
        """
        for context in (None, 65536):
            with self.subTest(context=context):
                plain = StubbedWorker("")
                plain.model = NotesModel("owner/Repo", "f.gguf", context=context)
                reasoning = StubbedWorker("")
                reasoning.model = NotesModel("owner/Repo", "f.gguf",
                                             context=context,
                                             reasoning_effort="low")

                self.assertEqual(
                    reasoning._max_new_tokens(),
                    plain._max_new_tokens() + llm.REASONING_EXTRA_TOKENS)
                self.assertEqual(
                    reasoning._prompt_budget(0) + reasoning._max_new_tokens(),
                    plain._prompt_budget(0) + plain._max_new_tokens())

    def test_enormous_instructions_cannot_drive_the_budget_to_nothing(self):
        worker = StubbedWorker("", instructions="x" * 400_000)
        self.assertEqual(worker._prompt_budget(0), llm.MIN_PROMPT_TOKENS)

    def test_packing_never_exceeds_the_budget(self):
        items = [f"line {i} " + "word " * 40 for i in range(200)]
        groups = self.worker._pack(0, items, "\n", 500, 4.0)
        self.assertTrue(groups)
        for group in groups:
            tokens = self.worker._try_count_tokens(0, "\n".join(group))
            self.assertLessEqual(tokens, 500)

    def test_packing_keeps_every_item_in_order(self):
        items = [f"item{i}" for i in range(50)]
        groups = self.worker._pack(0, items, "\n", 50, 4.0)
        self.assertEqual([item for group in groups for item in group], items)

    def test_an_oversized_item_is_cut_rather_than_passed_through(self):
        items = ["x" * 40_000]
        groups = self.worker._pack(0, items, "\n", 500, 4.0,
                                   split_item=_split_long_line)
        self.assertGreater(len(groups), 1)
        for group in groups:
            self.assertLessEqual(
                self.worker._try_count_tokens(0, "\n".join(group)), 500)

    def test_a_transcript_splits_into_parts_that_fit(self):
        worker = StubbedWorker("\n".join(f"Speaker {i}: " + "word " * 60
                                         for i in range(400)))
        parts = worker._split_transcript(0, 1000, 4.0)
        self.assertGreater(len(parts), 1)
        for part in parts:
            self.assertLessEqual(worker._try_count_tokens(0, part), 1000)

    def test_splitting_loses_no_text(self):
        transcript = "\n".join(f"Speaker {i}: hello there" for i in range(300))
        worker = StubbedWorker(transcript)
        parts = worker._split_transcript(0, 200, 4.0)
        self.assertEqual("\n".join(parts).split(), transcript.split())

    def test_the_measured_ratio_is_clamped_to_something_sane(self):
        worker = StubbedWorker("a" * 1000)
        self.assertEqual(worker._chars_per_token(0), llm.CHARS_PER_TOKEN)
        self.assertLessEqual(worker._chars_per_token(1), 8.0)
        self.assertGreaterEqual(worker._chars_per_token(10_000), 1.0)

    def test_trimming_says_so_in_the_output(self):
        text = "x" * 100_000
        trimmed = self.worker._trim_to_budget(0, text, 500, 4.0)
        self.assertLess(len(trimmed), len(text))
        self.assertIn("truncated", trimmed)


if __name__ == "__main__":
    unittest.main()
