"""The request llama-server receives, and what comes back out of the stream.

Runs against a fake llama-server on localhost — a few dozen lines of
http.server — because the payload fields (thinking switches, reasoning
effort), the stream fields (reasoning_content, finish_reason, usage) and the
size of every prompt a whole run sends are the parts of llm.py with no other
way to be checked without a 10 GB model.
"""

from __future__ import annotations

import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from unittest import mock

import tests  # noqa: F401  (puts app/ on sys.path)

import llm
from llm import NotesModel, NotesWorker, _apply_identity_policy

# What the fake's chat template adds around the messages, and what it adds
# besides when thinking is switched off — Qwen3's renders an empty think
# block then — so a count taken without the request's policy comes out short.
TEMPLATE_TOKENS = 12
NO_THINKING_TOKENS = 6


def rendered_tokens(request: dict) -> int:
    """The fake's count of a whole chat-completion body: a quarter of the
    characters of its messages, plus its template."""
    text = "".join(message["content"] for message in request["messages"])
    tokens = len(text) // 4 + TEMPLATE_TOKENS
    if (request.get("chat_template_kwargs") or {}).get("enable_thinking") is False:
        tokens += NO_THINKING_TOKENS
    return tokens


class FakeServer(BaseHTTPRequestHandler):
    """Records the requests it was sent and replays a scripted SSE stream.

    /tokenize counts a quarter of the characters, and
    /v1/chat/completions/input_tokens counts a whole request the same way
    plus its template. Either can be made to fail with an HTTP status.
    """

    events: list[dict] = []
    received: list[tuple[str, dict]] = []
    tokenize_status = 200
    count_status = 200

    def log_message(self, *_args):
        pass  # keep the test output clean

    def reply_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        body = self.rfile.read(int(self.headers["Content-Length"]))
        request = json.loads(body)
        type(self).received.append((self.path, request))
        if self.path.endswith("/tokenize"):
            if type(self).tokenize_status != 200:
                self.reply_json(type(self).tokenize_status, {"error": "no"})
                return
            tokens = list(range(max(1, len(request.get("content", "")) // 4)))
            self.reply_json(200, {"tokens": tokens})
            return
        if self.path.endswith("/input_tokens"):
            if type(self).count_status != 200:
                self.reply_json(type(self).count_status, {"error": "no"})
                return
            self.reply_json(200, {"object": "response.input_tokens",
                                  "input_tokens": rendered_tokens(request)})
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for event in type(self).events:
            self.wfile.write(f"data: {json.dumps(event)}\n\n".encode())
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()


def delta(content=None, reasoning=None, finish=None):
    payload = {}
    if content is not None:
        payload["content"] = content
    if reasoning is not None:
        payload["reasoning_content"] = reasoning
    return {"choices": [{"delta": payload, "finish_reason": finish}]}


class GenerationBase(unittest.TestCase):
    def setUp(self):
        FakeServer.events = []
        FakeServer.received = []
        FakeServer.tokenize_status = 200
        FakeServer.count_status = 200
        self.server = HTTPServer(("127.0.0.1", 0), FakeServer)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.chunks: list[str] = []
        self.statuses: list[str] = []

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)

    def worker(self, model_name="qwen3-4b"):
        worker = NotesWorker("Alice: hello.", model_name=model_name)
        worker.model = _apply_identity_policy(llm._resolve_model(model_name))
        worker.chunk.connect(self.chunks.append)
        worker.status.connect(self.statuses.append)
        return worker

    @staticmethod
    def requests_to(endpoint: str) -> list[dict]:
        return [request for path, request in FakeServer.received
                if path.endswith(endpoint)]

    @property
    def completion_request(self) -> dict:
        for request in self.requests_to("/v1/chat/completions"):
            return request
        raise AssertionError("no chat-completion request was made")


class Payload(GenerationBase):
    def test_a_thinking_model_is_told_not_to_think(self):
        FakeServer.events = [delta("notes", finish="stop")]
        self.worker("qwen3-14b")._generate(self.port, "prompt", stream=True)
        self.assertEqual(self.completion_request["chat_template_kwargs"],
                         {"enable_thinking": False})

    def test_a_direct_answer_model_gets_no_template_kwargs(self):
        FakeServer.events = [delta("notes", finish="stop")]
        self.worker("qwen3-4b")._generate(self.port, "prompt", stream=True)
        self.assertNotIn("chat_template_kwargs", self.completion_request)

    def test_gpt_oss_gets_its_reasoning_effort_both_ways(self):
        FakeServer.events = [delta("notes", finish="stop")]
        self.worker("gpt-oss-20b")._generate(self.port, "prompt", stream=True)
        request = self.completion_request
        self.assertEqual(request["reasoning_effort"], "low")
        self.assertEqual(request["chat_template_kwargs"]["reasoning_effort"], "low")

    def test_a_reasoning_model_asks_for_a_bigger_completion(self):
        FakeServer.events = [delta("notes", finish="stop")]
        self.worker("gpt-oss-20b")._generate(self.port, "prompt", stream=True)
        self.assertEqual(self.completion_request["max_tokens"],
                         llm.MAX_NEW_TOKENS + llm.REASONING_EXTRA_TOKENS)

    def test_a_vendor_temperature_is_used_when_there_is_one(self):
        FakeServer.events = [delta("notes", finish="stop")]
        self.worker("ministral-3-8b")._generate(self.port, "prompt", stream=True)
        self.assertEqual(self.completion_request["temperature"], 0.1)

    def test_the_app_default_applies_otherwise(self):
        FakeServer.events = [delta("notes", finish="stop")]
        self.worker("qwen3-4b")._generate(self.port, "prompt", stream=True)
        self.assertEqual(self.completion_request["temperature"], 0.7)
        self.assertEqual(self.completion_request["top_p"], 0.8)

    def test_usage_is_requested(self):
        FakeServer.events = [delta("notes", finish="stop")]
        self.worker()._generate(self.port, "prompt", stream=True)
        self.assertEqual(self.completion_request["stream_options"],
                         {"include_usage": True})


class Stream(GenerationBase):
    def test_content_is_streamed_and_returned(self):
        FakeServer.events = [delta("Some "), delta("notes."), delta(finish="stop")]
        result = self.worker()._generate(self.port, "prompt", stream=True)
        self.assertEqual(result, "Some notes.")
        self.assertEqual("".join(self.chunks), "Some notes.")

    def test_nothing_is_streamed_on_a_map_pass(self):
        FakeServer.events = [delta("Part notes."), delta(finish="stop")]
        result = self.worker()._generate(self.port, "prompt", stream=False)
        self.assertEqual(result, "Part notes.")
        self.assertEqual(self.chunks, [])

    def test_reasoning_is_counted_but_never_shown(self):
        FakeServer.events = [delta(reasoning="thinking"), delta(reasoning="more"),
                             delta("Answer."), delta(finish="stop")]
        worker = self.worker("gpt-oss-20b")
        result = worker._generate(self.port, "prompt", stream=True)
        self.assertEqual(result, "Answer.")
        self.assertNotIn("thinking", "".join(self.chunks))
        self.assertEqual(worker._reasoning_tokens, 2)

    def test_a_truncated_pass_says_so_in_the_notes(self):
        FakeServer.events = [delta("Cut off here"), delta(finish="length")]
        worker = self.worker()
        worker._generate(self.port, "prompt", stream=True)
        self.assertEqual(worker._truncated_passes, 1)
        self.assertIn("cut off", "".join(self.chunks).lower())
        self.assertTrue(any("limit" in s for s in self.statuses))

    def test_a_complete_pass_says_nothing(self):
        FakeServer.events = [delta("All of it."), delta(finish="stop")]
        worker = self.worker()
        worker._generate(self.port, "prompt", stream=True)
        self.assertEqual(worker._truncated_passes, 0)
        self.assertNotIn("cut off", "".join(self.chunks).lower())

    def test_the_usage_only_event_is_tolerated(self):
        """llama-server ends an include_usage stream with an empty choices list."""
        FakeServer.events = [delta("Notes."), delta(finish="stop"),
                             {"choices": [], "usage": {"prompt_tokens": 1234}}]
        worker = self.worker()
        result = worker._generate(self.port, "prompt", stream=True)
        self.assertEqual(result, "Notes.")
        self.assertEqual(worker._peak_prompt_tokens, 1234)

    def oversized(self, model_name: str, prompt_tokens: int,
                  estimated: bool = False) -> list[str]:
        FakeServer.events = [delta("Notes."), delta(finish="stop"),
                             {"choices": [],
                              "usage": {"prompt_tokens": prompt_tokens}}]
        worker = self.worker(model_name)
        worker._sizing_estimated = estimated
        worker._generate(self.port, "prompt", stream=True)
        worker._report_run()
        return [s for s in self.statuses if s.startswith("Warning")]

    def test_an_oversized_rendered_prompt_is_reported(self):
        """The server held more than the prompt was sized for."""
        warnings = self.oversized("qwen3-4b", llm.CONTEXT_SIZE - 1)
        self.assertEqual(len(warnings), 1)
        self.assertIn("CONTEXT_HEADROOM", warnings[0])

    def test_an_estimated_one_blames_the_template_guess(self):
        warnings = self.oversized("qwen3-4b", llm.CONTEXT_SIZE - 1, estimated=True)
        self.assertEqual(len(warnings), 1)
        self.assertIn("TEMPLATE_OVERHEAD_TOKENS", warnings[0])

    def test_a_prompt_that_leaves_the_whole_answer_room_is_not(self):
        self.assertEqual(
            self.oversized("qwen3-4b", llm.CONTEXT_SIZE - llm.MAX_NEW_TOKENS), [])

    def test_a_reasoning_models_answer_room_includes_its_reasoning(self):
        """gpt-oss needs MAX_NEW_TOKENS + REASONING_EXTRA_TOKENS left over."""
        context = llm.LLM_REPOS["gpt-oss-20b"].context
        answer = llm.MAX_NEW_TOKENS + llm.REASONING_EXTRA_TOKENS
        self.assertEqual(self.oversized("gpt-oss-20b", context - answer), [])
        self.assertEqual(len(self.oversized("gpt-oss-20b", context - answer + 1)), 1)

    def test_cancelling_stops_reading_the_stream(self):
        FakeServer.events = [delta("one"), delta("two"), delta("three"),
                             delta(finish="stop")]
        worker = self.worker()
        worker.stop()
        result = worker._generate(self.port, "prompt", stream=True)
        self.assertEqual(result, "")


class Runs(GenerationBase):
    """Whole runs against the fake server: what every prompt costs it."""

    LONG = "\n".join(f"Speaker {i}: " + "word " * 60 for i in range(600))

    def run_worker(self, transcript: str, model_name: str = "qwen3-14b",
                   instructions: str = "", notes: str = "Notes.",
                   model: NotesModel | None = None):
        FakeServer.events = [delta(notes), delta(finish="stop")]
        worker = NotesWorker(transcript, instructions, model_name=model_name)
        worker.chunk.connect(self.chunks.append)
        worker.status.connect(self.statuses.append)
        self.failures: list[str] = []
        worker.failed.connect(self.failures.append)
        model = model or _apply_identity_policy(llm._resolve_model(model_name))
        with mock.patch.object(llm, "_load", return_value=(self.port, model)):
            worker.run()
        return worker

    def assert_every_prompt_fits(self, worker: NotesWorker) -> list[dict]:
        sent = self.requests_to("/v1/chat/completions")
        self.assertTrue(sent)
        for request in sent:
            self.assertLessEqual(rendered_tokens(request), worker._max_input_tokens())
            self.assertLessEqual(rendered_tokens(request) + request["max_tokens"],
                                 worker._context_size())
        return sent

    def test_a_short_transcript_is_one_prompt(self):
        worker = self.run_worker("Alice: hello.\nBob: hi.")
        self.assertEqual(self.failures, [])
        self.assertEqual(len(self.assert_every_prompt_fits(worker)), 1)
        self.assertEqual("".join(self.chunks), "Notes.")

    def test_every_pass_of_a_long_run_fits(self):
        """Parts, a round of combining and the final merge, all of them.

        Each pass answers with ~12,000 tokens of notes, so no two of them
        fit one combining prompt: they are paired up and trimmed to fit, and
        merged in rounds before the final merge.
        """
        worker = self.run_worker(self.LONG, notes="n" * 48_000)
        self.assertEqual(self.failures, [])
        prompts = [r["messages"][1]["content"]
                   for r in self.assert_every_prompt_fits(worker)]
        self.assertTrue(any("This is part 1 of" in p for p in prompts))
        self.assertTrue(any("### Notes on part 1" in p for p in prompts))
        self.assertTrue(any("### Notes on section 1" in p for p in prompts))
        self.assertTrue(any("truncated to fit" in p for p in prompts))

    def test_a_reasoning_model_at_its_own_window(self):
        """gpt-oss: a 40,960 window, less 4,000 of answer — once."""
        worker = self.run_worker(self.LONG, model_name="gpt-oss-20b",
                                 notes="n" * 30_000)
        self.assertEqual(self.failures, [])
        self.assertEqual(worker._max_input_tokens(),
                         40960 - llm.MAX_NEW_TOKENS - llm.REASONING_EXTRA_TOKENS
                         - llm.CONTEXT_HEADROOM)
        self.assert_every_prompt_fits(worker)

    def test_the_count_is_taken_with_the_requests_own_policy(self):
        self.run_worker("Alice: hello.", model_name="qwen3-14b")
        counted = self.requests_to("/input_tokens")
        self.assertTrue(counted)
        sent = self.completion_request
        for request in counted:
            self.assertEqual(request["chat_template_kwargs"],
                             sent["chat_template_kwargs"])
            self.assertEqual(request["messages"][0], sent["messages"][0])

    def test_oversized_instructions_never_reach_generation(self):
        for transcript in ("Alice: hello.", self.LONG):
            with self.subTest(long=transcript is self.LONG):
                FakeServer.received = []
                self.run_worker(transcript, instructions="x" * 100_000)
                self.assertEqual(self.requests_to("/v1/chat/completions"), [])
                self.assertEqual(len(self.failures), 1)
                self.assertIn("Shorten the AI instructions", self.failures[0])

    def test_instructions_just_inside_the_limit_still_run(self):
        """~17,600 tokens of instructions in a 20,000-token prompt limit."""
        worker = self.run_worker("Alice: hello.", instructions="x" * 70_000)
        self.assertEqual(self.failures, [])
        self.assert_every_prompt_fits(worker)

    def test_without_the_servers_count_a_run_still_fits(self):
        """An older llama-server: no input_tokens endpoint."""
        FakeServer.count_status = 404
        worker = self.run_worker(self.LONG, notes="n" * 30_000)
        self.assertEqual(self.failures, [])
        self.assertTrue(worker._sizing_estimated)
        self.assert_every_prompt_fits(worker)
        self.assertTrue(any("sized by estimate" in s for s in self.statuses))

    def test_without_any_count_a_run_still_completes(self):
        FakeServer.count_status = 500
        FakeServer.tokenize_status = 500
        worker = self.run_worker("Alice: hello.\nBob: hi.")
        self.assertEqual(self.failures, [])
        self.assertEqual("".join(self.chunks), "Notes.")
        self.assertTrue(any("sized by estimate" in s for s in self.statuses))

    def test_a_counted_run_says_nothing_about_estimates(self):
        self.run_worker("Alice: hello.")
        self.assertFalse(any("estimate" in s for s in self.statuses))


if __name__ == "__main__":
    unittest.main()
