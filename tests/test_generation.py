"""The request llama-server receives, and what comes back out of the stream.

Runs against a fake llama-server on localhost — a few dozen lines of
http.server — because the payload fields (thinking switches, reasoning
effort) and the stream fields (reasoning_content, finish_reason, usage) are
the parts of llm.py with no other way to be checked without a 10 GB model.
"""

from __future__ import annotations

import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer

import tests  # noqa: F401  (puts app/ on sys.path)

import llm
from llm import NotesWorker, _apply_identity_policy


class FakeServer(BaseHTTPRequestHandler):
    """Records the request it was sent and replays a scripted SSE stream."""

    events: list[dict] = []
    received: list[dict] = []

    def log_message(self, *_args):
        pass  # keep the test output clean

    def do_POST(self):
        body = self.rfile.read(int(self.headers["Content-Length"]))
        request = json.loads(body)
        type(self).received.append(request)
        if self.path.endswith("/tokenize"):
            tokens = list(range(max(1, len(request.get("content", "")) // 4)))
            payload = json.dumps({"tokens": tokens}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
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

    @property
    def completion_request(self) -> dict:
        for request in FakeServer.received:
            if "messages" in request:
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

    def test_an_oversized_rendered_prompt_is_reported(self):
        """The text fit; the chat template pushed it past the context."""
        FakeServer.events = [delta("Notes."), delta(finish="stop"),
                             {"choices": [],
                              "usage": {"prompt_tokens": llm.CONTEXT_SIZE - 1}}]
        worker = self.worker()
        worker._generate(self.port, "prompt", stream=True)
        worker._report_run()
        self.assertTrue(any("TEMPLATE_OVERHEAD_TOKENS" in s for s in self.statuses))

    def test_cancelling_stops_reading_the_stream(self):
        FakeServer.events = [delta("one"), delta("two"), delta("three"),
                             delta(finish="stop")]
        worker = self.worker()
        worker.stop()
        result = worker._generate(self.port, "prompt", stream=True)
        self.assertEqual(result, "")


if __name__ == "__main__":
    unittest.main()
