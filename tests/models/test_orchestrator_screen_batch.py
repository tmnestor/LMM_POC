"""screen_batch must work on a backend that cannot batch.

The first real inference run died here. `screen_batch` was written by copying
`detect_batch`, including its `assert isinstance(backend, BatchInference)`.
That assert is sound in `detect_batch` because DocumentPipeline only calls it
when `supports_batch` is true and routes elsewhere otherwise -- a precondition
that lives in the caller, not the method. Copying the method without the
precondition turned a routing decision into a crash.

On this branch the vLLM backend implements no `generate_batch` at all, so
`supports_batch` is false and sequential is the ONLY path. These tests bind the
method to a stand-in rather than building an orchestrator, so the routing is
checked without a model.
"""

import pytest

from models.orchestrator import DocumentOrchestrator


class FakeBackend:
    """A backend with no batch support, like this branch's vLLM backend."""


class BatchingBackend:
    """A backend that can batch."""

    def __init__(self):
        self.batch_calls = []

    def generate_batch(self, images, prompts, params):
        self.batch_calls.append((images, prompts, params))
        return [f"batched {i}" for i in range(len(images))]


class FakeOrchestrator:
    """The smallest object screen_batch needs."""

    def __init__(self, backend, *, supports_batch: bool):
        self._backend = backend
        self.supports_batch = supports_batch
        self.generate_calls = []

    def load_document_image(self, path):
        return f"image:{path}"

    def generate(self, image, prompt, max_tokens, extra=None):
        self.generate_calls.append((image, prompt, max_tokens, extra))
        return f"answer for {image}"

    screen_batch = DocumentOrchestrator.screen_batch


def test_a_backend_without_batching_falls_back_to_one_call_per_image():
    """The case that crashed the first GPU run."""
    orchestrator = FakeOrchestrator(FakeBackend(), supports_batch=False)

    responses = orchestrator.screen_batch(["/a.png", "/b.png"], "ask", 200)

    assert responses == ["answer for image:/a.png", "answer for image:/b.png"]
    assert len(orchestrator.generate_calls) == 2


def test_the_sequential_path_sends_the_same_prompt_and_budget_every_time():
    orchestrator = FakeOrchestrator(FakeBackend(), supports_batch=False)

    orchestrator.screen_batch(["/a.png", "/b.png"], "ask", 200)

    assert {call[1] for call in orchestrator.generate_calls} == {"ask"}
    assert {call[2] for call in orchestrator.generate_calls} == {200}


def test_responses_come_back_in_image_order():
    """Records are paired positionally downstream, so order is load-bearing."""
    orchestrator = FakeOrchestrator(FakeBackend(), supports_batch=False)

    responses = orchestrator.screen_batch(["/z.png", "/a.png"], "ask", 200)

    assert responses == ["answer for image:/z.png", "answer for image:/a.png"]


def test_a_batching_backend_is_used_in_one_call():
    """Where batching exists it should still be one engine call, not a loop."""
    backend = BatchingBackend()
    orchestrator = FakeOrchestrator(backend, supports_batch=True)

    responses = orchestrator.screen_batch(["/a.png", "/b.png"], "ask", 200)

    assert len(backend.batch_calls) == 1
    assert responses == ["batched 0", "batched 1"]
    assert orchestrator.generate_calls == []


def test_tile_budgets_reach_the_model():
    """Without a budget the backend skips pre-tiling and lets vLLM choose the
    grid by aspect ratio, which puts a small receipt on about one tile. The
    floor is the lever, and it only works if it actually arrives."""
    orchestrator = FakeOrchestrator(FakeBackend(), supports_batch=False)

    orchestrator.screen_batch(["/a.png"], "ask", 200, tile_extra={"min_tiles": 6, "max_tiles": 6})

    assert orchestrator.generate_calls[0][3] == {"min_tiles": 6, "max_tiles": 6}


def test_no_tile_budget_passes_none_rather_than_an_empty_dict():
    """An empty dict and None mean different things downstream: the backend
    tests `max_tiles` truthiness to decide whether to pre-tile at all."""
    orchestrator = FakeOrchestrator(FakeBackend(), supports_batch=False)

    orchestrator.screen_batch(["/a.png"], "ask", 200)

    assert orchestrator.generate_calls[0][3] is None


@pytest.mark.parametrize("supports_batch", [True, False])
def test_an_empty_image_list_calls_nothing(supports_batch):
    backend = BatchingBackend()
    orchestrator = FakeOrchestrator(backend, supports_batch=supports_batch)

    assert orchestrator.screen_batch([], "ask", 200) == []
    assert backend.batch_calls == []
    assert orchestrator.generate_calls == []
