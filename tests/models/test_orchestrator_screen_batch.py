"""screen_batch sends one image at a time, and nothing else.

The first real inference run died here. `screen_batch` was written by copying
`detect_batch`, including its `assert isinstance(backend, BatchInference)`.
That assert was sound where it came from, because DocumentPipeline only called
`detect_batch` when `supports_batch` was true -- a precondition that lived in
the caller, not the method. Copied without its caller, a routing decision
became a crash.

The batched path is now gone rather than guarded. No backend in this repo has
ever implemented `generate_batch`, so it could not run; keeping an unreachable
branch alive meant keeping the crash reachable the moment anything set the flag.
Throughput comes from sharding across GPUs in `common.vllm_dp`, not from
batching inside one process.

These tests bind the method to a stand-in rather than building an orchestrator,
so what it does is checked without a model.
"""

from models.orchestrator import DocumentOrchestrator


class FakeBackend:
    """The backend plays no part now; screen_batch only calls self.generate."""


class FakeOrchestrator:
    """The smallest object screen_batch needs."""

    def __init__(self, backend=None):
        self._backend = backend or FakeBackend()
        self.generate_calls = []

    def load_document_image(self, path):
        return f"image:{path}"

    def generate(self, image, prompt, max_tokens, extra=None):
        self.generate_calls.append((image, prompt, max_tokens, extra))
        return f"answer for {image}"

    screen_batch = DocumentOrchestrator.screen_batch


def test_one_call_per_image():
    """The case that crashed the first GPU run."""
    orchestrator = FakeOrchestrator()

    responses = orchestrator.screen_batch(["/a.png", "/b.png"], "ask", 200)

    assert responses == ["answer for image:/a.png", "answer for image:/b.png"]
    assert len(orchestrator.generate_calls) == 2


def test_the_same_prompt_and_budget_go_with_every_image():
    orchestrator = FakeOrchestrator()

    orchestrator.screen_batch(["/a.png", "/b.png"], "ask", 200)

    assert {call[1] for call in orchestrator.generate_calls} == {"ask"}
    assert {call[2] for call in orchestrator.generate_calls} == {200}


def test_responses_come_back_in_image_order():
    """Records are paired positionally downstream, so order is load-bearing."""
    orchestrator = FakeOrchestrator()

    responses = orchestrator.screen_batch(["/z.png", "/a.png"], "ask", 200)

    assert responses == ["answer for image:/z.png", "answer for image:/a.png"]


def test_a_backend_that_offers_generate_batch_is_still_not_used():
    """No routing on the backend at all any more.

    A backend growing a `generate_batch` must not silently change how the
    screen runs -- that is how the deleted branch came back to life in the
    first place. Adopting batching would be a deliberate edit here, with the
    per-image trace attribution rethought at the same time.
    """

    class BatchingBackend:
        def __init__(self):
            self.batch_calls = []

        def generate_batch(self, images, prompts, params):
            self.batch_calls.append((images, prompts, params))
            return [f"batched {i}" for i in range(len(images))]

    backend = BatchingBackend()
    orchestrator = FakeOrchestrator(backend)

    responses = orchestrator.screen_batch(["/a.png", "/b.png"], "ask", 200)

    assert backend.batch_calls == []
    assert responses == ["answer for image:/a.png", "answer for image:/b.png"]


def test_tile_budgets_reach_the_model():
    """Without a budget the backend skips pre-tiling and lets vLLM choose the
    grid by aspect ratio, which puts a small receipt on about one tile. The
    floor is the lever, and it only works if it actually arrives."""
    orchestrator = FakeOrchestrator()

    orchestrator.screen_batch(["/a.png"], "ask", 200, tile_extra={"min_tiles": 6, "max_tiles": 6})

    assert orchestrator.generate_calls[0][3] == {"min_tiles": 6, "max_tiles": 6}


def test_no_tile_budget_passes_none_rather_than_an_empty_dict():
    """An empty dict and None mean different things downstream: the backend
    tests `max_tiles` truthiness to decide whether to pre-tile at all."""
    orchestrator = FakeOrchestrator()

    orchestrator.screen_batch(["/a.png"], "ask", 200)

    assert orchestrator.generate_calls[0][3] is None


def test_an_empty_image_list_calls_nothing():
    orchestrator = FakeOrchestrator()

    assert orchestrator.screen_batch([], "ask", 200) == []
    assert orchestrator.generate_calls == []
