"""Offline tests for the BFCL translation command."""

import importlib.util
import threading
import time
from concurrent.futures import ALL_COMPLETED, FIRST_COMPLETED, Future, wait
from pathlib import Path

import litellm
import pytest
from click.testing import CliRunner

from multi_bfcl.data_models import Example
from multi_bfcl.languages import Language

SCRIPT_PATH = Path(__file__).parents[1] / "src" / "scripts" / "translate_bfcl.py"
SPEC = importlib.util.spec_from_file_location("translate_bfcl", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
translate_bfcl = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(translate_bfcl)


def _examples(count: int) -> list[Example]:
    return [
        Example(id=f"example-{index}", question=[], function=[], ground_truth=[])
        for index in range(count)
    ]


def _run_translations(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, concurrency: int
) -> tuple[int, list[str]]:
    lock = threading.Lock()
    active = 0
    maximum_active = 0

    def fake_translate(
        example: Example,
        language: Language,
        language_example: str,
        model: str,
        api_base: str | None,
    ) -> Example:
        nonlocal active, maximum_active
        with lock:
            active += 1
            maximum_active = max(maximum_active, active)
        time.sleep(0.03)
        with lock:
            active -= 1
        return example

    monkeypatch.setattr(translate_bfcl, "translate_example", fake_translate)
    path = tmp_path / "output.jsonl"
    translate_bfcl._translate_examples(
        examples=_examples(5),
        contexts=["Ordinary text without any punctuation."],
        language=Language(code="xx", name="Example"),
        output_path=path,
        model="offline",
        api_base="",
        concurrency=concurrency,
    )
    ids = [
        Example.model_validate_json(line).id for line in path.read_text().splitlines()
    ]
    return maximum_active, ids


def test_default_concurrency_is_sequential(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The default worker count preserves serial translation."""
    maximum_active, ids = _run_translations(monkeypatch, tmp_path, concurrency=1)
    assert maximum_active == 1
    assert len(ids) == 5
    assert len(set(ids)) == 5
    help_result = CliRunner().invoke(translate_bfcl.main, ["--help"])
    assert "[default: 1; x>=1]" in help_result.output


def test_concurrency_runs_multiple_translations(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Configured workers translate multiple examples simultaneously."""
    maximum_active, ids = _run_translations(monkeypatch, tmp_path, concurrency=3)
    assert 2 <= maximum_active <= 3
    assert len(ids) == 5
    assert len(set(ids)) == 5


@pytest.mark.parametrize("concurrency", [1, 20])
def test_transient_translation_failures_are_retried(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, concurrency: int
) -> None:
    """Transient outages recover in sequential and worker-pool modes."""
    attempts: dict[str, int] = {}

    def fake_translate(
        example: Example,
        language: Language,
        language_example: str,
        model: str,
        api_base: str | None,
    ) -> Example:
        attempts[example.id] = attempts.get(example.id, 0) + 1
        if attempts[example.id] == 1:
            raise RuntimeError("fetch failed")
        return example

    monkeypatch.setattr(translate_bfcl, "translate_example", fake_translate)
    monkeypatch.setattr(translate_bfcl.time, "sleep", lambda delay: None)
    output_path = tmp_path / "output.jsonl"
    translate_bfcl._translate_examples(
        examples=_examples(3),
        contexts=["ordinary context"],
        language=Language(code="xx", name="Example"),
        output_path=output_path,
        model="offline",
        api_base="",
        concurrency=concurrency,
    )
    ids = [
        Example.model_validate_json(line).id
        for line in output_path.read_text().splitlines()
    ]
    assert sorted(ids) == ["example-0", "example-1", "example-2"]
    assert set(attempts.values()) == {2}


@pytest.mark.parametrize(
    ("status_code", "retryable"), [(499, True), (429, True), (400, False)]
)
def test_proxy_error_status_classification(status_code: int, retryable: bool) -> None:
    """A proxy-aborted request is retryable, unlike a bad request."""
    error = litellm.APIError(
        status_code=status_code,
        message="Request was aborted",
        llm_provider="openai",
        model="gpt-6-sol",
    )
    assert translate_bfcl._is_transient_translation_error(error) is retryable


def test_permanent_translation_failure_aborts_and_keeps_checkpoint(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A permanent error stops the run while preserving successful rows."""
    calls: list[str] = []

    def fake_translate(
        example: Example,
        language: Language,
        language_example: str,
        model: str,
        api_base: str | None,
    ) -> Example:
        calls.append(example.id)
        if example.id == "example-1":
            raise ValueError("invalid request")
        return example

    monkeypatch.setattr(translate_bfcl, "translate_example", fake_translate)
    output_path = tmp_path / "output.jsonl"
    with pytest.raises(RuntimeError, match="example-1"):
        translate_bfcl._translate_examples(
            examples=_examples(3),
            contexts=["ordinary context"],
            language=Language(code="xx", name="Example"),
            output_path=output_path,
            model="offline",
            api_base="",
            concurrency=1,
        )
    assert calls == ["example-0", "example-1"]
    assert [
        Example.model_validate_json(line).id
        for line in output_path.read_text().splitlines()
    ] == ["example-0"]


def test_concurrent_failure_checkpoints_completed_successes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A failed future does not discard other already-completed results."""

    def fake_translate(
        example: Example,
        language: Language,
        language_example: str,
        model: str,
        api_base: str | None,
    ) -> Example:
        if example.id == "example-0":
            raise ValueError("invalid request")
        return example

    def report_failure_first(
        futures: set[Future[Example]], return_when: str = FIRST_COMPLETED
    ) -> tuple[set[Future[Example]], set[Future[Example]]]:
        done, _ = wait(futures, return_when=ALL_COMPLETED)
        failures = {future for future in done if future.exception() is not None}
        return failures, done - failures

    monkeypatch.setattr(translate_bfcl, "translate_example", fake_translate)
    monkeypatch.setattr(translate_bfcl, "wait", report_failure_first)
    output_path = tmp_path / "output.jsonl"
    with pytest.raises(RuntimeError, match="example-0"):
        translate_bfcl._translate_examples(
            examples=_examples(3),
            contexts=["ordinary context"],
            language=Language(code="xx", name="Example"),
            output_path=output_path,
            model="offline",
            api_base="",
            concurrency=3,
        )
    ids = [
        Example.model_validate_json(line).id
        for line in output_path.read_text().splitlines()
    ]
    assert sorted(ids) == ["example-1", "example-2"]


def test_resume_skips_existing_ids(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Existing checkpoint identifiers are not submitted again."""
    monkeypatch.chdir(tmp_path)
    existing = _examples(1)[0]
    output_path = tmp_path / "data" / "bfcl-xx.jsonl"
    output_path.parent.mkdir()
    output_path.write_text(existing.model_dump_json() + "\n")
    monkeypatch.setattr(translate_bfcl, "load_bfcl", lambda: _examples(2))
    monkeypatch.setattr(
        translate_bfcl, "load_languages", lambda: [Language("xx", "Example")]
    )
    monkeypatch.setattr(
        translate_bfcl,
        "load_dataset",
        lambda *args, **kwargs: [{"context": "ordinary language context"}],
    )
    translated: list[str] = []

    def fake_translate(
        example: Example,
        language: Language,
        language_example: str,
        model: str,
        api_base: str | None,
    ) -> Example:
        translated.append(example.id)
        return example

    monkeypatch.setattr(translate_bfcl, "translate_example", fake_translate)
    result = CliRunner().invoke(translate_bfcl.main, [])
    assert result.exit_code == 0, result.output
    assert translated == ["example-1"]
    ids = [
        Example.model_validate_json(line).id
        for line in output_path.read_text().splitlines()
    ]
    assert ids == ["example-0", "example-1"]


def test_language_dataset_cache_error_is_retried(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Hugging Face's cache ValueError is retried as a possible outage."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(translate_bfcl, "load_bfcl", lambda: _examples(1))
    monkeypatch.setattr(
        translate_bfcl, "load_languages", lambda: [Language("xx", "Example")]
    )
    monkeypatch.setattr(translate_bfcl.time, "sleep", lambda delay: None)
    calls = 0

    def fake_load_dataset(*args: object, **kwargs: object) -> list[dict[str, str]]:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ValueError("Couldn't find cache for dataset config 'xx'")
        return [{"context": "ordinary context"}]

    monkeypatch.setattr(translate_bfcl, "load_dataset", fake_load_dataset)
    monkeypatch.setattr(
        translate_bfcl, "translate_example", lambda example, **kwargs: example
    )
    result = CliRunner().invoke(translate_bfcl.main, [])
    assert result.exit_code == 0, result.output
    assert calls == 2


def test_completed_language_skips_dataset_loading(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A complete checkpoint does not need a language dataset download."""
    monkeypatch.chdir(tmp_path)
    output_path = tmp_path / "data" / "bfcl-xx.jsonl"
    output_path.parent.mkdir()
    output_path.write_text(_examples(1)[0].model_dump_json() + "\n")
    monkeypatch.setattr(translate_bfcl, "load_bfcl", lambda: _examples(1))
    monkeypatch.setattr(
        translate_bfcl, "load_languages", lambda: [Language("xx", "Example")]
    )

    def fail_if_loaded(*args: object, **kwargs: object) -> None:
        raise AssertionError("dataset should not be loaded for a completed language")

    monkeypatch.setattr(translate_bfcl, "load_dataset", fail_if_loaded)
    result = CliRunner().invoke(translate_bfcl.main, [])
    assert result.exit_code == 0, result.output


def test_resume_discards_only_invalid_partial_final_record(tmp_path: Path) -> None:
    """A killed final write is removed while a valid unterminated tail survives."""
    valid = _examples(1)[0].model_dump_json()
    partial_path = tmp_path / "partial.jsonl"
    partial_path.write_text(valid + "\n{")

    loaded = translate_bfcl._load_checkpoint(partial_path)

    assert [example.id for example in loaded] == ["example-0"]
    assert partial_path.read_text() == valid + "\n"

    unterminated_path = tmp_path / "unterminated.jsonl"
    unterminated_path.write_text(valid)
    loaded = translate_bfcl._load_checkpoint(unterminated_path)

    assert [example.id for example in loaded] == ["example-0"]
    assert unterminated_path.read_text() == valid + "\n"


def test_resume_rejects_malformed_non_final_record(tmp_path: Path) -> None:
    """Malformed records before the final line are not silently discarded."""
    path = tmp_path / "malformed.jsonl"
    path.write_text("{\n" + _examples(1)[0].model_dump_json() + "\n")

    with pytest.raises(ValueError):
        translate_bfcl._load_checkpoint(path)


def test_interrupt_does_not_wait_for_inflight_translations(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Interrupt returns promptly and retains results already checkpointed."""
    output_path = tmp_path / "translations.jsonl"
    blocked = threading.Event()
    calls: list[str] = []

    def fake_translate(
        example: Example,
        language: Language,
        language_example: str,
        model: str,
        api_base: str,
    ) -> Example:
        calls.append(example.id)
        if example.id == "example-0":
            return example
        blocked.wait()
        return example

    monkeypatch.setattr(translate_bfcl, "translate_example", fake_translate)
    real_wait = wait
    wait_calls = 0

    def interrupt_after_checkpoint(
        futures: set[Future[Example]], return_when: str = FIRST_COMPLETED
    ) -> tuple[set[Future[Example]], set[Future[Example]]]:
        nonlocal wait_calls
        wait_calls += 1
        if wait_calls == 1:
            return real_wait(futures, return_when=return_when)
        raise KeyboardInterrupt

    monkeypatch.setattr(translate_bfcl, "wait", interrupt_after_checkpoint)
    started = time.monotonic()
    try:
        with pytest.raises(KeyboardInterrupt):
            translate_bfcl._translate_examples(
                examples=_examples(5),
                contexts=["context"],
                language=Language("xx", "Example"),
                output_path=output_path,
                model="offline",
                api_base="offline",
                concurrency=2,
            )
        assert time.monotonic() - started < 1
        saved_ids = [
            Example.model_validate_json(line).id
            for line in output_path.read_text().splitlines()
        ]
        assert saved_ids == ["example-0"]
        assert len(calls) <= 3
    finally:
        blocked.set()


def test_concurrency_must_be_positive() -> None:
    """The CLI rejects a zero worker count."""
    result = CliRunner().invoke(translate_bfcl.main, ["--concurrency", "0"])
    assert result.exit_code != 0
    assert "0 is not in the range x>=1" in result.output
