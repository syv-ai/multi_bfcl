"""Offline tests for the BFCL translation command."""

import importlib.util
import threading
import time
from concurrent.futures import FIRST_COMPLETED, Future, wait
from pathlib import Path

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


def test_failures_do_not_prevent_successful_checkpoints(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A failed translation is reported without blocking other output."""

    def fake_translate(
        example: Example,
        language: Language,
        language_example: str,
        model: str,
        api_base: str | None,
    ) -> Example:
        if example.id == "example-1":
            raise RuntimeError("offline failure")
        return example

    monkeypatch.setattr(translate_bfcl, "translate_example", fake_translate)
    output_path = tmp_path / "output.jsonl"
    translate_bfcl._translate_examples(
        examples=_examples(3),
        contexts=["ordinary context"],
        language=Language(code="xx", name="Example"),
        output_path=output_path,
        model="offline",
        api_base="",
        concurrency=2,
    )
    ids = [
        Example.model_validate_json(line).id
        for line in output_path.read_text().splitlines()
    ]
    assert sorted(ids) == ["example-0", "example-2"]
    assert "example-1" in capsys.readouterr().err


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
