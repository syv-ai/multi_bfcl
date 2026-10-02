"""Translate the BFCL-v2 dataset to different languages.

Usage:
    uv run src/scripts/translate_bfcl.py [--model MODEL] [--api-base API_BASE]
"""

import collections.abc as c
import queue
import random
import threading
import typing as t
import warnings
from concurrent.futures import FIRST_COMPLETED, Future, wait
from pathlib import Path
from string import punctuation

import click
from datasets import Dataset, DownloadConfig, disable_progress_bars, load_dataset
from dotenv import load_dotenv
from tqdm.auto import tqdm

from multi_bfcl.data_loading import load_bfcl, load_languages
from multi_bfcl.data_models import Example
from multi_bfcl.languages import Language
from multi_bfcl.translation import translate_example

load_dotenv()


class _DaemonExecutor:
    """Small executor whose in-flight network calls cannot block process exit."""

    def __init__(self, max_workers: int) -> None:
        self._tasks: queue.Queue[
            tuple[Future[Example], c.Callable[[], Example]] | None
        ] = queue.Queue()
        self._closed = False
        self._lock = threading.Lock()
        self._workers = [
            threading.Thread(target=self._work, daemon=True) for _ in range(max_workers)
        ]
        for worker in self._workers:
            worker.start()

    def submit(
        self,
        function: c.Callable[[Example, str], Example],
        example: Example,
        context: str,
    ) -> Future[Example]:
        """Schedule a translation on a daemon worker.

        Returns:
            The future for the scheduled translation.

        Raises:
            RuntimeError:
                If the executor has been shut down.
        """
        with self._lock:
            if self._closed:
                raise RuntimeError("Cannot submit to a shut down executor")
            future: Future[Example] = Future()
            self._tasks.put((future, lambda: function(example, context)))
            return future

    def shutdown(self, wait: bool, cancel_futures: bool = False) -> None:
        """Stop accepting work and optionally wait for workers to finish."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            if cancel_futures:
                while True:
                    try:
                        item = self._tasks.get_nowait()
                    except queue.Empty:
                        break
                    if item is not None:
                        item[0].cancel()
            for _ in self._workers:
                self._tasks.put(None)
        if wait:
            for worker in self._workers:
                worker.join()

    def _work(self) -> None:
        while True:
            task = self._tasks.get()
            if task is None:
                return
            future, function = task
            if not future.set_running_or_notify_cancel():
                continue
            try:
                future.set_result(function())
            except BaseException as error:
                future.set_exception(error)


def _load_checkpoint(path: Path) -> list[Example]:
    """Load a checkpoint and discard only an invalid, unterminated last record.

    Args:
        path:
            JSONL checkpoint path.

    Returns:
        Valid examples from the checkpoint.

    Raises:
        ValueError:
            If any complete record is malformed.
    """
    content = path.read_bytes()
    lines = content.splitlines(keepends=True)
    examples: list[Example] = []
    offset = 0
    for index, line in enumerate(lines):
        is_final_partial = index == len(lines) - 1 and not line.endswith(b"\n")
        try:
            example = Example.model_validate_json(line)
        except ValueError:
            if not is_final_partial:
                raise
            path.write_bytes(content[:offset])
            break
        examples.append(example)
        offset += len(line)
    else:
        if content and not content.endswith(b"\n"):
            with path.open("ab") as checkpoint:
                checkpoint.write(b"\n")
    return examples


@click.command()
@click.option(
    "--model",
    "-m",
    type=str,
    default="gemini/gemini-3.1-flash-lite-preview",
    help="The model to use for translation.",
)
@click.option(
    "--api-base",
    type=str,
    default=None,
    help="The base URL of the API to use for translation.",
)
@click.option(
    "--concurrency",
    type=click.IntRange(min=1),
    default=1,
    show_default=True,
    help="Maximum simultaneous translations per language.",
)
def main(model: str, api_base: str, concurrency: int) -> None:
    """Translate the BFCL-v2 dataset to different languages."""
    disable_progress_bars()
    warnings.filterwarnings("ignore", category=UserWarning)

    output_dir = Path("data")
    output_dir.mkdir(exist_ok=True)

    examples = load_bfcl()

    for language in tqdm(
        iterable=load_languages(), desc="Translating datasets", unit="dataset"
    ):
        language_examples = list(examples)

        language_output_path = output_dir / f"bfcl-{language.code}.jsonl"
        if language_output_path.exists():
            existing_examples = _load_checkpoint(language_output_path)
            existing_ids = {example.id for example in existing_examples}
            language_examples = [
                example
                for example in language_examples
                if example.id not in existing_ids
            ]

        dataset = t.cast(
            Dataset,
            load_dataset(
                "alexandrainst/multi-wiki-qa",
                name=language.code,
                split="train",
                download_config=DownloadConfig(disable_tqdm=True),
            ),
        )

        contexts: list[str] = []
        for index in range(len(dataset)):
            row = t.cast(dict[str, str], dataset[index])
            if row["context"]:
                contexts.append(row["context"])
        random.shuffle(contexts)
        _translate_examples(
            examples=language_examples,
            contexts=contexts,
            language=language,
            output_path=language_output_path,
            model=model,
            api_base=api_base,
            concurrency=concurrency,
        )


def _translate_examples(
    examples: list[Example],
    contexts: list[str],
    language: Language,
    output_path: Path,
    model: str,
    api_base: str,
    concurrency: int,
) -> None:
    """Translate examples while keeping all checkpoint writes on this thread.

    Raises:
        KeyboardInterrupt:
            When the user interrupts translation; completed calls are checkpointed.
    """
    if not examples:
        return
    if not contexts:
        warnings.warn(f"No contexts available for {language.name}; skipping language.")
        return

    context_iterator = _usable_contexts(contexts)
    progress = tqdm(
        iterable=zip(examples, context_iterator),
        total=len(examples),
        desc=f"Translating examples to {language.name}",
        unit="example",
        leave=False,
    )

    def translate(example: Example, context: str) -> Example:
        return translate_example(
            example=example,
            language=language,
            language_example=context,
            model=model,
            api_base=api_base,
        )

    def save(future: Future[Example], example: Example) -> None:
        try:
            translated = future.result()
        except Exception as error:
            click.echo(
                f"Failed to translate example {example.id} to {language.name}. "
                f"Skipping. Here are the errors that occurred:\n{error}",
                err=True,
            )
            return
        with output_path.open("a") as output_file:
            output_file.write(translated.model_dump_json() + "\n")

    if concurrency == 1:
        for example, context in progress:
            try:
                translated = translate(example, context)
            except Exception as error:
                click.echo(
                    f"Failed to translate example {example.id} to {language.name}. "
                    f"Skipping. Here are the errors that occurred:\n{error}",
                    err=True,
                )
                continue
            with output_path.open("a") as output_file:
                output_file.write(translated.model_dump_json() + "\n")
        return

    executor = _DaemonExecutor(max_workers=concurrency)
    pending: dict[Future[Example], Example] = {}
    iterator = iter(progress)
    try:
        while True:
            while len(pending) < concurrency:
                try:
                    example, context = next(iterator)
                except StopIteration:
                    break
                pending[executor.submit(translate, example, context)] = example
            if not pending:
                break
            completed, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in completed:
                example = pending.pop(future)
                save(future, example)
    except KeyboardInterrupt:
        for future in pending:
            future.cancel()
        executor.shutdown(wait=False, cancel_futures=True)
        for future, example in pending.items():
            if future.done() and not future.cancelled():
                save(future, example)
        raise
    finally:
        executor.shutdown(wait=False, cancel_futures=True)


def _usable_contexts(contexts: list[str]) -> c.Iterator[str]:
    """Yield varied contexts, preferring examples with little punctuation."""
    pool = list(contexts)
    random.shuffle(pool)
    position = 0
    while True:
        if position >= len(pool):
            random.shuffle(pool)
            position = 0
        best_context = ""
        best_fraction = float("inf")
        for _ in range(min(11, len(pool) - position)):
            context = pool[position]
            position += 1
            fraction = sum(char in punctuation for char in context) / len(context)
            if fraction < best_fraction:
                best_context = context
                best_fraction = fraction
            if fraction < 0.05:
                break
        yield best_context


if __name__ == "__main__":
    main()
