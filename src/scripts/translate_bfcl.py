"""Translate the BFCL-v2 dataset to different languages.

Usage:
    uv run src/scripts/translate_bfcl.py [--model MODEL] [--api-base API_BASE]
"""

import collections.abc as c
import random
import warnings
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path
from string import punctuation

import click
from datasets import DownloadConfig, disable_progress_bars, load_dataset
from dotenv import load_dotenv
from tqdm.auto import tqdm

from multi_bfcl.data_loading import load_bfcl, load_languages
from multi_bfcl.data_models import Example
from multi_bfcl.languages import Language
from multi_bfcl.translation import translate_example

load_dotenv()


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
            with language_output_path.open() as f:
                existing_examples = [
                    Example.model_validate_json(line)
                    for line in f.readlines()
                    if line.strip()
                ]
            existing_ids = {example.id for example in existing_examples}
            language_examples = [
                example
                for example in language_examples
                if example.id not in existing_ids
            ]

        dataset = load_dataset(
            "alexandrainst/multi-wiki-qa",
            name=language.code,
            split="train",
            download_config=DownloadConfig(disable_tqdm=True),
        )

        contexts = [row["context"] for row in dataset if row["context"]]
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

    executor = ThreadPoolExecutor(max_workers=concurrency)
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
        executor.shutdown(wait=True, cancel_futures=True)
        for future, example in pending.items():
            if not future.cancelled():
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
