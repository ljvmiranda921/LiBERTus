import itertools
import random
from collections import Counter
from enum import Enum
from pathlib import Path
from typing import List, Optional

import typer
from conllu import parse_incr
from tqdm import tqdm
from wasabi import msg


class SamplingStrategy(str, Enum):
    upsample = "upsample"
    average = "average"


def convert_to_pretrain(
    # fmt: off
    sources: List[Path] = typer.Argument(..., help="Directories to source the CoNLL-U files."),
    output_path: Path = typer.Option(Path("corpus/pretraining.txt"), "--output-path", "-o", help="Path to save the pretraining corpus."),
    shuffle: bool = typer.Option(False, "--shuffle", "-s", help="Shuffle the examples before saving to disk."),
    sampling_strategy: Optional[SamplingStrategy] = typer.Option(None, "--sampling-strategy", "-S", help="Sampling strategy to use."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show additional information."),
    seed: int = typer.Option(42, "--seed", help="Set random seed for shuffling."),
    # fmt: on
):
    """Create corpus for multilingual LM pretraining

    This script reads all the CoNLL-U files from the specified `sources`, and then
    compile them together in a single .txt file.
    """
    random.seed(seed)

    files: List[Path] = []
    for source in sources:
        files.extend(list(source.glob("*.conllu")))
    msg.info(f"Found {len(files)} files from source/s.")

    examples = {file.stem.split("_")[0]: get_examples(file) for file in tqdm(files)}
    pretraining = list(itertools.chain.from_iterable(examples))
    msg.info(f"Found {len(pretraining)} examples in total.")

    if sampling_strategy:
        # Sample the files
        msg.text(f"Sampling using '{sampling_strategy.value}'")
        pretraining = sample_corpora(pretraining, strategy=sampling_strategy)

    if shuffle:
        msg.info(f"Shuffling examples using seed '{seed}'")
        random.shuffle(pretraining)

    with open(output_path, "w") as f:
        for line in tqdm(pretraining):
            f.write(f"{line}\n")
    msg.good(f"Saved output to '{str(output_path)}'")


def get_examples(file: Path) -> List[str]:
    """Get examples from the provided ConLL-u file."""
    examples = []
    with open(file, "r", encoding="utf-8") as f:
        for conllu_instance in parse_incr(f):
            sep = " " if file.stem.split("_")[0] != "lzh" else ""
            text = sep.join([token.get("form") for token in conllu_instance])
            examples.append(text)
    return examples


def sample_corpora(pretraining: List[str], strategy: SamplingStrategy) -> List[str]:
    return pretraining


if __name__ == "__main__":
    typer.run(convert_to_pretrain)
