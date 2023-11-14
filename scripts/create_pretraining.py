import random
from pathlib import Path
from typing import List

import typer
from conllu import parse_incr
from tqdm import tqdm
from wasabi import msg


def create_pretraining(
    # fmt: off
    sources: List[Path] = typer.Argument(..., help="Directories to source the CoNLL-U files."),
    output_path: Path = typer.Option(Path("corpus/pretraining.txt"), "--output-path", "-o", help="Path to save the pretraining corpus."),
    shuffle: bool = typer.Option(False, "--shuffle", "-s", help="Shuffle the examples before saving to disk."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show additional information."),
    seed: int = typer.Option(42, "--seed", help="Set random seed for shuffling."),
    # fmt: on
):
    """Create corpus for multilingual LM pretraining

    This script reads all the CoNLL-U files from the specified `sources`, and then
    compile them together in a single .txt file.
    """
    files: List[Path] = []
    for source in sources:
        files.extend(list(source.glob("*.conllu")))
    msg.info(f"Found {len(files)} files from source/s.")

    examples = []
    for file in files:
        egs = get_examples(file)
        msg.text(f"Found {len(egs)} examples in '{file}'", show=verbose)
        examples.extend(egs)
    msg.info(f"Found {len(examples)} examples in total.")

    if shuffle:
        msg.info(f"Shuffling examples using seed '{seed}'")
        random.seed(seed)
        random.shuffle(examples)

    with open(output_path, "w") as f:
        for line in tqdm(examples):
            f.write(f"{line}\n")
    msg.good(f"Saved output to '{str(output_path)}'")


def get_examples(file: Path) -> List[str]:
    """Get examples from the provided ConLL-u file."""
    examples = []
    with open(file, "r", encoding="utf-8") as f:
        for conllu_instance in parse_incr(f):
            text = "".join([token.get("form") for token in conllu_instance])
            examples.append(text)
    return examples


if __name__ == "__main__":
    typer.run(create_pretraining)
