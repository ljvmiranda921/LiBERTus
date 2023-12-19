import concurrent.futures
import itertools
import random
import statistics
from collections import Counter
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional
from multiprocessing import Pool

import typer
from conllu import parse_incr
from tqdm import tqdm
from wasabi import msg


class SamplingStrategy(str, Enum):
    upsample = "upsample"
    average = "average"
    propn_augment = "propn_augment"


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
    pretraining = list(itertools.chain.from_iterable(examples.values()))
    msg.info(f"Found {len(pretraining)} examples in total.")

    if sampling_strategy:
        # Sample the files
        msg.text(f"Sampling using '{sampling_strategy.value}'")
        pretraining = sample_corpora(examples, sampling_strategy, verbose=verbose)

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


def sample_corpora(
    examples: Dict[str, List[str]],
    strategy: SamplingStrategy,
    verbose: bool = True,
) -> List[str]:
    """Sample the corpora based on some token statistics

    We want to sample sentences based on the number of unique tokens, not their raw
    number nor the number of sentences. Yes, increasing the number of sentences by X
    doesn't mean it will increase the number of tokens by the same proportion, however,
    we want to have a rough estimate.
    """

    def _count_tokens(lang: str, sents: List[str]) -> int:
        sep = None if lang == "lzh" else " "
        count = 0
        for sent in sents:
            # Count unique tokens only
            count += len(set(sent.split(sep)))
        return count

    # Get token counts
    # We want to sample sentences based on the number of unique tokens, not
    # just its raw value NOR the number of sentences.
    token_counts = Counter(
        {lang: _count_tokens(lang, sents) for lang, sents in examples.items()}
    )
    msg.text(
        title="Token counts",
        text=" ".join([f"{lang} ({count})" for lang, count in token_counts.items()]),
        show=verbose,
    )

    augmented_corpora: Dict[str, List[str]] = {}
    if strategy == SamplingStrategy.upsample:

        def _worker(args):
            lang, sents, most_common, token_counts = args
            num_tokens_to_add = int(
                most_common * (1 - (token_counts[lang] / most_common))
            )
            sents_to_add = []
            while True:
                sents_to_add.append(random.choice(sents))
                num_tokens_added = _count_tokens(lang, sents_to_add)
                if num_tokens_added >= num_tokens_to_add:
                    break

            msg.text(
                f"Adding {len(sents_to_add)} sentences ({num_tokens_added} tokens) to {lang}",
                show=verbose,
            )

            return lang, sents + sents_to_add

        msg.info("Using upsampling strategy")
        # Upsample: get percentage of each with respect to max and sample by that amount.
        _, most_common = token_counts.most_common()[0]

        args_list = [
            (lang, sents, most_common, token_counts) for lang, sents in examples.items()
        ]
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = executor.map(_worker, args_list)

        for lang, augmented_sents in results:
            augmented_corpora[lang] = augmented_sents

    if strategy == SamplingStrategy.average:
        msg.text("Using averaging strategy")
        # Average: get median and upsample or downsample based on that.
        median = statistics.median(token_counts.values())
        for lang, sents in examples.items():
            approach = "Downsampling" if token_counts[lang] >= median else "Upsampling"
            msg.text(f"{approach} '{lang}'...", show=verbose)
            new_sents = []
            while True:
                new_sents.append(random.choice(sents))
                num_tokens_added = _count_tokens(lang, new_sents)
                if num_tokens_added >= median:
                    break

            augmented_corpora[lang] = new_sents

    # Report the new token counts
    new_token_counts = Counter(
        {lang: _count_tokens(lang, sents) for lang, sents in augmented_corpora.items()}
    )
    msg.text(
        title="Token counts",
        text=" ".join(
            [f"{lang} ({count})" for lang, count in new_token_counts.items()]
        ),
        show=verbose,
    )

    pretraining = list(itertools.chain.from_iterable(augmented_corpora.values()))
    return pretraining


if __name__ == "__main__":
    typer.run(convert_to_pretrain)
