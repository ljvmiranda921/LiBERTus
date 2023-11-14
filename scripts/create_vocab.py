from pathlib import Path

import typer
from wasabi import msg
from tokenizers.implementations import ByteLevelBPETokenizer


def create_vocab(
    # fmt: off
    input_path: Path = typer.Argument(..., help="Path to the pretraining corpus."),
    output_dir: Path = typer.Argument(..., dir_okay=True, help="Directory to save the output files."),
    # fmt: on
):
    """Train a tokenizer to create a vocabulary"""

    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(
        files=str(input_path),
        vocab_size=52000,
        min_frequency=2,
        special_tokens=[
            "<s>",
            "<pad>",
            "</s>",
            "<unk>",
            "<mask>",
        ],
    )
    tokenizer.save_model(str(output_dir))
    msg.good(f"Saved tokenizer training artifacts to {output_dir}")


if __name__ == "__main__":
    typer.run(create_vocab)
