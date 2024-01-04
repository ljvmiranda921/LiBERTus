from pathlib import Path

import spacy
import typer
from spacy.tokens import DocBin
import conllu
import srsly
from wasabi import msg

LANGUAGES = [
    "chu",
    "cop",
    "fro",
    "got",
    "grc",
    "hbo",
    "isl",
    "lat",
    "latm",
    "lzh",
    "ohu",
    "orv",
    "san",
]

TASK_NAMES = ["lemmatisation", "pos_tagging", "morph_features"]


def check_inconsistencies(predictions_dir: Path, reference_dir: Path):
    """Check inconsistencies between your predictions and the reference files"""
    for lang in LANGUAGES:
        refs = [
            sentence
            for sentence in conllu.parse_incr(
                (reference_dir / f"{lang}.conllu").open(encoding="utf-8")
            )
        ]
        for task in TASK_NAMES:
            preds = srsly.read_json(predictions_dir / task / f"{lang}.json")


if __name__ == "__main__":
    typer.run(check_inconsistencies)
