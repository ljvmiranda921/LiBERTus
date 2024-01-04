from pathlib import Path
from typing import Optional

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


def check_inconsistencies(
    predictions_dir: Path,
    reference_dir: Path,
    languages: Optional[str] = None,
    verbose: bool = False,
):
    """Check inconsistencies between your predictions and the reference files"""
    if languages:
        langs = languages.split(",")
    else:
        langs = LANGUAGES
    for lang in langs:
        sep = "" if lang == "lzh" else " "
        msg.divider(f"Checking {lang}")
        refs = [
            sentence
            for sentence in conllu.parse_incr(
                (reference_dir / f"{lang}_test.conllu").open(encoding="utf-8")
            )
        ]
        for task in TASK_NAMES:
            preds = srsly.read_json(predictions_dir / task / f"{lang}.json")
            assert len(refs) == len(preds), "Length of refs and preds not the same!"

            for idx, (ref, pred) in enumerate(zip(refs, preds)):
                # Check if token lengths are the same
                ref_text = sep.join([token["form"] for token in ref])
                if len(ref) != len(pred):
                    msg.warn(
                        f"Unequal lengths ({len(ref)} != {len(pred)}) id={idx+1} text={ref_text}"
                    )
                for idx, (tok_ref, tok_pred) in enumerate(zip(ref, pred)):
                    orth_ref = tok_ref["form"]
                    if task == "morph_features":
                        orth_pred = tok_pred["Form"]
                    elif task == "lemmatisation":
                        orth_pred, _ = tok_pred
                    else:  # pos_tagging
                        orth_pred, _ == tok_pred
                    if orth_ref != orth_pred:
                        msg.text(
                            f"Mismatch tokens id={idx+1}, ref={orth_ref}, pred={orth_pred}",
                            show=verbose,
                        )
                        break


if __name__ == "__main__":
    typer.run(check_inconsistencies)
