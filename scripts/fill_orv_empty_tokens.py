from pathlib import Path
from typing import List, Optional

import conllu
import srsly
import typer
from wasabi import msg

TASK_TO_EMPTY_PREDS = {
    "lemmatisation": ["", ["", "", ""]],
    "morph_features": {"Form": "", "UPOS": ""},
    "pos_tagging": ["", ""],
}


def fill_orv_empty_tokens(
    submissions_dir: Path,
    reference_path: Path,
    output_path: Optional[Path] = None,
):
    """Add empty predictions for empty orv tokens"""
    tasks = list(TASK_TO_EMPTY_PREDS.keys())
    refs = [sent for sent in conllu.parse_incr(reference_path.open(encoding="utf-8"))]
    for task in tasks:
        preds = srsly.read_json(submissions_dir / task / "orv.json")
        for ref, pred in zip(refs, preds):
            ref_tokens = [token["form"] for token in ref]
            pred_tokens = [get_orth(token, task) for token in pred]


def get_orth(pred, task: str) -> str:
    if task not in TASK_TO_EMPTY_PREDS.keys():
        msg.fail(f"Unrecognized task: {task}", exits=1)
    if task == "lemmatisation":
        orth, _ = pred
    if task == "morph_features":
        orth = pred["Form"]
    if task == "pos_tagging":
        orth, _ = pred
    return orth


def validate(ref_tokens: List[str], pred_tokens: List[str]) -> bool:
    pass


if __name__ == "__main__":
    typer.run(fill_orv_empty_tokens)
