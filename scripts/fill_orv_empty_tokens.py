from pathlib import Path
from typing import Any, List, Optional

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
        new_preds = []
        for idx, (ref, pred) in enumerate(zip(refs, preds)):
            ref_tokens = [token["form"] for token in ref]
            pred_tokens = [get_orth(token, task) for token in pred]
            if not is_equal(ref_tokens, pred_tokens):
                msg.divider(f"Sentence {idx} not equal!")
                new_pred = fix_tokens(pred, task=task, ref_tokens=ref_tokens)
                assert is_equal(
                    ref_tokens, [get_orth(token, task) for token in new_pred]
                ), "Still not fixed!"
                new_preds.append(new_pred)


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


def is_equal(ref_tokens: List[str], pred_tokens: List[str]) -> bool:
    parity = True
    if len(ref_tokens) != len(pred_tokens):
        msg.warn(
            "Unequal lengths for ref and preds! "
            f"{len(ref_tokens)} != {len(pred_tokens)}"
        )
        parity = False

    for idx, (ref_tok, pred_tok) in enumerate(zip(ref_tokens, pred_tokens)):
        if ref_tok != pred_tok:
            msg.warn(f"Position {idx} tokens unequal: {ref_tok} != {pred_tok}")

    return parity


def fix_tokens(pred: List[Any], task: str, ref_tokens: List[str]) -> List[Any]:
    pass


if __name__ == "__main__":
    typer.run(fill_orv_empty_tokens)
