from enum import Enum
from pathlib import Path
from typing import Dict, List

import srsly
import typer
import numpy as np
from wasabi import msg

from .constants import COMPONENT_TO_METRIC


class Split(str, Enum):
    dev = "dev"
    test = "test"


def read_scores(
    scores_dir: Path,
    split: Split = typer.Option(Split.dev, help="Dataset split to plot."),
):
    """Print out scores"""

    languages = sorted([dir.stem for dir in scores_dir.iterdir() if dir.is_dir()])
    msg.text(f"Found languages: {', '.join(lang for lang in languages)}")

    # Get score filepaths
    scores_fp = [
        scores_dir / lang / f"metrics-{lang}-{split.value}.json" for lang in languages
    ]

    # Get actual scores
    component_to_rects: Dict[str, List[float]] = {
        "tagger": [],
        "morphologizer": [],
        "trainable_lemmatizer": [],
    }

    components = COMPONENT_TO_METRIC.keys()

    def _get_score(scores_dict, component: str) -> float:
        score = (
            scores_dict.get("morphologizer").get("pos_acc")
            if component == "tagger"
            else scores_dict.get(component).get(COMPONENT_TO_METRIC[component])
        )
        return round(score, 3)

    for fp in scores_fp:
        scores = srsly.read_json(fp)
        for component in components:
            component_to_rects[component].append(_get_score(scores, component))

    # Format for table
    keys = component_to_rects.keys()
    table_data = [scores for scores in zip(*(component_to_rects[key] for key in keys))]
    msg.info(f"Scores in the {split.value} split")
    msg.table(data=table_data, header=list(keys))

    msg.info("Top-3 performers (average of all tasks)")
    avg_scores = sorted(
        [(lang, np.mean(scores)) for lang, scores in zip(languages, table_data)],
        key=lambda x: x[1],
        reverse=True,
    )
    msg.text(
        ", ".join([f"{lang} ({round(score,3)})" for lang, score in avg_scores[:3]])
    )
    msg.info("Bottom-3 performers (average of all tasks)")
    msg.text(
        ", ".join([f"{lang} ({round(score,3)})" for lang, score in avg_scores[-3:]])
    )
    msg.info("Worst than random chance")
    msg.text(
        ", ".join(
            [f"{lang} ({round(score,3)})" for lang, score in avg_scores if score <= 0.5]
        )
    )


if __name__ == "__main__":
    typer.run(read_scores)
