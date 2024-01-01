from pathlib import Path
from typing import Optional

import spacy
import typer
import srsly
from wasabi import msg


def run_pipeline(
    # fmt: off
    input_path: Path = typer.Argument(..., help="Path to the input test file."),
    output_dir: Path = typer.Argument(..., help="Directory to save the outputs"),
    model: str = typer.Argument(..., help="spaCy pipeline to use"),
    lang: Optional[str] = typer.Option(None, help="Language code of the file. If None, will infer from input_path"),
    # fmt: on
):
    """Run a pipeline on a file and then output it in a directory

    The output files follow the shared task's submission format:
    https://github.com/sigtyp/ST2024?tab=readme-ov-file#submission-format
    """

    if model not in spacy.util.get_installed_models():
        msg.fail(f"Model {model} not installed!", exits=1)
    nlp = spacy.load(model)

    # TODO: Get files from the input path
    texts = ["Ad astra per aspera", "Cogito, ergo sum", "Veni, vidi, vici"]

    docs = nlp.pipe(texts)


if __name__ == "__main__":
    typer.run(run_pipeline)
