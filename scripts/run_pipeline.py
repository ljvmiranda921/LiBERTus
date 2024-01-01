from pathlib import Path
from typing import Optional

import typer
from wasabi import msg


def run_pipeline(
    # fmt: off
    input_path: Path = typer.Argument(..., help="Path to the input test file."),
    output_dir: Path = typer.Argument(..., help="Directory to save the outputs"),
    model: str = typer.Argument(..., help="spaCy pipeline to use"),
    lang: Optional[str] = typer.Option(None, help="Language code of the file. If None, will infer from input_path"),
    # fmt: on
):
    """Make a submissio"""
    pass


if __name__ == "__main__":
    typer.run(run_pipeline)
