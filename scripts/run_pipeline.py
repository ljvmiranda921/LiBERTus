from pathlib import Path
from typing import List, Optional

import spacy
import srsly
import typer
from spacy.tokens import DocBin
from wasabi import msg


def run_pipeline(
    # fmt: off
    input_path: Path = typer.Argument(..., help="Path to the input spaCy file."),
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

    doc_bin = DocBin().from_disk(input_path)
    _docs = doc_bin.get_docs(nlp.vocab)
    docs = nlp.pipe(_docs)

    results = {"pos_tagging": [], "morph_features": [], "lemmatisation": []}
    for doc in docs:
        sentence = {"pos": [], "morph": [], "lemma": []}
        for token in doc:
            # Add POS-tagging results
            sentence["pos"].append((token.text, token.pos_))
            # Add morphological analysis results
            morphs = token.morph.to_dict()
            morphs["UPOS"] = token.pos_
            morphs["Form"] = token.orth_
            sentence["morph"].append(morphs)
            # Add lemmatization results
            lemma = [token.text, [token.lemma_, "", ""]]
            sentence["lemma"].append(lemma)

        # Append each sentence
        results["pos_tagging"].append(sentence["pos"])
        results["morph_features"].append(sentence["morph"])
        results["lemmatisation"].append(sentence["lemma"])

    # Save results
    lang_code = lang if lang else input_path.stem.split("_")[0]
    for task, outputs in results.items():
        task_dir = output_dir / task
        task_dir.mkdir(parents=True, exist_ok=True)
        output_path = task_dir / f"{lang_code}.json"
        srsly.write_json(output_path, outputs)
        msg.good(f"Saved file to {output_path}!")


if __name__ == "__main__":
    typer.run(run_pipeline)
