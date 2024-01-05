import json
from enum import Enum
from pathlib import Path
from typing import List, Optional, Tuple

import conllu
import spacy
import typer
from spacy.cli._util import setup_gpu
from spacy.tokens import Doc, DocBin
from spacy.util import get_words_and_spaces
from tqdm import tqdm
from wasabi import msg


class MWE(str, Enum):
    all: str = "all"
    subtokens_only: str = "subtokens_only"
    mwe_only: str = "mwe_only"


def run_pipeline(
    # fmt: off
    input_path: Path = typer.Argument(..., help="Path to the input CoNLL-U file."),
    output_dir: Path = typer.Argument(..., help="Directory to save the outputs."),
    model: str = typer.Argument(..., help="spaCy pipeline to use."),
    lang: Optional[str] = typer.Option(None, help="Language code of the file. If None, will infer from input_path."),
    save_preds_path: Optional[Path] = typer.Option(None, help="Optional path to save the predictions as a spaCy file."),
    use_gpu: int = typer.Option(-1, "--gpu-id", "-g", help="GPU ID or -1 for CPU."),
    n_process: int = typer.Option(1, "--n-process", "-n", help="Number of processors to use. Doesn't work for GPUs.")
    # fmt: on
):
    """Run a pipeline on a file and then output it in a directory

    The output files follow the shared task's submission format:
    https://github.com/sigtyp/ST2024?tab=readme-ov-file#submission-format
    """
    setup_gpu(use_gpu)

    if model not in spacy.util.get_installed_models():
        msg.fail(f"Model {model} not installed!", exits=1)
    nlp = spacy.load(model)
    lang_code = lang if lang else input_path.stem.split("_")[0]
    msg.good(f"Loaded '{model}' for lang code '{lang_code}'")

    raw_docs = []
    for sentence in conllu.parse_incr(input_path.open(encoding="utf-8")):
        words = [token.get("form") for token in sentence if token.get("form")]
        doc = Doc(nlp.vocab, words=words)
        raw_docs.append(doc)

    docs = nlp.pipe(raw_docs, n_process=n_process)

    results = {"pos_tagging": [], "morph_features": [], "lemmatisation": []}
    for doc in tqdm(docs):
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
            lemma = [token.text, [token.lemma_.strip(), "", ""]]
            sentence["lemma"].append(lemma)

        # Append each sentence
        results["pos_tagging"].append(sentence["pos"])
        results["morph_features"].append(sentence["morph"])
        results["lemmatisation"].append(sentence["lemma"])

    # Save results
    for task, outputs in results.items():
        task_dir = output_dir / task
        task_dir.mkdir(parents=True, exist_ok=True)
        output_path = task_dir / f"{lang_code}.json"
        msg.text(f"Saving outputs ({len(outputs)} docs) for {task} in {output_path}...")

        with open(output_path, "w", encoding="utf-8") as file:
            json.dump(outputs, file, ensure_ascii=False, indent=2)

        msg.good(f"Saved file to {output_path}!")

    if save_preds_path:
        doc_bin_out = DocBin(docs=docs)
        doc_bin_out.to_disk(save_preds_path)


if __name__ == "__main__":
    typer.run(run_pipeline)
