import json
from enum import Enum
from pathlib import Path
from typing import List, Optional, Tuple

import conllu
import spacy
import typer
from spacy.cli._util import setup_gpu
from spacy.tokens import Doc, DocBin
from tqdm import tqdm
from wasabi import msg


class MWE(str, Enum):
    all: str = "all"
    subtokens_only: str = "subtokens_only"
    mwe_only: str = "mwe_only"


def run_pipeline(
    # fmt: off
    input_path: Path = typer.Argument(..., help="Path to the input spaCy / Co file."),
    output_dir: Path = typer.Argument(..., help="Directory to save the outputs."),
    model: str = typer.Argument(..., help="spaCy pipeline to use."),
    lang: Optional[str] = typer.Option(None, help="Language code of the file. If None, will infer from input_path."),
    save_preds_path: Optional[Path] = typer.Option(None, help="Optional path to save the predictions as a spaCy file."),
    multiword_handling: MWE = typer.Option(MWE.all, "--multiword-handling", "--multiword", "--mwe", "-M", help="Dictates how multiword expressions are handled in cop and hbo."),
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

    texts = get_texts(input_path, nlp, lang_code, multiword=multiword_handling)
    docs = nlp.pipe(texts, n_process=n_process)
    # infer reference CoNLL-U file
    conllu_file = Path(f"assets/test/{lang_code}_test.conllu")
    references = [sent for sent in conllu.parse_incr(conllu_file)]

    if len(references) != len(docs):
        msg.fail("Unequal num of sentences", exits=1)

    results = {"pos_tagging": [], "morph_features": [], "lemmatisation": []}
    for doc, ref in tqdm(zip(docs, references)):
        if len(doc) != len(ref):
            doc = retokenize_doc(doc, ref)
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


def retokenize_doc(doc: Doc, reference: conllu.TokenList) -> Doc:
    while True:
        # Get offsets of tokens that need to be retokenized
        offsets: List[Tuple[int, int]] = []
        for i, (tok_doc, tok_ref) in enumerate(zip(doc, reference)):
            if tok_doc.text != tok_ref["form"]:
                offsets.append((i, i + 1))

        # If no mismatches found, break the loop
        if not offsets:
            break

        for start, end in offsets:
            with doc.retokenize() as retokenizer:
                retokenizer.merge(doc[start:end])
    return doc


def get_texts(
    input_path: Path, nlp: spacy.language.Language, lang_code: str, *, multiword: MWE
) -> List[str]:
    """Read the file and get the texts"""
    SPECIAL_CASE_MWE = ["cop", "hbo"]
    SPECIAL_CASE_PARSE = ["orv"]

    def _check_if_conllu(input_path: Path):
        if input_path.suffix != ".conllu":
            msg.fail("This language code requires the CoNLL-U file", exits=1)

    if lang_code in SPECIAL_CASE_MWE:
        msg.info(f"Language {lang_code} contains multiword tokens")
        if multiword == MWE.all:
            _check_if_conllu(input_path)
            msg.text("Getting both multiword expressions and subtokens")
            texts = []
            for sentence in conllu.parse_incr(input_path.open(encoding="utf-8")):
                tokens = [token["form"] for token in sentence]
                texts.append(" ".join(tokens))
        elif multiword == MWE.subtokens_only:
            _check_if_conllu(input_path)
            msg.text("Getting the subtokens only")
            texts = []
            for sentence in conllu.parse_incr(input_path.open(encoding="utf-8")):
                tokens = []
                for token in sentence:
                    if isinstance(token["id"], int):
                        tokens.append(token["form"])
                texts.append(" ".join(tokens))
        elif multiword == MWE.mwe_only:
            msg.text("Getting the multiword expressions only")
            doc_bin = DocBin().from_disk(input_path)
            _docs = doc_bin.get_docs(nlp.vocab)
            # Convert to text for faster processing
            texts = [_doc.text for _doc in _docs]
        else:
            msg.fail(f"Unknown multiword handler: {multiword}", exits=1)
    elif lang_code in SPECIAL_CASE_PARSE:
        _check_if_conllu(input_path)
        texts = [
            sent.metadata["text"]
            for sent in conllu.parse_incr(input_path.open(encoding="utf-8"))
        ]
    else:
        doc_bin = DocBin().from_disk(input_path)
        _docs = doc_bin.get_docs(nlp.vocab)
        # Convert to text for faster processing
        texts = [_doc.text for _doc in _docs]

    return texts


if __name__ == "__main__":
    typer.run(run_pipeline)
