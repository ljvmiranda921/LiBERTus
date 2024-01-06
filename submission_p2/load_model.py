from typing import Any

import numpy as np
import spacy
from spacy.language import Language
from wasabi import msg


class TransformerEmbeddings(dict):
    def __init__(self, model_name: str):
        super().__init__()
        self.nlp = spacy.load(model_name)
        self.nlp.add_pipe("tensor2attr")

    def __getitem__(self, __key: str) -> Any:
        doc = self.nlp(__key)
        return doc.vector.shape


def load_model(model_name: str) -> TransformerEmbeddings:
    if model_name not in spacy.util.get_installed_models():
        msg.fail(
            f"Model '{model_name}' not found in your environment. "
            f" Found: {', '.join(spacy.util.get_installed_models())}",
            exits=1,
        )

    embeddings = TransformerEmbeddings(model_name)
    return embeddings
