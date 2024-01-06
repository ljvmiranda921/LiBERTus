import spacy
from spacy.language import Language
import numpy as np
from wasabi import msg
from typing import Any


def load_model():
    pass


class TrfEmbeddingsDict(dict):
    def __init__(self, model_name: str):
        super().__init__()
        self.nlp = spacy.load(model_name)
        self.nlp.add_pipe("tensor2attr")

    def __getitem__(self, __key: str) -> Any:
        doc = self.nlp(__key)
        return doc.vector.shape
