from typing import Any

import numpy as np
import spacy
from spacy.tokens import Doc, Span, Token
from spacy.language import Language
from wasabi import msg

print("hi")


@Language.factory("tensor2attr")
class Tensor2Attr:
    def __init__(self, name, nlp):
        pass

    def __call__(self, doc: Doc):
        self.add_attributes(doc)
        return doc

    def add_attributes(self, doc: Doc):
        doc.user_hooks["vector"] = self.doc_tensor
        doc.user_span_hooks["vector"] = self.span_tensor
        doc.user_token_hooks["vector"] = self.token_tensor

        doc.user_hooks["similarity"] = self.get_similarity
        doc.user_span_hooks["similarity"] = self.get_similarity
        doc.user_token_hooks["similarity"] = self.get_similarity

    def doc_tensor(self, doc: Doc) -> np.ndarray:
        return doc._.trf_data.tensors[-1].mean(axis=0)

    def span_tensor(self, span: Span) -> np.ndarray:
        tensor_ix = span.doc._.trf_data.align[span.start : span.end].data.flatten()
        out_dim = span.doc._.trf_data.tensors[0].shape[-1]
        tensor = span.doc._.trf_data.tensors[0].reshape(-1, out_dim)[tensor_ix]
        return tensor.mean(axis=0)

    def token_tensor(self, token: Token) -> np.ndarray:
        tensor_ix = token.doc._.trf_data.align[token.i].data.flatten()
        out_dim = token.doc._.trf_data.tensors[0].shape[-1]
        tensor = token.doc._.trf_data.tensors[0].reshape(-1, out_dim)[tensor_ix]
        return tensor.mean(axis=0)

    def get_similarity(self, doc1: Doc, doc2: Doc) -> np.ndarray:
        return np.dot(doc1.vector, doc2.vector) / (doc1.vector_norm * doc2.vector_norm)


class TransformerEmbeddings(dict):
    def __init__(self, model_name: str):
        super().__init__()
        self.nlp = spacy.load(model_name)
        self.nlp.add_pipe("tensor2attr")

    def __getitem__(self, __key: str) -> Any:
        doc = self.nlp(__key)
        return doc.vector


def load_model(model_name: str) -> TransformerEmbeddings:
    if model_name not in spacy.util.get_installed_models():
        msg.fail(
            f"Model '{model_name}' not found in your environment. "
            f" Found: {', '.join(spacy.util.get_installed_models())}."
            " Please install the model from requirements.txt",
            " To get the list of models per language code, import and call "
            " model_table()",
            exits=1,
        )

    embeddings = TransformerEmbeddings(model_name)
    return embeddings


def model_table():
    pass
