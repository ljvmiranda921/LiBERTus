from typing import Dict, List

ACL_STYLE = {
    "legend.fontsize": "large",
    "axes.labelsize": "x-large",
    "axes.titlesize": "x-large",
    "xtick.labelsize": "larger",
    "ytick.labelsize": "larger",
    "text.usetex": False,
    "font.family": "STIX",
}


COMPONENT_TO_METRIC: Dict[str, str] = {
    "tagger": "tag_acc",
    "morphologizer": "morph_acc",
    "trainable_lemmatizer": "lemma_acc",
}

COMPONENT_TO_TASK: Dict[str, str] = {
    "tagger": "Parts-of-speech (POS) tagging",
    "morphologizer": "Morphological annotation",
    "trainable_lemmatizer": "Lemmatization",
}
