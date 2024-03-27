<!-- WEASEL: AUTO-GENERATED DOCS START (do not remove) -->

# 🪐 LiBERTus - A Multilingual Language Model for Ancient and Historical Languages

Submission to Task 1 (Constrained) of the [SIGTYP 2024 Shared Task on Word
Embedding Evaluation for Ancient and Historical
Languages](https://sigtyp.github.io/st2024.html).  The system is built by
first pretraining a multilingual language model and then finetuning it for a
downstream task. The submission for Phase 1 and 2 of the Shared Task can be
found in the `submission_p1` and `submission_p2` directories.


## 📋 project.yml

The [`project.yml`](project.yml) defines the data assets required by the
project, as well as the available commands and workflows. For details, see the
[Weasel documentation](https://github.com/explosion/weasel).

### ⏯ Commands

The following commands are defined by the project. They
can be executed using [`weasel run [name]`](https://github.com/explosion/weasel/tree/main/docs/cli.md#rocket-run).
Commands are only re-run if their inputs have changed.

| Command | Description |
| --- | --- |
| `create-pretraining` | Create corpus for multilingual LM pretraining |
| `create-vocab` | Train a tokenizer to create a vocabulary |
| `pretrain-model` | Pretrain a multilingual LM from a corpus |
| `pretrain-model-from-checkpoint` | Pretrain a multilingual LM from a corpus based on a checkpoint |
| `upload-to-hf` | Upload pretrained model and corresponding tokenizer to the HuggingFace repository |
| `convert-to-spacy-merged` | Convert CoNLL-U files into spaCy format for finetuning |
| `convert-to-spacy` | Convert CoNLL-U files into spaCy format for finetuning |
| `finetune-tok2vec-model` | Finetune a tok2vec model given a training and validation corpora |
| `finetune-trf-model` | Finetune a transformer model given a training and validation corpora |
| `finetune-with-merged-corpus` | Finetune a transformer model on the combined training and validation corpora |
| `package-model` | Package model and upload to HuggingFace |
| `evaluate-model-dev` | Evaluate a model on the validation set |
| `plot-figures` | Plot figures for the writeup |
| `setup-test` | Install models from HuggingFace via pip |
| `download-models-locally` | Download models from HuggingFace |
| `get-test-results` | Get results from the test file |
| `zip-results-p1` | Zip the results into a single file for submission (Phase 1) |
| `zip-results-p2` | Zip teh results into a single file for submission (Phase 2) |

### ⏭ Workflows

The following workflows are defined by the project. They
can be executed using [`weasel run [name]`](https://github.com/explosion/weasel/tree/main/docs/cli.md#rocket-run)
and will run the specified commands in order. Commands are only re-run if their
inputs have changed.

| Workflow | Steps |
| --- | --- |
| `pretrain` | `create-pretraining` &rarr; `create-vocab` &rarr; `pretrain-model` |
| `finetune` | `convert-to-spacy` &rarr; `finetune-trf-model` &rarr; `evaluate-model-dev` |
| `experiment-merged` | `convert-to-spacy-merged` &rarr; `finetune-with-merged-corpus` |
| `experiment-sampling` | `create-vocab` &rarr; `pretrain-model` |
| `make-submission-p1` | `setup-test` &rarr; `get-test-results` &rarr; `zip-results-p1` |
| `make-submission-p2` | `download-models-locally` &rarr; `zip-results-p2` |

### 🗂 Assets

The following assets are defined by the project. They can
be fetched by running [`weasel assets`](https://github.com/explosion/weasel/tree/main/docs/cli.md#open_file_folder-assets)
in the project directory.

| File | Source | Description |
| --- | --- | --- |
| `assets/train/` | Git | CoNLL-U training datasets for Task 0 (morphology/lemma/POS) |
| `assets/dev/` | Git | CoNLL-U validation datasets for Task 0 (morphology/lemma/POS) |
| `assets/test/` | Git | CoNLL-U test datasets for Task 0 (morphology/lemma/POS) |

## 📄 Cite

If you used any of the code or the models, don't forget to cite

```
@inproceedings{miranda-2024-allen,
    title = "{A}llen Institute for {AI} @ {SIGTYP} 2024 Shared Task on Word Embedding Evaluation for Ancient and Historical Languages",
    author = "Miranda, Lester",
    booktitle = "Proceedings of the 6th Workshop on Research in Computational Linguistic Typology and Multilingual NLP",
    month = mar,
    year = "2024",
    address = "St. Julian's, Malta",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.sigtyp-1.18",
    pages = "151--159",
}
```


<!-- WEASEL: AUTO-GENERATED DOCS END (do not remove) -->
