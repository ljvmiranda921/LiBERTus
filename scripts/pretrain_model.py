import os
from enum import Enum
from pathlib import Path
from typing import Optional

import typer
from tokenizers.implementations import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing
from transformers import DataCollatorForLanguageModeling
from transformers import LineByLineTextDataset, RobertaConfig
from transformers import RobertaForMaskedLM, RobertaTokenizerFast, Trainer
from transformers import TrainingArguments
from wasabi import msg

MODEL_CONFIG = {
    "base": {
        "vocab_size": 52_000,
        "max_position_embeddings": 512,
        "hidden_size": 768,
        "intermediate_size": 3072,
        "num_attention_heads": 12,
        "num_hidden_layers": 12,
        "classifier_dropout": 0.1,
        "attention_probs_dropout_prob": 0.1,
        "hidden_dropout_prob": 0.1,
        "type_vocab_size": 2,
    },
    "large": {
        "vocab_size": 52_000,
        "max_position_embeddings": 512,
        "hidden_size": 1024,
        "intermediate_size": 4096,
        "num_attention_heads": 16,
        "num_hidden_layers": 24,
        "classifier_dropout": 0.1,
        "attention_probs_dropout_prob": 0.1,
        "hidden_dropout_prob": 0.1,
        "type_vocab_size": 2,
    },
}


class ModelSize(str, Enum):
    base = "base"
    large = "large"


def pretrain_model(
    # fmt: off
    output_dir: Path = typer.Argument(..., help="Path to save the trained model."),
    model_size: ModelSize = typer.Option(ModelSize.base, "--size", "-sz", help="Size of the model to train."),
    name: Optional[str] = typer.Option(None, "--name", "-n", help="Optional name for the run for W&B tracking."),
    pretraining_corpus: Path = typer.Option(None, help="Path to the pretraining corpus."),
    vocab: Path = typer.Option(None, help="Path to vocab.json to initialize the tokenizer."),
    merges: Path = typer.Option(None, help="Path to merges.txt to initialize the tokenizer."),
    learning_rate: float = typer.Option(6e-4, help="Set the learning rate."),
    batch_size: int = typer.Option(64, help="Set the batch size for GPU training."),
    epochs: int = typer.Option(5, help="Number of epochs to train."),
    max_steps: int = typer.Option(-1, help="Maximum number of steps to run. Overrides epochs."),
    wandb_project: str = typer.Option("sigtyp2024", help="W&B project for tracking model training."),
    seed: int = typer.Option(42, help="Set the random seed."),
    # fmt: on
):
    """Pretrain a model using the RoBERTa architecture."""

    msg.info(f"Loading tokenizer from '{vocab}' and '{merges}'...")
    tokenizer = ByteLevelBPETokenizer(vocab=str(vocab), merges=str(merges))
    tokenizer._tokenizer.post_processor = BertProcessing(
        ("</s>", tokenizer.token_to_id("</s>")),
        ("<s>", tokenizer.token_to_id("<s>")),
    )
    tokenizer.enable_truncation(max_length=512)

    msg.info("Setting-up the pretraining corpus")
    tokenizer = RobertaTokenizerFast.from_pretrained(str(vocab.parent), max_len=512)
    dataset = LineByLineTextDataset(
        tokenizer=tokenizer,
        file_path=str(pretraining_corpus),
        block_size=128,
    )
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15,
    )

    msg.info("Initializing model and Trainer")
    model_config = MODEL_CONFIG.get(model_size.value)
    config = RobertaConfig(**model_config)
    model = RobertaForMaskedLM(config=config)
    msg.text(f"Training a '{model_size.value}' model")
    msg.text(f"Number of parameters to train: {model.num_parameters()}")
    msg.text(f"Model config: {model_config}")

    checkpoint_dir = output_dir / "checkpoints"
    model_dir = output_dir / "model"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    msg.info("Setting up training arguments")
    os.environ["WANDB_PROJECT"] = wandb_project
    os.environ["WANDB_LOGIN_MODEL"] = "checkpoint"
    msg.text(f"Tracking pretraining on W&B project: '{wandb_project}' ('checkpoint')")

    training_args = TrainingArguments(
        output_dir=str(checkpoint_dir),
        # Artifacts
        overwrite_output_dir=True,
        # Optimization parameters
        num_train_epochs=epochs,
        max_steps=max_steps,
        optim="adamw_torch",
        adam_beta2=0.98,
        weight_decay=0.1,
        learning_rate=learning_rate,
        per_gpu_train_batch_size=batch_size,
        lr_scheduler_type="linear",
        warmup_steps=25_000,
        prediction_loss_only=True,
        save_steps=10_000,
        save_total_limit=3,
        # Reproducibility
        seed=seed,
        data_seed=seed,
        # Tracking and reporting
        report_to="wandb",
        logging_steps=100,
        run_name=name,
        log_level="info",
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
    )
    trainer.train()
    trainer.save_model(str(model_dir))


if __name__ == "__main__":
    typer.run(pretrain_model)
