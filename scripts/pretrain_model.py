import os
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

DEFAULT_WANDB_PROJECT = "sigtyp2024"


def pretrain_model(
    # fmt: off
    output_dir: Path = typer.Argument(..., help="Path to save the trained model."),
    name: Optional[str] = typer.Option(None, "--name", "-n", help="Optional name for the run for W&B tracking."),
    pretraining_corpus: Path = typer.Option(None, help="Path to the pretraining corpus."),
    vocab: Path = typer.Option(None, help="Path to vocab.json to initialize the tokenizer."),
    merges: Path = typer.Option(None, help="Path to merges.txt to initialize the tokenizer."),
    batch_size: int = typer.Option(64, help="Set the batch size for GPU training."),
    epochs: int = typer.Option(5, help="Number of epochs to train."),
    wandb_project: str = typer.Option(DEFAULT_WANDB_PROJECT, help="W&B project for tracking model training."),
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
    config = RobertaConfig(
        vocab_size=52_000,
        max_position_embeddings=512,
        num_attention_heads=12,
        num_hidden_layers=12,
        type_vocab_size=2,
    )
    model = RobertaForMaskedLM(config=config)
    msg.text(f"Number of parameters to train: {model.num_parameters()}")

    checkpoint_dir = output_dir / "checkpoints"
    model_dir = output_dir / "model"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    msg.info("Setting up training arguments")
    os.environ["WANDB_PROJECT"] = wandb_project
    os.environ["WANDB_LOGIN_MODEL"] = "checkpoint"
    msg.text(f"Tracking pretraining on W&B project: '{wandb_project}' ('checkpoint')")

    training_args = TrainingArguments(
        seed=seed,
        # Artifacts
        output_dir=str(checkpoint_dir),
        overwrite_output_dir=True,
        num_train_epochs=epochs,
        # Train time parameters
        per_gpu_train_batch_size=batch_size,
        prediction_loss_only=True,
        save_steps=10_000,
        save_total_limit=3,
        # Tracking and reporting
        report_to="wandb",
        logging_steps=1,
        run_name=name,
        load_best_model_at_end=True,
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
