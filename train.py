"""
train.py — 
Model loading, Trainer setup, and training loop with W&B tracking
MLOps Assignment 2 | IIT Jodhpur
"""

import os
import wandb

from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    TrainingArguments,
    Trainer,
)

from data import load_all_genres, train_test_split, encode_datasets
from utils import build_label_maps, compute_metrics


# ── Hyperparameters ──────────────────────────────────────────────────────────
MODEL_NAME   = "distilbert-base-cased"
MAX_LENGTH   = 512
DEVICE       = "cuda"        # change to "cpu" if no GPU is available
NUM_EPOCHS   = 3
BATCH_TRAIN  = 16
BATCH_EVAL   = 32
LEARNING_RATE = 3e-5
WARMUP_STEPS  = 100
WEIGHT_DECAY  = 0.01
OUTPUT_DIR    = "./results"
WANDB_PROJECT = "mlops-assignment2"
WANDB_RUN     = "distilbert-run-1"
HF_REPO       = "nagaananth/distilbert-goodreads-genres"  # update with your HF username


def main():
    # ── 1. Load and prepare data ─────────────────────────────────────────────
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)

    genre_reviews = load_all_genres()
    tr_texts, tr_labels, te_texts, te_labels = train_test_split(genre_reviews)
    train_dataset, test_dataset = encode_datasets(
        tr_texts, tr_labels, te_texts, te_labels, tokenizer, max_length=MAX_LENGTH
    )

    # Build label maps (populated inside encode_datasets via build_label_maps)
    from utils import id2label
    num_labels = len(id2label)

    # ── 2. Load pre-trained model ────────────────────────────────────────────
    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_labels,
        id2label=id2label,
        label2id={v: k for k, v in id2label.items()},
    ).to(DEVICE)

    # ── 3. Initialise W&B ────────────────────────────────────────────────────
    wandb.init(
        project=WANDB_PROJECT,
        name=WANDB_RUN,
        config={
            "model":         MODEL_NAME,
            "epochs":        NUM_EPOCHS,
            "batch_size":    BATCH_TRAIN,
            "learning_rate": LEARNING_RATE,
            "max_length":    MAX_LENGTH,
            "dataset":       "UCSD Goodreads",
        },
    )

    # ── 4. Training arguments ────────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_TRAIN,
        per_device_eval_batch_size=BATCH_EVAL,
        warmup_steps=WARMUP_STEPS,
        weight_decay=WEIGHT_DECAY,
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        report_to="wandb",   # single line enables full W&B logging
        run_name=WANDB_RUN,
        learning_rate=LEARNING_RATE,
    )

    # ── 5. Trainer ───────────────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    # ── 6. Train ─────────────────────────────────────────────────────────────
    print("Starting training …")
    trainer.train()

    # ── 7. Save model locally ────────────────────────────────────────────────
    trainer.save_model("distilbert-goodreads-genres")
    tokenizer.save_pretrained("distilbert-goodreads-genres")
    print("Model saved to ./distilbert-goodreads-genres")

    # ── 8. Push to Hugging Face Hub ──────────────────────────────────────────
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        from huggingface_hub import login
        login(token=hf_token)
        model.push_to_hub(HF_REPO)
        tokenizer.push_to_hub(HF_REPO)
        hf_url = f"https://huggingface.co/{HF_REPO}"
        wandb.run.summary["huggingface_model"] = hf_url
        print(f"Model pushed to {hf_url}")
    else:
        print("HF_TOKEN not set — skipping Hugging Face Hub push.")

    wandb.finish()
    print("Training complete.")

    return trainer, test_dataset, te_labels


if __name__ == "__main__":
    main()
