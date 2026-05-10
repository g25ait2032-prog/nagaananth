"""
eval.py — 
Final evaluation on the test set, metrics logging, and W&B Artifact upload
MLOps Assignment 2 | IIT Jodhpur

Usage:
    python eval.py
    
"""

import json
import wandb

from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from sklearn.metrics import classification_report

from data import load_all_genres, train_test_split, encode_datasets
from utils import build_label_maps, compute_metrics, id2label


MODEL_DIR     = "./distilbert-goodreads-genres"
MODEL_NAME    = "distilbert-base-cased"
MAX_LENGTH    = 512
DEVICE        = "cuda"
WANDB_PROJECT = "mlops-assignment2"
WANDB_RUN     = "distilbert-eval"


def main():
    # ── 1. Rebuild test dataset ──────────────────────────────────────────────
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_DIR)

    genre_reviews = load_all_genres()
    tr_texts, tr_labels, te_texts, te_labels = train_test_split(genre_reviews)
    _, test_dataset = encode_datasets(
        tr_texts, tr_labels, te_texts, te_labels, tokenizer, max_length=MAX_LENGTH
    )

    # ── 2. Load fine-tuned model ─────────────────────────────────────────────
    model = DistilBertForSequenceClassification.from_pretrained(MODEL_DIR).to(DEVICE)

    # ── 3. Minimal TrainingArguments for evaluation ──────────────────────────
    eval_args = TrainingArguments(
        output_dir="./eval_output",
        per_device_eval_batch_size=32,
        report_to="wandb",
        run_name=WANDB_RUN,
    )

    trainer = Trainer(
        model=model,
        args=eval_args,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    # ── 4. Initialise W&B for eval run ───────────────────────────────────────
    wandb.init(project=WANDB_PROJECT, name=WANDB_RUN)

    # ── 5. Run evaluation ────────────────────────────────────────────────────
    eval_results = trainer.evaluate()
    print("Evaluation results:", eval_results)

    # ── 6. Log final metrics explicitly ─────────────────────────────────────
    wandb.log({
        "final/loss":     eval_results["eval_loss"],
        "final/accuracy": eval_results["eval_accuracy"],
        "final/f1":       eval_results["eval_f1"],
    })

    # ── 7. Full classification report ───────────────────────────────────────
    pred_output = trainer.predict(test_dataset)
    preds = pred_output.predictions.argmax(-1).flatten().tolist()

    # Convert integer labels back to genre strings for the report
    from utils import id2label as _id2label
    preds_str = [_id2label[p] for p in preds]

    report = classification_report(
        te_labels,
        preds_str,
        output_dict=True,
    )
    print("\nClassification Report:")
    print(classification_report(te_labels, preds_str))

    # ── 8. Save report and upload as W&B Artifact ───────────────────────────
    report_path = "eval_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    artifact = wandb.Artifact("eval-report", type="evaluation")
    artifact.add_file(report_path)
    wandb.log_artifact(artifact)
    print(f"Saved and uploaded {report_path} as W&B Artifact.")

    wandb.finish()
    print("Evaluation complete.")


if __name__ == "__main__":
    main()
