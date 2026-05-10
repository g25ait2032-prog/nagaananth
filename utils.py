"""
utils.py — 
Shared helpers: label maps, dataset class, compute_metrics
MLOps Assignment 2 | IIT Jodhpur
"""

import torch
from sklearn.metrics import accuracy_score, f1_score


# ── Label maps (populated at runtime after data is loaded) ──────────────────
label2id: dict = {}
id2label: dict = {}


def build_label_maps(labels: list[str]) -> tuple[dict, dict]:
    """
    Build label2id and id2label dictionaries from a list of string labels.
    Also updates the module-level dictionaries so other modules can import them.
    """
    global label2id, id2label
    unique = sorted(set(labels))           # sorted for determinism
    label2id = {lbl: idx for idx, lbl in enumerate(unique)}
    id2label = {idx: lbl for lbl, idx in label2id.items()}
    return label2id, id2label


# ── PyTorch Dataset ──────────────────────────────────────────────────────────
class MyDataset(torch.utils.data.Dataset):
    """
    Wraps HuggingFace tokenizer encodings and integer labels into a
    PyTorch Dataset that the HuggingFace Trainer understands.
    """

    def __init__(self, encodings, labels: list[int]):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx: int) -> dict:
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self) -> int:
        return len(self.labels)


# ── Evaluation metrics ───────────────────────────────────────────────────────
def compute_metrics(pred):
    """
    Returns accuracy and weighted F1 for the HuggingFace Trainer.
    """
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="weighted"),
    }
