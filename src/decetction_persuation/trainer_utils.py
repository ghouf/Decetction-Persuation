"""
Shared Utilities, Datasets, and Loss Functions for Arabic Transformer Fine-Tuning.
Supports Task 1A (Binary) and Task 1B (Multi-label).
"""

import os
import random
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import torch
import torch.nn as nn
from torch.utils.data import Dataset

# Default 24 classes for Task 1B in standard alphabetical order
TASK1B_TECHNIQUES = [
    "Appeal_to_Authority",
    "Appeal_to_Fear-Prejudice",
    "Appeal_to_Hypocrisy",
    "Appeal_to_Popularity",
    "Appeal_to_Time",
    "Appeal_to_Values",
    "Causal_Oversimplification",
    "Consequential_Oversimplification",
    "Conversation_Killer",
    "Doubt",
    "Exaggeration-Minimisation",
    "False_Dilemma-No_Choice",
    "Flag_Waving",
    "Guilt_by_Association",
    "Loaded_Language",
    "Name_Calling-Labeling",
    "Obfuscation-Vagueness-Confusion",
    "Questioning_the_Reputation",
    "Red_Herring",
    "Repetition",
    "Slogans",
    "Straw_Man",
    "Whataboutism",
    "no_technique",
]


def set_seed(seed: int = 42) -> None:
    """Sets random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class PersuasionDataset(Dataset):
    """
    PyTorch Dataset for Persuasion Detection.
    Handles both Task 1A (Binary) and Task 1B (Multi-label).
    """

    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer,
        max_len: int = 128,
        task: str = "task1A",
        label_list: Optional[List[str]] = None,
    ):
        self.texts = df["text"].astype(str).tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.task = task.lower()
        self.label_list = label_list or (
            ["false", "true"] if "1a" in self.task else TASK1B_TECHNIQUES
        )
        self.label2id = {lbl: i for i, lbl in enumerate(self.label_list)}
        self.id2label = {i: lbl for i, lbl in enumerate(self.label_list)}

        # Format labels
        self.labels = []
        if "1a" in self.task:
            # Binary task: "true"/"false" or 1/0
            for val in df.get("label", df.get("label_binary", [0] * len(df))):
                if str(val).lower() in ["1", "true"]:
                    self.labels.append(1)
                else:
                    self.labels.append(0)
        else:
            # Task 1B Multi-label: list of technique names -> multi-hot binary vector
            num_classes = len(self.label_list)
            for row_labels in df.get("labels", [[]] * len(df)):
                multi_hot = np.zeros(num_classes, dtype=np.float32)
                if isinstance(row_labels, list):
                    for lbl in row_labels:
                        if lbl in self.label2id:
                            multi_hot[self.label2id[lbl]] = 1.0
                elif isinstance(row_labels, str) and row_labels in self.label2id:
                    multi_hot[self.label2id[row_labels]] = 1.0
                self.labels.append(multi_hot)

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        item = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
        }
        if "token_type_ids" in encoding:
            item["token_type_ids"] = encoding["token_type_ids"].squeeze(0)

        if "1a" in self.task:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        else:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float32)

        return item


class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss (ASL) for imbalanced multi-label classification.
    Significantly downweights easy negatives while preserving hard positive gradients.
    """

    def __init__(
        self,
        gamma_neg: float = 4.0,
        gamma_pos: float = 1.0,
        clip: float = 0.05,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        xs_pos = torch.sigmoid(logits)
        xs_neg = 1.0 - xs_pos

        if self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1.0)

        loss_pos = targets * torch.log(xs_pos.clamp(min=self.eps)) * ((1.0 - xs_pos) ** self.gamma_pos)
        loss_neg = (1.0 - targets) * torch.log(xs_neg.clamp(min=self.eps)) * (xs_pos ** self.gamma_neg)

        loss = -(loss_pos + loss_neg)
        return loss.sum()


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    task: str = "task1A"
) -> Dict[str, float]:
    """Computes standard NLP classification metrics."""
    if "1a" in task.lower():
        acc = accuracy_score(y_true, y_pred)
        f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
        f1_pos = f1_score(y_true, y_pred, average="binary", zero_division=0)
        prec = precision_score(y_true, y_pred, average="binary", zero_division=0)
        rec = recall_score(y_true, y_pred, average="binary", zero_division=0)
        return {
            "accuracy": float(acc),
            "f1_binary": float(f1_pos),
            "f1_macro": float(f1_macro),
            "precision": float(prec),
            "recall": float(rec),
        }
    else:
        # Multi-label metrics (Task 1B)
        f1_micro = f1_score(y_true, y_pred, average="micro", zero_division=0)
        f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
        prec_micro = precision_score(y_true, y_pred, average="micro", zero_division=0)
        rec_micro = recall_score(y_true, y_pred, average="micro", zero_division=0)
        return {
            "f1_micro": float(f1_micro),
            "f1_macro": float(f1_macro),
            "precision_micro": float(prec_micro),
            "recall_micro": float(rec_micro),
        }
