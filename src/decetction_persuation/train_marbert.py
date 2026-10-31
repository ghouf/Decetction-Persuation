"""
Fine-Tuning Module for MARBERT (UBC-NLP/MARBERTv2).
Supports Task 1A (Binary) and Task 1B (Multi-label Persuasion Detection).
All training hyperparameters are exposed.
"""

from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from .trainer_utils import (
    AsymmetricLoss,
    PersuasionDataset,
    TASK1B_TECHNIQUES,
    compute_classification_metrics,
    set_seed,
)


def train_marbert(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    task: str = "task1A",
    model_name: str = "UBC-NLP/MARBERTv2",
    epochs: int = 4,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    weight_decay: float = 0.01,
    max_len: int = 128,
    warmup_ratio: float = 0.1,
    loss_type: str = "auto",
    threshold: float = 0.5,
    early_stopping_patience: int = 2,
    output_dir: Union[str, Path] = "./saved_models/marbert_best",
    fp16: bool = True,
    seed: int = 42,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Fine-tunes MARBERT on Task 1A or Task 1B.

    Parameters:
    -----------
    train_df : pd.DataFrame
        Training dataset containing 'text' and 'label' (for 1A) or 'labels' (for 1B).
    val_df : pd.DataFrame
        Validation dataset.
    task : str
        'task1A' (binary) or 'task1B' (multi-label).
    model_name : str
        Pretrained Hugging Face checkpoint (default: 'UBC-NLP/MARBERTv2').
    epochs : int
        Number of training epochs (default: 4).
    batch_size : int
        Batch size for train and evaluation loaders (default: 16).
    learning_rate : float
        AdamW learning rate (default: 2e-5).
    weight_decay : float
        L2 regularization penalty (default: 0.01).
    max_len : int
        Maximum token length (default: 128).
    warmup_ratio : float
        Proportion of total training steps for linear warmup (default: 0.1).
    loss_type : str
        'auto', 'cross_entropy', 'bce', or 'asymmetric' (ASL).
    threshold : float
        Classification decision threshold for multi-label Task 1B (default: 0.5).
    early_stopping_patience : int
        Stop training after N epochs without validation score improvement.
    output_dir : str or Path
        Directory where best model weights and tokenizer will be saved.
    fp16 : bool
        Enable mixed-precision training (autocast) for faster training on CUDA.
    seed : int
        Random seed for reproducibility.
    device : str, optional
        'cuda', 'cpu', or None (auto-detects).

    Returns:
    --------
    dict with keys: 'model', 'tokenizer', 'history', 'best_metrics', 'id2label'
    """
    set_seed(seed)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 1. Device selection
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n🚀 [MARBERT] Training on device: {device} | Task: {task.upper()} | Model: {model_name}")

    is_binary = "1a" in task.lower()
    label_list = ["false", "true"] if is_binary else TASK1B_TECHNIQUES
    num_labels = len(label_list)

    # 2. Tokenizer and Datasets
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_dataset = PersuasionDataset(train_df, tokenizer, max_len=max_len, task=task, label_list=label_list)
    val_dataset = PersuasionDataset(val_df, tokenizer, max_len=max_len, task=task, label_list=label_list)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 3. Model Architecture
    config = AutoConfig.from_pretrained(
        model_name,
        num_labels=num_labels,
        problem_type="single_label_classification" if is_binary else "multi_label_classification",
        id2label={i: l for i, l in enumerate(label_list)},
        label2id={l: i for i, l in enumerate(label_list)},
    )
    model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config)
    model.to(device)

    # 4. Loss Function
    if loss_type == "auto":
        loss_fn = nn.CrossEntropyLoss() if is_binary else nn.BCEWithLogitsLoss()
    elif loss_type.lower() == "asymmetric":
        loss_fn = AsymmetricLoss()
    elif loss_type.lower() == "bce":
        loss_fn = nn.BCEWithLogitsLoss()
    else:
        loss_fn = nn.CrossEntropyLoss()

    # 5. Optimizer & LR Scheduler
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=learning_rate)
    total_steps = len(train_loader) * epochs
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    scaler = torch.cuda.amp.GradScaler(enabled=(fp16 and device == "cuda"))

    # 6. Training Loop with Early Stopping
    history = {"train_loss": [], "val_loss": [], "val_metrics": []}
    best_score = -1.0
    patience_counter = 0
    primary_metric = "f1_binary" if is_binary else "f1_micro"

    for epoch in range(1, epochs + 1):
        model.train()
        total_train_loss = 0.0
        start_time = time.time()

        train_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]")
        for batch in train_bar:
            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with torch.cuda.amp.autocast(enabled=(fp16 and device == "cuda")):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                loss = loss_fn(logits, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_train_loss += loss.item()
            train_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_train_loss = total_train_loss / len(train_loader)

        # Validation phase
        model.eval()
        total_val_loss = 0.0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                with torch.cuda.amp.autocast(enabled=(fp16 and device == "cuda")):
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    logits = outputs.logits
                    loss = loss_fn(logits, labels)

                total_val_loss += loss.item()

                if is_binary:
                    preds = torch.argmax(logits, dim=1).cpu().numpy()
                else:
                    probs = torch.sigmoid(logits).cpu().numpy()
                    preds = (probs >= threshold).astype(int)

                all_preds.extend(preds)
                all_targets.extend(labels.cpu().numpy())

        avg_val_loss = total_val_loss / len(val_loader)
        val_metrics = compute_classification_metrics(np.array(all_targets), np.array(all_preds), task=task)
        current_score = val_metrics[primary_metric]

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["val_metrics"].append(val_metrics)

        elapsed = time.time() - start_time
        print(f"Epoch {epoch:02d} ({elapsed:.1f}s) | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | {primary_metric.upper()}: {current_score:.4f}")

        # Checkpoint if best score achieved
        if current_score > best_score:
            best_score = current_score
            patience_counter = 0
            model.save_pretrained(output_path)
            tokenizer.save_pretrained(output_path)
            print(f"  ⭐ New best {primary_metric}: {best_score:.4f} -> Saved to {output_path.name}")
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f"  🛑 Early stopping triggered after {epoch} epochs.")
                break

    # Load best checkpoint before returning
    best_model = AutoModelForSequenceClassification.from_pretrained(output_path).to(device)
    return {
        "model": best_model,
        "tokenizer": tokenizer,
        "history": history,
        "best_metric": best_score,
        "primary_metric": primary_metric,
        "id2label": {i: l for i, l in enumerate(label_list)},
    }


def predict_marbert(
    model,
    tokenizer,
    texts: List[str],
    task: str = "task1A",
    max_len: int = 128,
    threshold: float = 0.5,
    device: Optional[str] = None,
) -> Union[List[int], List[List[str]]]:
    """Generates predictions for a list of Arabic texts using fine-tuned MARBERT."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()
    model.to(device)

    is_binary = "1a" in task.lower()
    label_list = ["false", "true"] if is_binary else TASK1B_TECHNIQUES
    predictions = []

    for text in texts:
        inputs = tokenizer(text, max_length=max_len, padding="max_length", truncation=True, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = model(**inputs).logits

        if is_binary:
            pred_id = torch.argmax(logits, dim=1).item()
            predictions.append(pred_id)
        else:
            probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()
            active_labels = [label_list[i] for i, p in enumerate(probs) if p >= threshold]
            if not active_labels:
                active_labels = ["no_technique"]
            predictions.append(active_labels)

    return predictions
