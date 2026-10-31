"""
Baselines and Heuristic Classifiers for Persuasion Detection (TP2 - Partie A).
Includes:
1. Trivial Majority Classifier
2. Baseline TF-IDF (words 1-2 + chars 2-5) + Logistic Regression (class_weight='balanced') with threshold tuning on val
3. Heuristic Shortcut Rules:
   - Rule 1: 'genre => persuasif' (Task 1A)
   - Rule 2: 'contient un mot très négatif => langage chargé' (Task 1B)
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import FeatureUnion

from .trainer_utils import TASK1B_TECHNIQUES, compute_classification_metrics

# Frequent negative/charged Arabic words extracted from persuasion corpora
FREQUENT_NEGATIVE_WORDS = [
    "إرهاب",
    "إرهابي",
    "إرهابيين",
    "فساد",
    "فاسد",
    "فاشل",
    "فشل",
    "جريمة",
    "جرائم",
    "عدوان",
    "احتلال",
    "سرقة",
    "خيانة",
    "عار",
    "كارثة",
    "تدمير",
    "دمار",
    "وحشية",
    "قمع",
    "نفاق",
    "منافق",
    "مؤامرة",
    "كذب",
    "تزوير",
    "دماء",
    "قتل",
    "عنف",
    "مجرم",
]


# =====================================================================
# 1. Trivial Classifier (Majority Class)
# =====================================================================
def evaluate_trivial_classifier(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    task: str = "task1A"
) -> Dict[str, Any]:
    """
    Evaluates a dummy classifier that always predicts the majority class observed on train.
    """
    is_binary = "1a" in task.lower()

    if is_binary:
        # Task 1A: Find most frequent label ('true' vs 'false')
        majority_label = train_df["label"].astype(str).str.lower().value_counts().idxmax()
        y_val_true = (val_df["label"].astype(str).str.lower() == "true").astype(int)
        y_val_pred = np.full(len(val_df), 1 if majority_label == "true" else 0)

        acc = accuracy_score(y_val_true, y_val_pred)
        f1_bin = f1_score(y_val_true, y_val_pred, average="binary", zero_division=0)
        f1_macro = f1_score(y_val_true, y_val_pred, average="macro", zero_division=0)

        results = {
            "task": "Task 1A",
            "majority_class": majority_label,
            "accuracy": float(acc),
            "f1_binary": float(f1_bin),
            "f1_macro": float(f1_macro),
        }
        print(f"[Trivial 1A] Majority: '{majority_label}' | Acc: {acc:.4f} | F1-Binary: {f1_bin:.4f} | F1-Macro: {f1_macro:.4f}")
        return results

    else:
        # Task 1B: Find most frequent technique (Loaded_Language)
        all_train_labels = [lbl for sublist in train_df.get("labels", []) for lbl in sublist]
        majority_tech = pd.Series(all_train_labels).value_counts().idxmax()

        # Build validation true binary matrix
        tech_list = TASK1B_TECHNIQUES
        tech2id = {t: i for i, t in enumerate(tech_list)}
        y_val_true = np.zeros((len(val_df), len(tech_list)), dtype=int)
        for idx, row in enumerate(val_df.get("labels", [])):
            for lbl in row:
                if lbl in tech2id:
                    y_val_true[idx, tech2id[lbl]] = 1

        # Predict only majority class for all samples
        y_val_pred = np.zeros_like(y_val_true)
        y_val_pred[:, tech2id[majority_tech]] = 1

        f1_micro = f1_score(y_val_true, y_val_pred, average="micro", zero_division=0)
        f1_macro = f1_score(y_val_true, y_val_pred, average="macro", zero_division=0)

        results = {
            "task": "Task 1B",
            "majority_technique": majority_tech,
            "f1_micro": float(f1_micro),
            "f1_macro": float(f1_macro),
        }
        print(f"[Trivial 1B] Majority: '{majority_tech}' | F1-Micro: {f1_micro:.4f} | F1-Macro: {f1_macro:.4f}")
        return results


# =====================================================================
# 2. Baseline TF-IDF + Logistic Regression with Validation Threshold Tuning
# =====================================================================
def train_baseline_tfidf_lr(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    task: str = "task1A",
    class_weight: str = "balanced",
    ngram_words: Tuple[int, int] = (1, 2),
    ngram_chars: Tuple[int, int] = (2, 5),
    max_features_word: int = 5000,
    max_features_char: int = 10000,
) -> Dict[str, Any]:
    """
    Fits TF-IDF (word n-grams 1-2 + char n-grams 2-5) strictly on train_df,
    trains Logistic Regression with class_weight='balanced',
    and searches for the optimal decision threshold on val_df.
    """
    is_binary = "1a" in task.lower()

    # 1. Build FeatureUnion for word and character TF-IDF
    word_vectorizer = TfidfVectorizer(ngram_range=ngram_words, max_features=max_features_word)
    char_vectorizer = TfidfVectorizer(analyzer="char", ngram_range=ngram_chars, max_features=max_features_char)
    tfidf = FeatureUnion([("word_tfidf", word_vectorizer), ("char_tfidf", char_vectorizer)])

    # Fit ONLY on train
    X_train = tfidf.fit_transform(train_df["text"].astype(str))
    X_val = tfidf.transform(val_df["text"].astype(str))

    if is_binary:
        y_train = (train_df["label"].astype(str).str.lower() == "true").astype(int)
        y_val = (val_df["label"].astype(str).str.lower() == "true").astype(int)

        clf = LogisticRegression(class_weight=class_weight, max_iter=1000, random_state=42)
        clf.fit(X_train, y_train)

        # Predict probability of class 1
        val_probs = clf.predict_proba(X_val)[:, 1]

        # Optimize threshold on val_df
        best_threshold = 0.5
        best_score = -1.0
        candidate_thresholds = np.linspace(0.1, 0.9, 81)

        for th in candidate_thresholds:
            preds = (val_probs >= th).astype(int)
            score = f1_score(y_val, preds, average="macro", zero_division=0)
            if score > best_score:
                best_score = score
                best_threshold = float(th)

        final_preds = (val_probs >= best_threshold).astype(int)
        acc = accuracy_score(y_val, final_preds)
        f1_bin = f1_score(y_val, final_preds, average="binary", zero_division=0)
        f1_mac = f1_score(y_val, final_preds, average="macro", zero_division=0)

        print(f"[TF-IDF + LR 1A] Optimal Threshold: {best_threshold:.2f} | Acc: {acc:.4f} | F1-Binary: {f1_bin:.4f} | F1-Macro: {f1_mac:.4f}")
        return {
            "model": clf,
            "vectorizer": tfidf,
            "optimal_threshold": best_threshold,
            "accuracy": float(acc),
            "f1_binary": float(f1_bin),
            "f1_macro": float(f1_mac),
        }

    else:
        # Task 1B Multi-label One-vs-Rest
        tech_list = TASK1B_TECHNIQUES
        tech2id = {t: i for i, t in enumerate(tech_list)}

        y_train = np.zeros((len(train_df), len(tech_list)), dtype=int)
        for idx, row in enumerate(train_df.get("labels", [])):
            for lbl in row:
                if lbl in tech2id:
                    y_train[idx, tech2id[lbl]] = 1

        y_val = np.zeros((len(val_df), len(tech_list)), dtype=int)
        for idx, row in enumerate(val_df.get("labels", [])):
            for lbl in row:
                if lbl in tech2id:
                    y_val[idx, tech2id[lbl]] = 1

        clf = OneVsRestClassifier(LogisticRegression(class_weight=class_weight, max_iter=1000, random_state=42))
        clf.fit(X_train, y_train)

        val_probs = clf.predict_proba(X_val)

        # Optimize global threshold on val for micro-F1
        best_threshold = 0.5
        best_score = -1.0
        candidate_thresholds = np.linspace(0.1, 0.8, 36)

        for th in candidate_thresholds:
            preds = (val_probs >= th).astype(int)
            score = f1_score(y_val, preds, average="micro", zero_division=0)
            if score > best_score:
                best_score = score
                best_threshold = float(th)

        final_preds = (val_probs >= best_threshold).astype(int)
        f1_mic = f1_score(y_val, final_preds, average="micro", zero_division=0)
        f1_mac = f1_score(y_val, final_preds, average="macro", zero_division=0)

        print(f"[TF-IDF + LR 1B] Optimal Threshold: {best_threshold:.2f} | F1-Micro: {f1_mic:.4f} | F1-Macro: {f1_mac:.4f}")
        return {
            "model": clf,
            "vectorizer": tfidf,
            "optimal_threshold": best_threshold,
            "f1_micro": float(f1_mic),
            "f1_macro": float(f1_mac),
        }


# =====================================================================
# 3. Raccourcis (Heuristic Single-Rule Shortcuts)
# =====================================================================
def evaluate_rule_genre_to_persuasion(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Règle heuristique (Task 1A) : « genre => persuasif ».
    Vérifie la corrélation entre le genre ('paragraph' vs 'tweet') et la persuasion ('true').
    """
    # 1. Calculate rate of 'true' per genre on train
    train_stats = train_df.groupby("type")["label"].apply(lambda s: (s.astype(str).str.lower() == "true").mean())
    preferred_genre = train_stats.idxmax()
    preferred_rate = train_stats.max()

    # Rule: If type == preferred_genre -> predict 'true', else 'false'
    y_val_true = (val_df["label"].astype(str).str.lower() == "true").astype(int)
    y_val_pred = (val_df["type"] == preferred_genre).astype(int)

    acc = accuracy_score(y_val_true, y_val_pred)
    f1_bin = f1_score(y_val_true, y_val_pred, average="binary", zero_division=0)
    f1_mac = f1_score(y_val_true, y_val_pred, average="macro", zero_division=0)

    print(f"[Raccourci 1] Regle: '{preferred_genre} => persuasif' (Train rate: {preferred_rate*100:.1f}%)")
    print(f"   Score sur Val: Acc: {acc:.4f} | F1-Binary: {f1_bin:.4f} | F1-Macro: {f1_mac:.4f}")

    return {
        "rule": f"genre '{preferred_genre}' => persuasif",
        "train_rates": train_stats.to_dict(),
        "accuracy": float(acc),
        "f1_binary": float(f1_bin),
        "f1_macro": float(f1_mac),
    }


def evaluate_rule_negative_word_to_loaded_language(
    val_df: pd.DataFrame,
    word_list: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Règle heuristique (Task 1B) : « contient un mot très négatif => langage chargé (Loaded_Language) ».
    """
    words = word_list or FREQUENT_NEGATIVE_WORDS

    # Ground truth: whether Loaded_Language is present in labels
    y_true = []
    for labels in val_df.get("labels", []):
        y_true.append(1 if "Loaded_Language" in labels else 0)

    # Heuristic Prediction: Check if any negative word is in text
    y_pred = []
    for text in val_df["text"].astype(str):
        has_negative = any(w in text for w in words)
        y_pred.append(1 if has_negative else 0)

    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    acc = accuracy_score(y_true, y_pred)

    print(f"[Raccourci 2] Regle: 'mot negatif => Loaded_Language' (Vocab: {len(words)} mots)")
    print(f"   Score sur Val: Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f} | Acc: {acc:.4f}")

    return {
        "rule": "contient un mot très négatif => langage chargé",
        "word_list_size": len(words),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "accuracy": float(acc),
    }
