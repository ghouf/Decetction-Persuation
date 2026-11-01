"""
Zero-Shot NLI & Unsupervised Sentiment Transfer Module (TP2 - Partie B).
Includes:
- B1 : Zero-shot NLI (MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7)
       with 3 language formulations (Arabic, French, English) and validation thresholding.
- B2 : Sentiment-to-Persuasion Transfer (CAMeL-Lab/bert-base-arabic-camelbert-da-sentiment)
       with the heuristic rule: 'négatif => langage chargé (Loaded_Language)'.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

from .trainer_utils import TASK1B_TECHNIQUES

# =====================================================================
# Hypotheses Formulations in 3 Languages (Arabic, French, English)
# =====================================================================
NLI_HYPOTHESES = {
    # Task 1A Formulations: Persuasion presence
    "task1A": {
        "ar": "هذا النص يحتوي على أسلوب إقناعي أو دعائي أو تلاعب بالرأي.",
        "fr": "Ce texte contient une technique de persuasion ou de propagande.",
        "en": "This text contains a persuasion technique or propaganda.",
    },
    # Task 1B Formulations: Definitions of persuasion techniques
    "task1B": {
        "Loaded_Language": {
            "ar": "هذا النص يستخدم لغة عاطفية مشحونة أو كلمات انفعالية للتأثير على القارئ.",
            "fr": "Ce texte utilise un langage chargé d'émotion pour influencer le lecteur.",
            "en": "This text uses loaded language or emotional words to influence the audience.",
        },
        "Name_Calling-Labeling": {
            "ar": "هذا النص يهاجم أو يصف الخصم بألقاب مهينة أو تسميات سلبية.",
            "fr": "Ce texte insulte ou colle des étiquettes négatives à des personnes ou groupes.",
            "en": "This text labels the target with insulting, offensive, or derogatory names.",
        },
        "Appeal_to_Fear-Prejudice": {
            "ar": "هذا النص يثير الخوف أو الأحكام المسبقة لتحذير الجمهور أو ترهيبه.",
            "fr": "Ce texte fait appel à la peur ou aux préjugés pour effrayer ou convaincre le public.",
            "en": "This text appeals to fear or prejudice to create anxiety or panic.",
        },
        "Exaggeration-Minimisation": {
            "ar": "هذا النص يبالغ في تضخيم الأحداث أو يقلل من شأنها بشكل غير واقعي.",
            "fr": "Ce texte exagère ou minimise la réalité de manière excessive.",
            "en": "This text exaggerates or minimizes facts or events unrealistically.",
        },
        "Doubt": {
            "ar": "هذا النص يشكك في مصداقية شخص أو مؤسسة ويثير الريبة حولها.",
            "fr": "Ce texte sème le doute ou remet en question la crédibilité de quelqu'un.",
            "en": "This text casts doubt and questions the credibility of a person or entity.",
        },
        "Questioning_the_Reputation": {
            "ar": "هذا النص يطعن في سمعة الخصم وأخلاقه بدلا من مناقشة حجته.",
            "fr": "Ce texte attaque la réputation ou le caractère de l'adversaire.",
            "en": "This text attacks the reputation and character of an opponent.",
        },
        "Causal_Oversimplification": {
            "ar": "هذا النص يختزل أسبابا معقدة في سبب واحد بسيط ومضلل.",
            "fr": "Ce texte simplifie excessivement les causes d'un problème complexe.",
            "en": "This text oversimplifies a complex issue by attributing it to a single cause.",
        },
        "Appeal_to_Hypocrisy": {
            "ar": "هذا النص يتهم الخصم بالنفاق وازدواجية المعايير (وماذا عنكم).",
            "fr": "Ce texte accuse l'adversaire d'hypocrisie et de double standard.",
            "en": "This text accuses the opponent of hypocrisy to deflect criticism.",
        },
        "Flag_Waving": {
            "ar": "هذا النص يوظف المشاعر الوطنية والقومية لحشد الدعم أو التعاطف.",
            "fr": "Ce texte fait vibrer la fibre patriotique et nationaliste pour rallier les gens.",
            "en": "This text appeals to patriotism or nationalism to mobilize support.",
        },
        "no_technique": {
            "ar": "هذا النص محايد وموضوعي ولا يحتوي على أي أسلوب إقناعي أو دعائي.",
            "fr": "Ce texte est neutre, objectif et sans aucune technique de persuasion.",
            "en": "This text is neutral, factual, and contains no persuasion technique.",
        },
    },
}


# =====================================================================
# B1 : Zero-Shot NLI Predictor & Evaluator
# =====================================================================
def evaluate_nli_task1a_zeroshot(
    val_df: pd.DataFrame,
    lang: str = "ar",
    model_name: str = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7",
    batch_size: int = 16,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Evaluates Zero-Shot NLI on Task 1A with a given language formulation ('ar', 'fr', 'en').
    Searches for the optimal decision threshold on validation data.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    hypothesis = NLI_HYPOTHESES["task1A"].get(lang, NLI_HYPOTHESES["task1A"]["ar"])
    print(f"\n[B1 NLI Zero-Shot 1A] Lang: '{lang.upper()}' | Model: {model_name} | Device: {device}")
    print(f"  Hypothesis: \"{hypothesis}\"")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    model.eval()

    # Find the entailment label index in model config
    entailment_idx = 0
    for idx, name in model.config.id2label.items():
        if "entail" in name.lower():
            entailment_idx = int(idx)
            break

    texts = val_df["text"].astype(str).tolist()
    y_true = (val_df["label"].astype(str).str.lower() == "true").astype(int).values

    # Batch inference
    entailment_scores = []
    for i in tqdm(range(0, len(texts), batch_size), desc=f"NLI 1A ({lang})"):
        batch_texts = texts[i : i + batch_size]
        pairs = [[t, hypothesis] for t in batch_texts]

        encoded = tokenizer(
            pairs,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            logits = model(**encoded).logits
            probs = torch.softmax(logits, dim=-1)
            scores = probs[:, entailment_idx].cpu().numpy()
            entailment_scores.extend(scores)

    entailment_scores = np.array(entailment_scores)

    # Threshold optimization on val
    best_th = 0.5
    best_score = -1.0
    candidate_thresholds = np.linspace(0.1, 0.9, 81)

    for th in candidate_thresholds:
        preds = (entailment_scores >= th).astype(int)
        score = f1_score(y_true, preds, average="macro", zero_division=0)
        if score > best_score:
            best_score = score
            best_th = float(th)

    final_preds = (entailment_scores >= best_th).astype(int)
    acc = accuracy_score(y_true, final_preds)
    f1_bin = f1_score(y_true, final_preds, average="binary", zero_division=0)
    f1_mac = f1_score(y_true, final_preds, average="macro", zero_division=0)
    prec = precision_score(y_true, final_preds, average="binary", zero_division=0)
    rec = recall_score(y_true, final_preds, average="binary", zero_division=0)

    print(f"  Result ({lang}): Optimal Threshold = {best_th:.2f} | Acc: {acc:.4f} | F1-Binary: {f1_bin:.4f} | F1-Macro: {f1_mac:.4f}")

    return {
        "lang": lang,
        "hypothesis": hypothesis,
        "optimal_threshold": best_th,
        "accuracy": float(acc),
        "f1_binary": float(f1_bin),
        "f1_macro": float(f1_mac),
        "precision": float(prec),
        "recall": float(rec),
        "scores": entailment_scores,
    }


def compare_nli_formulations_task1a(
    val_df: pd.DataFrame,
    model_name: str = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7",
) -> pd.DataFrame:
    """
    Compares the 3 language formulations (Arabic, French, English) for Task 1A.
    Directly answers Question Q4 of the TP2 assignment.
    """
    results = []
    for lang in ["ar", "fr", "en"]:
        res = evaluate_nli_task1a_zeroshot(val_df, lang=lang, model_name=model_name)
        results.append({
            "Language": lang.upper(),
            "Optimal Threshold": res["optimal_threshold"],
            "Accuracy": res["accuracy"],
            "F1-Binary": res["f1_binary"],
            "F1-Macro": res["f1_macro"],
            "Precision": res["precision"],
            "Recall": res["recall"],
            "Hypothesis": res["hypothesis"],
        })

    summary_df = pd.DataFrame(results).set_index("Language")
    print("\n================= NLI 3-FORMULATIONS COMPARISON (Q4) =================")
    print(summary_df[["Optimal Threshold", "F1-Macro", "F1-Binary", "Accuracy"]])
    return summary_df


# =====================================================================
# B2 : Sentiment-to-Persuasion Transfer
# =====================================================================
def evaluate_sentiment_transfer_b2(
    val_df: pd.DataFrame,
    model_name: str = "CAMeL-Lab/bert-base-arabic-camelbert-da-sentiment",
    batch_size: int = 16,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Applies B2: CAMeL-Lab/bert-base-arabic-camelbert-da-sentiment
    with rule: 'négatif => langage chargé (Loaded_Language)'.
    Directly answers Question Q5 of the TP2 assignment.
    """
    if device is None:
        device = 0 if torch.cuda.is_available() else -1

    print(f"\n[B2 Sentiment Transfer] Model: {model_name} | Rule: 'négatif => Loaded_Language'")
    sentiment_classifier = pipeline("sentiment-analysis", model=model_name, device=device)

    texts = val_df["text"].astype(str).tolist()
    y_true = [1 if "Loaded_Language" in labels else 0 for labels in val_df.get("labels", [])]

    # Predict sentiments in batches
    preds = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Sentiment B2"):
        batch_texts = texts[i : i + batch_size]
        outputs = sentiment_classifier(batch_texts, truncation=True, max_length=256)
        for out in outputs:
            label = str(out.get("label", "")).lower()
            # Rule: negative => 1 (Loaded_Language), otherwise 0
            preds.append(1 if "neg" in label else 0)

    acc = accuracy_score(y_true, preds)
    prec = precision_score(y_true, preds, zero_division=0)
    rec = recall_score(y_true, preds, zero_division=0)
    f1 = f1_score(y_true, preds, zero_division=0)

    print(f"  Result B2: Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f} | Accuracy: {acc:.4f}")

    return {
        "model": model_name,
        "rule": "négatif => Loaded_Language",
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "accuracy": float(acc),
    }
