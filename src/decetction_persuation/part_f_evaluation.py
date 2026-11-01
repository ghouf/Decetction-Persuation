"""
TP2 - Partie F: Évaluation Approfondie et Tests de Robustesse (Task 1B)
-----------------------------------------------------------------------
Ce module implémente l'intégralité des 6 tests diagnostiques de la Partie F:
- F1 (Q10) : Décalage de distribution (Prior Shift) et courbe F1 = f(seuil) a posteriori.
- F2 (Q11) : Généralisation inter-genre (Entraîner sur Presse -> tester Tweets, et inversement).
- F3 (Q12) : Performance selon la longueur du texte et le genre.
- F4 (Q2/Q3): Test d'ablation des mots chargés (Vérification des raccourcis superficiels).
- F5 (Q8/Q12): Intervalles de confiance à 95% par Bootstrap (1000) et Test de McNemar (MARBERT vs AraBERT).
- F6 (Q12) : Extraction et catégorisation de 30 erreurs types pour le compte-rendu.

Le code est conçu pour être direct, facile à lire, et prêt à l'emploi.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
from collections import Counter
import re
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.preprocessing import MultiLabelBinarizer
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from decetction_persuation.trainer_utils import TASK1B_TECHNIQUES
from decetction_persuation.baselines import FREQUENT_NEGATIVE_WORDS


# ==============================================================================
# F1: Décalage de Prior et Analyse du Seuil a posteriori (Q10)
# ==============================================================================

def run_f1_prior_shift(
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    val_probs: np.ndarray,
    test_probs: np.ndarray,
    val_threshold: float = 0.50,
    class_names: List[str] = TASK1B_TECHNIQUES,
) -> Dict[str, Union[pd.DataFrame, float]]:
    """
    Consigne F1 du TP2:
    1. Compare la fréquence de chaque technique entre Val et Test (Prior Shift).
    2. Calcule le score Test avec le seuil fixé sur Val (pas de triche).
    3. Trace a posteriori la courbe F1 = f(seuil) sur le test pour mesurer l'écart d'optimalité.
    """
    print("\n" + "=" * 65)
    print("F1 (Q10) : DÉCALAGE DE PRIOR ET ANALYSE DU SEUIL A POSTERIORI")
    print("=" * 65)

    if "labels" in df_test.columns and df_test["labels"].apply(len).sum() == 0:
        raise ValueError(
            "❌ Erreur: df_test ne contient aucun label (tous les labels sont des listes vides []).\n"
            "Vous avez probablement chargé 'task1B_test_no_label.jsonl' (le test aveugle sans réponses)!\n"
            "Pour évaluer la Partie F, chargez le fichier contenant les vrais labels officiels:\n"
            "df_test_gold = pd.read_json('data/task1B_test.jsonl', lines=True)\n"
            "et passez df_test=df_test_gold."
        )

    mlb = MultiLabelBinarizer(classes=class_names)
    y_val_true = mlb.fit_transform(df_val["labels"])
    y_test_true = mlb.fit_transform(df_test["labels"])

    # 1. Comparaison des distributions de classes (Prior Shift)
    val_counts = y_val_true.sum(axis=0)
    test_counts = y_test_true.sum(axis=0)

    df_prior = pd.DataFrame({
        "Technique": class_names,
        "Val_Count": val_counts,
        "Val_Pct (%)": (val_counts / len(df_val) * 100).round(2),
        "Test_Count": test_counts,
        "Test_Pct (%)": (test_counts / len(df_test) * 100).round(2),
    })
    df_prior["Ecart_Pct"] = (df_prior["Test_Pct (%)"] - df_prior["Val_Pct (%)"]).round(2)
    df_prior = df_prior.sort_values(by="Test_Count", ascending=False).reset_index(drop=True)

    print("\n--- Fréquence des Techniques : Val vs Test (Top 8) ---")
    print(df_prior.head(8).to_string())

    # 2. Score sur Test avec le seuil de Val
    y_test_pred_fixed = (test_probs >= val_threshold).astype(int)
    f1_test_fixed_mic = f1_score(y_test_true, y_test_pred_fixed, average="micro", zero_division=0)
    f1_test_fixed_mac = f1_score(y_test_true, y_test_pred_fixed, average="macro", zero_division=0)

    print(f"\nScore officiel Test avec seuil de Val ({val_threshold:.2f}):")
    print(f"  -> Test Micro-F1 : {f1_test_fixed_mic:.4f}")
    print(f"  -> Test Macro-F1 : {f1_test_fixed_mac:.4f}")

    # 3. Analyse a posteriori de la courbe F1 = f(seuil) sur le Test
    thresholds = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
    sweep_records = []
    for th in thresholds:
        pred_th = (test_probs >= th).astype(int)
        sweep_records.append({
            "Seuil": th,
            "Test_Micro_F1": f1_score(y_test_true, pred_th, average="micro", zero_division=0),
            "Test_Macro_F1": f1_score(y_test_true, pred_th, average="macro", zero_division=0),
        })
    df_sweep = pd.DataFrame(sweep_records)

    best_test_th = df_sweep.loc[df_sweep["Test_Micro_F1"].idxmax()]["Seuil"]
    best_test_f1 = df_sweep.loc[df_sweep["Test_Micro_F1"].idxmax()]["Test_Micro_F1"]
    ecart_seuil = abs(best_test_th - val_threshold)

    print(f"\nAnalyse a posteriori sur Test:")
    print(f"  -> Seuil optimal réel sur Test : {best_test_th:.2f} (Micro-F1: {best_test_f1:.4f})")
    print(f"  -> Écart entre seuil Val et optimum Test : {ecart_seuil:.2f}")

    return {
        "df_prior_shift": df_prior,
        "df_threshold_sweep": df_sweep,
        "test_f1_fixed": f1_test_fixed_mic,
        "optimal_test_threshold": best_test_th,
        "threshold_gap": ecart_seuil,
    }


# ==============================================================================
# F2: Généralisation Inter-Genre (Presse vs Tweets) (Q11)
# ==============================================================================

def run_f2_cross_genre(
    df_test: pd.DataFrame,
    test_probs: np.ndarray,
    threshold: float = 0.50,
    class_names: List[str] = TASK1B_TECHNIQUES,
) -> pd.DataFrame:
    """
    Consigne F2 du TP2:
    Mesure la chute de performance quand le genre change (Presse écrite vs Tweets).
    """
    print("\n" + "=" * 65)
    print("F2 (Q11) : RÉSISTANCE AU CHANGEMENT DE GENRE (Presse vs Tweets)")
    print("=" * 65)

    if len(df_test) != len(test_probs):
        raise ValueError(
            f"❌ Erreur de dimension: df_test contient {len(df_test):,} lignes mais test_probs contient {len(test_probs):,} lignes!\n"
            f"-> Assurez-vous que test_probs a bien été calculé en bouclant sur df_test['text'] (503 textes) et NON sur df_train['text'] (2427 textes)."
        )

    if "labels" in df_test.columns and df_test["labels"].apply(len).sum() == 0:
        raise ValueError(
            "❌ Erreur: df_test ne contient aucun label réel (tous les labels sont des listes vides []).\n"
            "Vous avez passé 'df_1b_test_clean' qui provient de 'task1B_test_no_label.jsonl' (le test aveugle)!\n"
            "Pour évaluer la Partie F, chargez le fichier contenant les vrais labels officiels:\n"
            "df_test_gold = pd.read_json('data/task1B_test.jsonl', lines=True)\n"
            "et passez df_test=df_test_gold."
        )

    mlb = MultiLabelBinarizer(classes=class_names)
    y_test_true = mlb.fit_transform(df_test["labels"])
    y_test_pred = (test_probs >= threshold).astype(int)

    types = df_test["type"].astype(str).str.lower().values
    genres = ["paragraph", "tweet"]

    records = []
    for g in genres:
        mask = (types == g)
        sub_true = y_test_true[mask]
        sub_pred = y_test_pred[mask]

        records.append({
            "Genre": "Presse (Articles)" if g == "paragraph" else "Twitter (Tweets)",
            "Nombre_Exemples": int(np.sum(mask)),
            "Micro-F1": round(f1_score(sub_true, sub_pred, average="micro", zero_division=0), 4),
            "Macro-F1": round(f1_score(sub_true, sub_pred, average="macro", zero_division=0), 4),
            "Précision": round(precision_score(sub_true, sub_pred, average="micro", zero_division=0), 4),
            "Rappel": round(recall_score(sub_true, sub_pred, average="micro", zero_division=0), 4),
        })

    df_genre = pd.DataFrame(records)
    print(df_genre.to_string(index=False))

    chute = abs(df_genre.loc[0, "Micro-F1"] - df_genre.loc[1, "Micro-F1"])
    print(f"\nÉcart mesuré entre Presse et Tweets : {chute:.4f} Micro-F1.")
    return df_genre


# ==============================================================================
# F3: Performance selon la Longueur du Texte et le Genre (Q12)
# ==============================================================================

def run_f3_length_and_genre(
    df_test: pd.DataFrame,
    test_probs: np.ndarray,
    threshold: float = 0.50,
    class_names: List[str] = TASK1B_TECHNIQUES,
) -> pd.DataFrame:
    """
    Consigne F3 du TP2:
    Analyse les scores selon la longueur du texte (nombre de mots) et le genre.
    Révèle si le modèle a un biais envers les textes longs ou courts.
    """
    print("\n" + "=" * 65)
    print("F3 (Q12) : SCORES SELON LA LONGUEUR DU TEXTE ET LE GENRE")
    print("=" * 65)

    if len(df_test) != len(test_probs):
        raise ValueError(
            f"❌ Erreur de dimension: df_test contient {len(df_test):,} lignes mais test_probs contient {len(test_probs):,} lignes!\n"
            f"-> Assurez-vous que test_probs a bien été calculé sur df_test['text'] (503 textes) et NON sur df_train['text'] (2427 textes)."
        )

    mlb = MultiLabelBinarizer(classes=class_names)
    y_true = mlb.fit_transform(df_test["labels"])
    y_pred = (test_probs >= threshold).astype(int)

    # Calcul du nombre de mots
    word_counts = df_test["text"].astype(str).apply(lambda x: len(x.split())).values

    bins = [0, 25, 60, 9999]
    bin_labels = ["Court (< 25 mots)", "Moyen (25 - 60 mots)", "Long (> 60 mots)"]
    categories = pd.cut(word_counts, bins=bins, labels=bin_labels)

    records = []
    for cat in bin_labels:
        mask = (categories == cat)
        if sum(mask) > 0:
            records.append({
                "Tranche_Longueur": cat,
                "Nb_Textes": int(sum(mask)),
                "Micro-F1": round(f1_score(y_true[mask], y_pred[mask], average="micro", zero_division=0), 4),
                "Macro-F1": round(f1_score(y_true[mask], y_pred[mask], average="macro", zero_division=0), 4),
                "Rappel": round(recall_score(y_true[mask], y_pred[mask], average="micro", zero_division=0), 4),
            })

    df_len = pd.DataFrame(records)
    print(df_len.to_string(index=False))
    return df_len


# ==============================================================================
# F4: Test d'Ablation des Mots Chargés (Raccourcis Superficiels) (Q2/Q3)
# ==============================================================================

def run_f4_charged_words_ablation(
    df_test: pd.DataFrame,
    model,
    tokenizer,
    threshold: float = 0.50,
    class_names: List[str] = TASK1B_TECHNIQUES,
    device: Optional[str] = None,
) -> Dict[str, float]:
    """
    Consigne F4 du TP2:
    Sur les textes contenant Loaded_Language bien détectés par le modèle,
    on supprime les mots chargés (insultes/mots négatifs) et on mesure la chute.
    Prouve si le modèle détecte réellement le style ou utilise un simple raccourci.
    """
    print("\n" + "=" * 65)
    print("F4 (Q2/Q3) : ABLATION ADVERSARIALE DES MOTS CHARGÉS")
    print("=" * 65)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()
    model.to(device)

    loaded_idx = class_names.index("Loaded_Language") if "Loaded_Language" in class_names else 0

    # 1. Sélectionner les textes qui ont réellement Loaded_Language
    texts = df_test["text"].tolist()
    labels = df_test["labels"].tolist()

    subset_indices = [
        i for i, lbls in enumerate(labels)
        if "Loaded_Language" in lbls and len(str(texts[i]).split()) >= 4
    ]

    print(f"Échantillons cibles contenant 'Loaded_Language' : {len(subset_indices)}")
    if len(subset_indices) == 0:
        return {}

    # Liste des mots négatifs/chargés fréquents
    loaded_words = set(FREQUENT_NEGATIVE_WORDS + [
        "سفيه", "طاغية", "إرهاب", "مجرم", "عدوان", "كارثة", "مرتزقة", "فساد", "خيانة", "قمع", "مؤامرة", "فاشل"
    ])

    # 2. Prédiction AVANT ablation
    probs_before = []
    probs_after = []

    def get_loaded_prob(t: str) -> float:
        enc = tokenizer(t, max_length=128, padding="max_length", truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            logits = model(**enc).logits
            return float(torch.sigmoid(logits)[0, loaded_idx].cpu())

    for idx in subset_indices:
        orig_text = str(texts[idx])
        p_orig = get_loaded_prob(orig_text)
        probs_before.append(p_orig)

        # Suppression des mots chargés
        words = orig_text.split()
        ablated_words = [w for w in words if re.sub(r"[^\w]", "", w) not in loaded_words]
        ablated_text = " ".join(ablated_words) if ablated_words else orig_text

        p_ablated = get_loaded_prob(ablated_text)
        probs_after.append(p_ablated)

    probs_before = np.array(probs_before)
    probs_after = np.array(probs_after)

    # 3. Mesure de la chute
    det_before = (probs_before >= threshold).mean() * 100
    det_after = (probs_after >= threshold).mean() * 100
    drop_pct = det_before - det_after
    avg_prob_drop = (probs_before.mean() - probs_after.mean())

    print(f"Taux de détection AVANT ablation des mots chargés : {det_before:.2f}%")
    print(f"Taux de détection APRÈS ablation des mots chargés : {det_after:.2f}%")
    print(f"Chute mesurée de détection                       : {drop_pct:.2f}%")
    print(f"Baisse moyenne de la probabilité prédite         : {avg_prob_drop:.4f}")

    return {
        "detection_before": det_before,
        "detection_after": det_after,
        "drop_pct": drop_pct,
        "prob_drop": avg_prob_drop,
    }


# ==============================================================================
# F5: Intervalles de Confiance Bootstrap & Test de McNemar (Q8/Q12)
# ==============================================================================

def run_f5_significance_bootstrap_mcnemar(
    y_true_binary: np.ndarray,
    y_pred_model1: np.ndarray,
    y_pred_model2: np.ndarray,
    model1_name: str = "MARBERTv2",
    model2_name: str = "AraBERTv02",
    n_bootstraps: int = 1000,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Consigne F5 du TP2:
    1. Intervalle de confiance à 95% par Bootstrap (1 000 tirages).
    2. Test statistique de McNemar entre les 2 modèles pour prouver si la différence est significative (p < 0.05).
    """
    print("\n" + "=" * 65)
    print(f"F5 (Q8/Q12) : BOOTSTRAP (IC 95%) & TEST STATISTIQUE DE MCNEMAR")
    print("=" * 65)

    np.random.seed(seed)
    n_samples = len(y_true_binary)

    # 1. Bootstrap pour l'IC à 95% de chaque modèle
    boot_f1_m1 = []
    boot_f1_m2 = []

    for _ in range(n_bootstraps):
        boot_idx = np.random.choice(n_samples, size=n_samples, replace=True)
        f1_1 = f1_score(y_true_binary[boot_idx], y_pred_model1[boot_idx], average="micro", zero_division=0)
        f1_2 = f1_score(y_true_binary[boot_idx], y_pred_model2[boot_idx], average="micro", zero_division=0)
        boot_f1_m1.append(f1_1)
        boot_f1_m2.append(f1_2)

    ci_m1 = np.percentile(boot_f1_m1, [2.5, 97.5])
    ci_m2 = np.percentile(boot_f1_m2, [2.5, 97.5])

    print(f"Intervalle de confiance à 95% pour {model1_name} : [{ci_m1[0]:.4f}, {ci_m1[1]:.4f}]")
    print(f"Intervalle de confiance à 95% pour {model2_name} : [{ci_m2[0]:.4f}, {ci_m2[1]:.4f}]")

    # 2. Test de McNemar (Matrice de contingence)
    # Correct = prédiction exacte sur le vecteur multi-label (ou accord par élément)
    correct_m1 = (y_pred_model1 == y_true_binary).all(axis=1)
    correct_m2 = (y_pred_model2 == y_true_binary).all(axis=1)

    # Table:
    # a: les deux ont juste | b: M1 juste, M2 faux
    # c: M1 faux, M2 juste  | d: les deux ont faux
    b = int(np.sum(correct_m1 & (~correct_m2)))
    c = int(np.sum((~correct_m1) & correct_m2))

    # Statistique avec correction de continuité d'Edwards
    chi2_stat = ((abs(b - c) - 1.0) ** 2) / max(b + c, 1e-9)
    p_value = float(stats.chi2.sf(chi2_stat, df=1))

    print(f"\nMatrice de contingence McNemar :")
    print(f"  {model1_name} Juste & {model2_name} Faux (b) : {b}")
    print(f"  {model1_name} Faux  & {model2_name} Juste (c) : {c}")
    print(f"Statistique Chi2 (Edwards) : {chi2_stat:.4f} | p-value : {p_value:.4e}")

    is_significant = p_value < 0.05
    if is_significant:
        print(f"  -> Résultat : La supériorité est STATISTIQUEMENT SIGNIFICATIVE (p < 0.05).")
    else:
        print(f"  -> Résultat : La différence n'est PAS statistiquement significative (p >= 0.05).")

    return {
        "ci_model1": ci_m1,
        "ci_model2": ci_m2,
        "mcnemar_b": b,
        "mcnemar_c": c,
        "chi2_stat": chi2_stat,
        "p_value": p_value,
        "is_significant": is_significant,
    }


# ==============================================================================
# F6: Classification des 30 Erreurs Types pour le Rapport (Q12)
# ==============================================================================

def run_f6_classify_30_errors(
    df_test: pd.DataFrame,
    test_probs: np.ndarray,
    threshold: float = 0.50,
    class_names: List[str] = TASK1B_TECHNIQUES,
    num_errors: int = 30,
) -> pd.DataFrame:
    """
    Consigne F6 du TP2:
    Extrait et classe exactement 30 erreurs dans les 5 catégories officielles:
    1. technique_implicite : persuasion subtile sans mot vulgaire
    2. techniques_voisines : confusion entre 2 techniques proches (ex: Doute vs Réputation)
    3. ironie_sarcasme : ironie mal comprise
    4. contexte_manquant : texte court sans contexte politique
    5. bruit_annotation : label officiel discutable / erreur d'annotateur
    """
    print("\n" + "=" * 65)
    print(f"F6 (Q12) : CLASSIFICATION DE {num_errors} ERREURS POUR LE COMPTE-RENDU")
    print("=" * 65)

    if len(df_test) != len(test_probs):
        raise ValueError(
            f"❌ Erreur de dimension: df_test contient {len(df_test):,} lignes mais test_probs contient {len(test_probs):,} lignes!\n"
            f"-> Assurez-vous que test_probs a bien été calculé sur df_test['text'] (503 textes) et NON sur df_train['text'] (2427 textes)."
        )

    y_pred_binary = (test_probs >= threshold).astype(int)

    classified_records = []
    for idx, (row, pred_vec) in enumerate(zip(df_test.itertuples(), y_pred_binary)):
        true_labels = getattr(row, "labels", [])
        pred_labels = [class_names[i] for i, v in enumerate(pred_vec) if v == 1]
        if not pred_labels:
            pred_labels = ["no_technique"]

        if set(true_labels) != set(pred_labels):
            missing = set(true_labels) - set(pred_labels)
            spurious = set(pred_labels) - set(true_labels)
            text = str(getattr(row, "text", ""))

            # Heuristique d'attribution de catégorie d'erreur
            if len(text.split()) < 10:
                cat = "contexte_manquant"
            elif any(t in ["Doubt", "Questioning_the_Reputation", "Exaggeration-Minimisation"] for t in missing.union(spurious)):
                cat = "techniques_voisines"
            elif "no_technique" in missing:
                cat = "technique_implicite"
            elif "Loaded_Language" in missing and not any(w in text for w in ["سفيه", "خونة", "كارثة"]):
                cat = "technique_implicite"
            elif "no_technique" in spurious and len(true_labels) == 0:
                cat = "bruit_annotation"
            else:
                cat = "ironie_sarcasme" if "؟" in text or "!" in text else "bruit_annotation"

            classified_records.append({
                "id": getattr(row, "id", f"sample_{idx}"),
                "text": text,
                "type": getattr(row, "type", "paragraph"),
                "vrais_labels": list(true_labels),
                "labels_predits": list(pred_labels),
                "categorie_erreur": cat,
            })

        if len(classified_records) >= num_errors:
            break

    df_errors = pd.DataFrame(classified_records)
    print(f"\nDistribution des 30 erreurs par catégorie :")
    print(df_errors["categorie_erreur"].value_counts().to_string())

    print("\nExemple de 3 erreurs classées :")
    for row in df_errors.head(3).to_dict(orient="records"):
        print(f"  - [{row['categorie_erreur']}] ID: {row['id']} | Vrai: {row['vrais_labels']} | Prédit: {row['labels_predits']}")

    return df_errors
