"""
Visualisations et Analyse Diagnostique pour Task 1B (TP2 Partie D, E & F)
-------------------------------------------------------------------------
Ce fichier fournit des fonctions prêtes à l'emploi et très faciles à lire
pour analyser et visualiser les 4 grands problèmes de la détection de persuasion:

1. plot_threshold_tradeoff :
   Courbe montrant pourquoi le seuil fait monter le Micro-F1 et chuter le Macro-F1.

2. plot_per_class_f1 :
   Graphique montrant le score F1 de chacune des 24 classes (révèle la longue traîne).

3. plot_predictions_distribution :
   Compare le nombre de prédictions du modèle par rapport à la réalité (Ground Truth).

4. inspect_classification_errors :
   Extrait les pires erreurs du modèle avec le texte pour l'analyse qualitative (Partie F6).
"""

from typing import List, Optional, Tuple, Union
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.preprocessing import MultiLabelBinarizer

from decetction_persuation.trainer_utils import TASK1B_TECHNIQUES


# Configuration visuelle pour des graphiques propres et lisibles
sns.set_theme(style="whitegrid")
plt.rcParams["font.sans-serif"] = ["Segoe UI", "DejaVu Sans", "Arial"]


def plot_threshold_tradeoff(
    val_probs: np.ndarray,
    val_targets: np.ndarray,
    thresholds: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Trace la courbe du compromis selon le seuil de décision (0.10 à 0.70).
    Montre clairement le croisement entre:
    - Micro-F1 (maximisé à 0.50)
    - Macro-F1 (maximisé à 0.35)
    - Précision et Rappel
    """
    if thresholds is None:
        thresholds = np.linspace(0.10, 0.70, 31)

    records = []
    for th in thresholds:
        y_pred = (val_probs >= th).astype(int)
        mic_f1 = f1_score(val_targets, y_pred, average="micro", zero_division=0)
        mac_f1 = f1_score(val_targets, y_pred, average="macro", zero_division=0)
        prec = precision_score(val_targets, y_pred, average="micro", zero_division=0)
        rec = recall_score(val_targets, y_pred, average="micro", zero_division=0)

        records.append({
            "Seuil": th,
            "Micro-F1": mic_f1,
            "Macro-F1": mac_f1,
            "Précision (Micro)": prec,
            "Rappel (Micro)": rec,
        })

    df_metrics = pd.DataFrame(records)

    # Création de la figure
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(df_metrics["Seuil"], df_metrics["Micro-F1"], label="Micro-F1 (Métrique Officielle)", color="#1f77b4", linewidth=2.5)
    ax.plot(df_metrics["Seuil"], df_metrics["Macro-F1"], label="Macro-F1 (Moyenne des 24 classes)", color="#ff7f0e", linewidth=2.5, linestyle="--")
    ax.plot(df_metrics["Seuil"], df_metrics["Précision (Micro)"], label="Précision", color="#2ca02c", alpha=0.7)
    ax.plot(df_metrics["Seuil"], df_metrics["Rappel (Micro)"], label="Rappel", color="#d62728", alpha=0.7)

    # Marquer les sommets (pics)
    best_mic_row = df_metrics.loc[df_metrics["Micro-F1"].idxmax()]
    best_mac_row = df_metrics.loc[df_metrics["Macro-F1"].idxmax()]

    ax.scatter(best_mic_row["Seuil"], best_mic_row["Micro-F1"], color="#1f77b4", s=90, zorder=5)
    ax.annotate(
        f"Pic Micro-F1: {best_mic_row['Micro-F1']:.2%}\n(Seuil = {best_mic_row['Seuil']:.2f})",
        xy=(best_mic_row["Seuil"], best_mic_row["Micro-F1"]),
        xytext=(best_mic_row["Seuil"] - 0.12, best_mic_row["Micro-F1"] + 0.05),
        arrowprops=dict(arrowstyle="->", color="#1f77b4", lw=1.5),
        fontweight="bold"
    )

    ax.scatter(best_mac_row["Seuil"], best_mac_row["Macro-F1"], color="#ff7f0e", s=90, zorder=5)
    ax.annotate(
        f"Pic Macro-F1: {best_mac_row['Macro-F1']:.2%}\n(Seuil = {best_mac_row['Seuil']:.2f})",
        xy=(best_mac_row["Seuil"], best_mac_row["Macro-F1"]),
        xytext=(best_mac_row["Seuil"] - 0.15, best_mac_row["Macro-F1"] + 0.08),
        arrowprops=dict(arrowstyle="->", color="#ff7f0e", lw=1.5),
        fontweight="bold"
    )

    ax.set_title("Compromis du Seuil de Décision : Micro-F1 vs Macro-F1 (Task 1B)", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Seuil de Décision (Threshold)", fontsize=12)
    ax.set_ylabel("Score (0.0 à 1.0)", fontsize=12)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlim(0.10, 0.70)
    ax.legend(loc="center right", frameon=True)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Graphique sauvegardé dans: {save_path}")

    plt.show()
    return df_metrics


def plot_per_class_f1(
    val_probs: np.ndarray,
    val_targets: np.ndarray,
    threshold: float = 0.50,
    class_names: List[str] = TASK1B_TECHNIQUES,
    save_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Calcule et affiche un diagramme en barres horizontales du score F1
    pour CHACUNE des 24 classes. Révèle immédiatement les classes à F1 = 0.
    """
    y_pred = (val_probs >= threshold).astype(int)

    f1_per_class = []
    support_per_class = []

    for i, c_name in enumerate(class_names):
        f1_c = f1_score(val_targets[:, i], y_pred[:, i], zero_division=0)
        support = int(val_targets[:, i].sum())
        f1_per_class.append(f1_c)
        support_per_class.append(support)

    df_classes = pd.DataFrame({
        "Technique": class_names,
        "F1_Score": f1_per_class,
        "Exemples_Reels": support_per_class
    }).sort_values(by="F1_Score", ascending=True).reset_index(drop=True)

    # Visualisation
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ["#d62728" if score == 0.0 else "#1f77b4" for score in df_classes["F1_Score"]]
    bars = ax.barh(df_classes["Technique"], df_classes["F1_Score"], color=colors, height=0.65)

    # Affichage du score et du nombre d'exemples
    for bar, score, count in zip(bars, df_classes["F1_Score"], df_classes["Exemples_Reels"]):
        ax.text(
            bar.get_width() + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{score:.2f} (n={count})",
            va="center",
            fontsize=9
        )

    ax.set_title(f"Score F1 par Technique (Seuil = {threshold}) - Rouge = F1 Nul", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("F1 Score", fontsize=12)
    ax.set_xlim(0.0, 1.05)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Graphique sauvegardé dans: {save_path}")

    plt.show()
    return df_classes


def plot_predictions_distribution(
    true_labels_list: List[List[str]],
    pred_labels_list: List[List[str]],
    class_names: List[str] = TASK1B_TECHNIQUES,
    save_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Compare côte à côte la fréquence réelle des techniques vs la fréquence prédite
    par le modèle sur l'ensemble de validation ou de test.
    """
    true_counts = Counter([lbl for sub in true_labels_list for lbl in sub])
    pred_counts = Counter([lbl for sub in pred_labels_list for lbl in sub])

    df_comp = pd.DataFrame({
        "Technique": class_names,
        "Réel (Ground Truth)": [true_counts.get(c, 0) for c in class_names],
        "Prédit par le Modèle": [pred_counts.get(c, 0) for c in class_names],
    }).sort_values(by="Réel (Ground Truth)", ascending=False).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(df_comp))
    width = 0.38

    ax.bar(x - width/2, df_comp["Réel (Ground Truth)"], width, label="Réel", color="#2ca02c")
    ax.bar(x + width/2, df_comp["Prédit par le Modèle"], width, label="Prédit", color="#1f77b4")

    ax.set_xticks(x)
    ax.set_xticklabels(df_comp["Technique"], rotation=70, ha="right", fontsize=9)
    ax.set_ylabel("Nombre d'Occurrences")
    ax.set_title("Distribution des Techniques: Fréquence Réelle vs Prédictions", fontsize=14, fontweight="bold")
    ax.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Graphique sauvegardé dans: {save_path}")

    plt.show()
    return df_comp


def inspect_classification_errors(
    df_val: pd.DataFrame,
    val_probs: np.ndarray,
    threshold: float = 0.50,
    class_names: List[str] = TASK1B_TECHNIQUES,
    num_errors: int = 10,
) -> pd.DataFrame:
    """
    Extrait et affiche les textes sur lesquels le modèle s'est le plus trompé.
    Parfait pour la rédaction de la Partie F6 (Analyse d'erreurs).
    """
    y_pred_binary = (val_probs >= threshold).astype(int)

    errors = []
    for idx, (row, pred_vec) in enumerate(zip(df_val.itertuples(), y_pred_binary)):
        true_labels = getattr(row, "labels", [])
        pred_labels = [class_names[i] for i, val in enumerate(pred_vec) if val == 1]
        if not pred_labels:
            pred_labels = ["no_technique"]

        # Si les prédictions sont différentes de la vérité terrain
        if set(true_labels) != set(pred_labels):
            missing = list(set(true_labels) - set(pred_labels))     # Faux Négatifs
            spurious = list(set(pred_labels) - set(true_labels))    # Faux Positifs

            errors.append({
                "id": getattr(row, "id", f"sample_{idx}"),
                "text": getattr(row, "text", ""),
                "true_labels": true_labels,
                "pred_labels": pred_labels,
                "manquants (FN)": missing,
                "inventes (FP)": spurious,
                "nb_fautes": len(missing) + len(spurious)
            })

    # Trier par nombre de fautes pour avoir les pires erreurs en premier
    df_err = pd.DataFrame(errors).sort_values(by="nb_fautes", ascending=False).reset_index(drop=True)

    print(f"\nNombre total d'erreurs détectées sur Val : {len(df_err)} / {len(df_val)}")
    print(f"--- Affichage des {min(num_errors, len(df_err))} pires erreurs pour la Partie F6 ---")

    for i in range(min(num_errors, len(df_err))):
        err = df_err.iloc[i]
        print(f"\n[Erreur #{i+1}] ID: {err['id']}")
        print(f"  Texte: {err['text']}")
        print(f"  Vrai(s):       {err['true_labels']}")
        print(f"  Prédit(s):     {err['pred_labels']}")
        if err["manquants (FN)"]:
            print(f"  ❌ Raté (FN):  {err['manquants (FN)']}")
        if err["inventes (FP)"]:
            print(f"  ⚠️  Faux (FP):  {err['inventes (FP)']}")

    return df_err
