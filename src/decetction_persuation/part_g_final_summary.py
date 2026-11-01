"""
TP2 - Partie G: Tableau Récapitulatif Final & Matrice de Confusion
------------------------------------------------------------------
Ce module implémente la dernière partie du TP2 (Partie G) :
1. Calcul des spécifications système du modèle :
   - Taille du modèle sur disque (Mo) et nombre de paramètres entraînés.
   - Latence d'inférence réelle sur CPU (millisecondes par exemple).
2. Matrice de confusion multi-label du modèle retenu (visualisation graphique).
3. Tableau final comparatif (Qualité, Robustesse, Coût, Temps, Paramètres)
   et journalisation automatique dans results/results.csv (consigne officielle).
"""

from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import f1_score, multilabel_confusion_matrix, precision_score, recall_score
from sklearn.preprocessing import MultiLabelBinarizer
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from decetction_persuation.trainer_utils import TASK1B_TECHNIQUES


# ==============================================================================
# 1. Mesure de Latence CPU et Taille du Modèle
# ==============================================================================

def measure_model_efficiency(
    model,
    tokenizer,
    sample_texts: List[str],
    n_runs: int = 50,
) -> Dict[str, Any]:
    """
    Mesure les métriques d'ingénierie et de déploiement en production :
    - Nombre de paramètres totaux et entraînables.
    - Taille estimée en Mo.
    - Latence d'inférence CPU par échantillon (ms).
    """
    print("\n--- Mesure de l'Efficacité et du Coût de Déploiement (CPU) ---")

    # 1. Comptage des paramètres
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    size_mb = (total_params * 4) / (1024 * 1024)  # 4 octets par float32

    # 2. Mesure de la latence sur CPU
    cpu_device = torch.device("cpu")
    model_cpu = model.to(cpu_device)
    model_cpu.eval()

    test_samples = sample_texts[:min(n_runs, len(sample_texts))]
    latencies = []

    # Warmup
    dummy = tokenizer(test_samples[0], return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        _ = model_cpu(**dummy)

    # Chronométrage
    for text in test_samples:
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=128, padding=True)
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model_cpu(**enc)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)  # Convertir en millisecondes

    avg_latency_ms = float(np.mean(latencies))
    p95_latency_ms = float(np.percentile(latencies, 95))

    print(f"  -> Paramètres Totaux : {total_params:,} ({total_params/1e6:.1f}M)")
    print(f"  -> Taille en Mémoire : {size_mb:.2f} Mo")
    print(f"  -> Latence Moyenne CPU : {avg_latency_ms:.2f} ms / texte (P95: {p95_latency_ms:.2f} ms)")

    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "size_mb": size_mb,
        "latency_cpu_ms": avg_latency_ms,
        "latency_p95_ms": p95_latency_ms,
    }


# ==============================================================================
# 2. Matrice de Confusion Multi-Label du Modèle Retenu
# ==============================================================================

def plot_multilabel_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str] = TASK1B_TECHNIQUES,
    top_k_classes: int = 6,
    save_path: Optional[str] = "results/confusion_matrix_part_g.png",
) -> None:
    """
    Consigne Partie G :
    Trace la matrice de confusion multi-label (TP, FP, FN, TN)
    pour les classes les plus fréquentes du corpus.
    """
    print(f"\n--- Génération de la Matrice de Confusion Multi-Label (Top {top_k_classes}) ---")

    # Calcul des matrices de confusion 2x2 pour chaque label
    mcm = multilabel_confusion_matrix(y_true, y_pred)

    # Sélectionner les classes avec le plus d'occurrences réelles
    supports = y_true.sum(axis=0)
    top_indices = np.argsort(supports)[::-1][:top_k_classes]

    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    axes = axes.flatten()

    for idx, class_idx in enumerate(top_indices):
        ax = axes[idx]
        cm = mcm[class_idx]
        c_name = class_names[class_idx]

        # [[TN, FP], [FN, TP]]
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            ax=ax,
            xticklabels=["Prédit 0", "Prédit 1"],
            yticklabels=["Réel 0", "Réel 1"],
        )
        ax.set_title(f"{c_name}\n(Positifs réels : {int(supports[class_idx])})", fontsize=11, fontweight="bold")

    plt.suptitle("Matrices de Confusion Multi-Label - Modèle Retenu (Partie G)", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"Matrice de confusion enregistrée dans : {save_path}")

    plt.show()


# ==============================================================================
# 3. Tableau Récapitulatif Final & Export results/results.csv
# ==============================================================================

def generate_part_g_summary_table(
    models_records: List[Dict[str, Any]],
    save_csv_path: str = "results/results.csv",
) -> pd.DataFrame:
    """
    Génère le Tableau Final de la Partie G et enregistre le journal results/results.csv.
    Colonnes exigées par le sujet :
    [Modèle, Prétraitement, Graine, Micro-F1, Macro-F1, Robustesse (Chute), Temps_Train (s), Latence_CPU (ms), Taille (Mo), Paramètres]
    """
    df_results = pd.DataFrame(models_records)

    print("\n" + "=" * 90)
    print("TABLEAU FINAL COMPARATIF (TP2 - PARTIE G)")
    print("=" * 90)
    print(df_results.to_string(index=False))

    # Sauvegarde dans results/results.csv
    out_path = Path(save_csv_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_results.to_csv(out_path, index=False, encoding="utf-8")
    print(f"\n✅ Résultats journalisés dans : {out_path.resolve()}")

    return df_results


# ==============================================================================
# 4. Pipeline Clé-en-main pour la Partie G
# ==============================================================================

def run_part_g(
    model,
    tokenizer,
    df_test_gold: pd.DataFrame,
    test_probs: np.ndarray,
    threshold: float = 0.50,
    model_name: str = "MARBERTv2 (Asymmetric Loss)",
    train_time_sec: float = 1600.0,
    genre_drop_pct: float = 4.2,
    seed: int = 42,
    save_results_csv: str = "results/results.csv",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Fonction globale pour exécuter toute la Partie G en un seul appel.
    """
    mlb = MultiLabelBinarizer(classes=TASK1B_TECHNIQUES)
    y_test_true = mlb.fit_transform(df_test_gold["labels"])
    y_test_pred = (test_probs >= threshold).astype(int)

    # 1. Calcul des métriques de qualité sur test
    mic_f1 = f1_score(y_test_true, y_test_pred, average="micro", zero_division=0)
    mac_f1 = f1_score(y_test_true, y_test_pred, average="macro", zero_division=0)
    prec = precision_score(y_test_true, y_test_pred, average="micro", zero_division=0)
    rec = recall_score(y_test_true, y_test_pred, average="micro", zero_division=0)

    # 2. Mesure de l'efficacité (Taille, Latence CPU, Paramètres)
    specs = measure_model_efficiency(
        model=model,
        tokenizer=tokenizer,
        sample_texts=df_test_gold["text"].tolist(),
        n_runs=40,
    )

    # 3. Tracé de la matrice de confusion
    plot_multilabel_confusion_matrix(
        y_true=y_test_true,
        y_pred=y_test_pred,
        class_names=TASK1B_TECHNIQUES,
        top_k_classes=6,
        save_path="results/confusion_matrix_part_g.png",
    )

    # 4. Construction de la ligne du tableau final
    record = {
        "Modèle": model_name,
        "Prétraitement": "Nettoyage arabe + Outliers (<100 mots)",
        "Graine": seed,
        "Test_Micro_F1": round(mic_f1, 4),
        "Test_Macro_F1": round(mac_f1, 4),
        "Précision": round(prec, 4),
        "Rappel": round(rec, 4),
        "Robustesse (Chute Genre F2)": f"{genre_drop_pct:.2f}%",
        "Temps_Train (s)": round(train_time_sec, 1),
        "Latence_CPU (ms)": round(specs["latency_cpu_ms"], 2),
        "Taille (Mo)": round(specs["size_mb"], 1),
        "Nb_Paramètres": f"{specs['total_params']:,}",
    }

    # 5. Export vers results/results.csv
    df_summary = generate_part_g_summary_table([record], save_csv_path=save_results_csv)

    return df_summary, specs
