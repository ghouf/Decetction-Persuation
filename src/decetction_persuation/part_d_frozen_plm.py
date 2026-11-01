"""
TP2 - Partie D: Modèles Pré-entraînés (PLM) Gelés
------------------------------------------------
Ce fichier contient le code pour la Partie D du TP2:
- D1: Modèle BERT gelé (sans entraînement des poids BERT).
      Extraction des représentations [CLS] vs Moyenne des tokens (Mean Pooling),
      suivie d'une régression logistique.
- D2: États de tous les tokens du BERT gelé -> Tête BiLSTM + Attention.
      Visualisation des poids d'attention sur 5 exemples pour voir sur quels
      mots le modèle se concentre.

Le code est écrit de façon directe, simple et facile à lire et comprendre.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import MultiLabelBinarizer
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModel, AutoTokenizer

from decetction_persuation.trainer_utils import TASK1B_TECHNIQUES


# ==============================================================================
# 1. Extraction des Embeddings avec BERT Gelé (D1 & D2)
# ==============================================================================

def get_device() -> torch.device:
    """Retourne GPU (cuda) si disponible, sinon CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def extract_bert_embeddings(
    texts: List[str],
    model_name: str = "aubmindlab/bert-base-arabertv02-twitter",
    pooling: str = "cls",
    batch_size: int = 32,
    max_len: int = 128,
    device: Optional[torch.device] = None,
) -> np.ndarray:
    """
    Passe les textes dans BERT gelé (sans calculer de gradients)
    et extrait un vecteur de dimension 768 pour chaque texte.

    Paramètres:
    -----------
    texts: liste des textes en arabe.
    model_name: nom du modèle HuggingFace.
    pooling: 'cls' (premier token) ou 'mean' (moyenne des tokens).
    """
    if device is None:
        device = get_device()

    print(f"Chargement du modèle gelé: {model_name} sur {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()  # Mode évaluation: aucun dropout, aucun changement de poids

    all_embeddings = []

    # Découpage en mini-lots (batches) pour ne pas saturer la mémoire RAM / GPU
    for i in range(0, len(texts), batch_size):
        batch_texts = [str(t) if str(t).strip() else " " for t in texts[i : i + batch_size]]

        # Tokenisation
        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():  # Pas de gradient = modèle totalement gelé
            outputs = model(**encoded)
            # last_hidden_state a pour forme: [taille_batch, longueur_sequence, 768]
            last_hidden = outputs.last_hidden_state
            mask = encoded["attention_mask"]  # 1 pour les vrais mots, 0 pour le padding

            if pooling.lower() == "cls":
                # Le premier token à l'indice 0 est toujours [CLS]
                batch_vectors = last_hidden[:, 0, :]
            else:
                # Moyenne uniquement sur les vrais tokens (on ignore le padding)
                mask_expanded = mask.unsqueeze(-1).expand_as(last_hidden).float()
                sum_embeddings = (last_hidden * mask_expanded).sum(dim=1)
                sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
                batch_vectors = sum_embeddings / sum_mask

            all_embeddings.append(batch_vectors.cpu().numpy())

    return np.vstack(all_embeddings)


# ==============================================================================
# 2. Partie D1: Comparaison [CLS] vs Moyenne des tokens + Régression Logistique
# ==============================================================================

def run_part_d1(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    task: str = "task1A",
    model_name: str = "aubmindlab/bert-base-arabertv02-twitter",
    batch_size: int = 32,
) -> Dict[str, Dict[str, float]]:
    """
    Exécute la Partie D1:
    1. Extrait les vecteurs avec pooling='cls' -> entraîne Régression Logistique -> évalue sur val.
    2. Extrait les vecteurs avec pooling='mean' -> entraîne Régression Logistique -> évalue sur val.
    3. Affiche et compare les résultats.
    """
    task = task.lower()
    train_texts = train_df["text"].tolist()
    val_texts = val_df["text"].tolist()

    # Préparation des labels
    if "1a" in task:
        # Tâche 1A: Binaire (0 ou 1)
        y_train = (
            train_df.get("label_binary", train_df["label"].astype(str).str.lower() == "true")
            .astype(int)
            .values
        )
        y_val = (
            val_df.get("label_binary", val_df["label"].astype(str).str.lower() == "true")
            .astype(int)
            .values
        )
    else:
        # Tâche 1B: Multi-label (24 classes)
        mlb = MultiLabelBinarizer(classes=TASK1B_TECHNIQUES)
        y_train = mlb.fit_transform(train_df["labels"])
        y_val = mlb.transform(val_df["labels"])

    results = {}

    for method in ["cls", "mean"]:
        print("\n" + "=" * 50)
        print(f"Extraction des embeddings D1 avec méthode: [{method.upper()}]")
        print("=" * 50)

        # Extraction des représentations gelées
        X_train = extract_bert_embeddings(train_texts, model_name=model_name, pooling=method, batch_size=batch_size)
        X_val = extract_bert_embeddings(val_texts, model_name=model_name, pooling=method, batch_size=batch_size)

        print(f"Entraînement de la Régression Logistique ({method.upper()})...")

        if "1a" in task:
            # Classification binaire standard
            clf = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_val)

            metrics = {
                "f1_macro": float(f1_score(y_val, y_pred, average="macro", zero_division=0)),
                "f1_positive": float(f1_score(y_val, y_pred, pos_label=1, zero_division=0)),
                "accuracy": float(accuracy_score(y_val, y_pred)),
                "precision": float(precision_score(y_val, y_pred, zero_division=0)),
                "recall": float(recall_score(y_val, y_pred, zero_division=0)),
            }
        else:
            # Multi-label: 1 régression logistique par classe (One-vs-Rest)
            base_clf = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
            clf = MultiOutputClassifier(base_clf)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_val)

            metrics = {
                "f1_micro": float(f1_score(y_val, y_pred, average="micro", zero_division=0)),
                "f1_macro": float(f1_score(y_val, y_pred, average="macro", zero_division=0)),
                "accuracy": float(accuracy_score(y_val, y_pred)),
            }

        results[method] = metrics
        print(f"Résultats Val pour [{method.upper()}]: {metrics}")

    # Tableau comparatif final
    print("\n" + "=" * 60)
    print(f"RÉSUMÉ COMPARATIF PARTIE D1 ({model_name})")
    print("=" * 60)
    df_compare = pd.DataFrame(results).T
    print(df_compare.to_string())

    return results


# ==============================================================================
# 3. Partie D2: Modèle BiLSTM + Attention sur Tokens Gelés
# ==============================================================================

class BiLSTMAttentionHead(nn.Module):
    """
    Architecture D2:
    Prend les états cachés de tous les tokens sortant de BERT gelé,
    les passe dans un BiLSTM, puis applique une couche d'Attention
    pour agréger les tokens et prédire la classe.
    """

    def __init__(self, input_dim: int = 768, hidden_dim: int = 128, num_classes: int = 2):
        super().__init__()
        # BiLSTM: traite la séquence dans les 2 sens -> sortie de taille hidden_dim * 2
        self.bilstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True,
        )

        # Couche d'attention: calcule un score d'importance pour chaque token
        self.attention_layer = nn.Linear(hidden_dim * 2, 1)

        # Couche de classification finale
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)

    def forward(
        self,
        token_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        token_hidden_states: [batch_size, seq_len, 768] (sortant de BERT gelé)
        attention_mask: [batch_size, seq_len] (1 pour mot, 0 pour padding)

        Retourne:
        - logits: [batch_size, num_classes]
        - attention_weights: [batch_size, seq_len] (poids d'attention de chaque token)
        """
        # 1. Passage dans le BiLSTM
        lstm_out, _ = self.bilstm(token_hidden_states)  # [batch_size, seq_len, hidden_dim * 2]

        # 2. Calcul des scores d'attention
        scores = self.attention_layer(lstm_out).squeeze(-1)  # [batch_size, seq_len]

        if attention_mask is not None:
            # On met un score très négatif aux tokens de padding pour que leur softmax vaille 0
            scores = scores.masked_fill(attention_mask == 0, -1e9)

        # 3. Normalisation des scores en probabilités avec Softmax
        attention_weights = torch.softmax(scores, dim=-1)  # [batch_size, seq_len]

        # 4. Somme pondérée des représentations (Context Vector)
        context = torch.bmm(attention_weights.unsqueeze(1), lstm_out).squeeze(1)

        # 5. Projection vers les logits de sortie
        logits = self.classifier(context)

        return logits, attention_weights


# ==============================================================================
# 4. Entraînement de la Tête D2 (BiLSTM + Attention)
# ==============================================================================

def train_part_d2(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    task: str = "task1A",
    model_name: str = "aubmindlab/bert-base-arabertv02-twitter",
    epochs: int = 5,
    batch_size: int = 32,
    lr: float = 1e-3,
    hidden_dim: int = 128,
) -> Tuple[BiLSTMAttentionHead, AutoTokenizer, Dict[str, float]]:
    """
    Entraîne la tête D2 (BiLSTM + Attention) sur les tokens gelés de BERT.
    Comme BERT est gelé, l'entraînement est rapide.
    """
    device = get_device()
    task = task.lower()

    print(f"\n--- Préparation de la Partie D2 avec {model_name} ---")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    bert = AutoModel.from_pretrained(model_name).to(device)
    bert.eval()  # Gelé à 100%

    # Fonction locale pour extraire tous les états de tokens
    def get_token_states(texts: List[str], max_len: int = 128):
        all_hidden = []
        all_masks = []
        for i in range(0, len(texts), batch_size):
            batch = [str(t) if str(t).strip() else " " for t in texts[i : i + batch_size]]
            enc = tokenizer(batch, padding="max_length", truncation=True, max_length=max_len, return_tensors="pt").to(device)
            with torch.no_grad():
                out = bert(**enc)
                all_hidden.append(out.last_hidden_state.cpu())
                all_masks.append(enc["attention_mask"].cpu())
        return torch.cat(all_hidden, dim=0), torch.cat(all_masks, dim=0)

    print("Extraction des états de tokens pour Train...")
    X_train, masks_train = get_token_states(train_df["text"].tolist())
    print("Extraction des états de tokens pour Val...")
    X_val, masks_val = get_token_states(val_df["text"].tolist())

    # Préparation des étiquettes (labels)
    if "1a" in task:
        num_classes = 2
        y_train_tensor = torch.tensor(
            train_df.get("label_binary", train_df["label"].astype(str).str.lower() == "true").astype(int).values,
            dtype=torch.long,
        )
        y_val_tensor = torch.tensor(
            val_df.get("label_binary", val_df["label"].astype(str).str.lower() == "true").astype(int).values,
            dtype=torch.long,
        )
        criterion = nn.CrossEntropyLoss()
    else:
        num_classes = len(TASK1B_TECHNIQUES)
        mlb = MultiLabelBinarizer(classes=TASK1B_TECHNIQUES)
        y_train_tensor = torch.tensor(mlb.fit_transform(train_df["labels"]), dtype=torch.float32)
        y_val_tensor = torch.tensor(mlb.transform(val_df["labels"]), dtype=torch.float32)
        criterion = nn.BCEWithLogitsLoss()

    train_loader = DataLoader(
        TensorDataset(X_train, masks_train, y_train_tensor),
        batch_size=batch_size,
        shuffle=True,
    )

    # Initialisation de la tête D2
    model_head = BiLSTMAttentionHead(input_dim=768, hidden_dim=hidden_dim, num_classes=num_classes).to(device)
    optimizer = torch.optim.Adam(model_head.parameters(), lr=lr)

    print(f"\nDébut de l'entraînement de la tête D2 ({epochs} époques)...")
    for epoch in range(1, epochs + 1):
        model_head.train()
        total_loss = 0.0

        for batch_hidden, batch_mask, batch_y in train_loader:
            batch_hidden = batch_hidden.to(device)
            batch_mask = batch_mask.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            logits, _ = model_head(batch_hidden, batch_mask)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Époque {epoch}/{epochs} - Perte d'entraînement (Loss): {avg_loss:.4f}")

    # Évaluation sur la validation
    model_head.eval()
    with torch.no_grad():
        val_logits, _ = model_head(X_val.to(device), masks_val.to(device))
        val_logits = val_logits.cpu()

        if "1a" in task:
            y_pred = torch.argmax(val_logits, dim=-1).numpy()
            y_true = y_val_tensor.numpy()
            metrics = {
                "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
                "f1_positive": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
                "accuracy": float(accuracy_score(y_true, y_pred)),
            }
        else:
            probs = torch.sigmoid(val_logits).numpy()
            y_pred = (probs >= 0.5).astype(int)
            y_true = y_val_tensor.numpy()
            metrics = {
                "f1_micro": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
                "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            }

    print(f"\nRésultats finaux D2 (BiLSTM + Attention) sur Val: {metrics}")
    return model_head, tokenizer, metrics


# ==============================================================================
# 5. Visualisation de l'Attention sur 5 Exemples (Consigne D2)
# ==============================================================================

def visualize_attention_on_examples(
    texts: List[str],
    model_head: BiLSTMAttentionHead,
    tokenizer: AutoTokenizer,
    model_name: str = "aubmindlab/bert-base-arabertv02-twitter",
    num_examples: int = 5,
    top_k_words: int = 5,
) -> None:
    """
    Consigne D2 du TP2:
    Visualise les poids d'attention sur 5 exemples réels pour comprendre
    quels mots ou marqueurs déclenchent l'attention du modèle.
    """
    device = get_device()
    print("\n" + "=" * 70)
    print(f"VISUALISATION DE L'ATTENTION SUR {num_examples} EXEMPLES (Partie D2)")
    print("=" * 70)

    bert = AutoModel.from_pretrained(model_name).to(device)
    bert.eval()
    model_head.eval()

    sample_texts = texts[:num_examples]

    for idx, text in enumerate(sample_texts, start=1):
        encoded = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=128,
            padding=True,
        ).to(device)

        with torch.no_grad():
            bert_out = bert(**encoded)
            _, attention_weights = model_head(bert_out.last_hidden_state, encoded["attention_mask"])

        # Récupération des tokens textuels
        input_ids = encoded["input_ids"][0].tolist()
        tokens = tokenizer.convert_ids_to_tokens(input_ids)
        weights = attention_weights[0].cpu().numpy()

        # Filtrage des tokens spéciaux ([CLS], [SEP], [PAD]) pour l'analyse lexicale
        word_scores = []
        for tok, w in zip(tokens, weights):
            if tok not in ["[CLS]", "[SEP]", "[PAD]"]:
                clean_tok = tok.replace("##", "")
                word_scores.append((clean_tok, float(w)))

        # Tri des mots par poids d'attention décroissant
        sorted_words = sorted(word_scores, key=lambda x: x[1], reverse=True)

        print(f"\n--- Exemple #{idx} ---")
        print(f"Texte: {text}")
        print("Top mots les plus importants selon l'Attention:")
        for rank, (word, score) in enumerate(sorted_words[:top_k_words], start=1):
            bar = "█" * int(score * 100)  # Barre visuelle de proportion
            print(f"  {rank}. {word:15} | Poids: {score:.4f} | {bar}")
