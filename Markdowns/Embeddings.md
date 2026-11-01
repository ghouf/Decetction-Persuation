# Guide des Formats et Stratégies d'Agrégation d'Embeddings

Ce document synthétise les représentations vectorielles de texte basées sur les plongements de mots (FastText, GloVe, Word2Vec) et détaille les modes de transformation selon les architectures de modèles.

---

## 1. Vue d'ensemble : Représentation 3D vs 2D

| Type de Représentation | Forme (Shape) | Modèles Cibles | Ordre des mots conservé ? |
| :--- | :--- | :--- | :---: |
| **Séquentielle (3D)** | `(N, max_len, dim)` | BiLSTM, GRU, CNN 1D, Transformers | **Oui** (ordre temporel préservé) |
| **Pooling / Agrégation (2D)** | `(N, dim)` ou `(N, 2 * dim)` | Régression Logistique, SVM, Random Forest, XGBoost | **Non** (résumé en un seul vecteur fixe) |

---

## 2. Le Mode 'sequence' : Tenseur 3D (N, max_len, dim)

### 2.1 Description et analogie

Ce mode conserve **l'ordre temporel des mots** dans chaque phrase. Imaginez un classeur ou un livre :

* **Axe 1 (N) — Les phrases :** Nombre de textes du dataset (ex: 2 535 textes).
* **Axe 2 (max_len) — La position temporelle :** Position du mot dans la phrase (ex: du 1ᵉʳ au 50ᵉ mot).
* **Axe 3 (dim) — La dimension sémantique :** Le vecteur de chaque mot (les 300 nombres décrivant le sens lexical).

```text
Structure du Tenseur 3D : (N, max_len, dim)

               [Mot 1]        [Mot 2]        ...      [Mot 50]
Phrase 1 :  [300 nombres]  [300 nombres]     ...   [300 nombres]
Phrase 2 :  [300 nombres]  [300 nombres]     ...   [300 nombres]
...
Phrase N :  [300 nombres]  [300 nombres]     ...   [300 nombres]
```

---

### 2.2 Pourquoi les BiLSTM, CNN et Transformers ont besoin d'un Tenseur 3D ?

Parce que l'ordre des mots est crucial pour comprendre le sens et la grammaire :

> **Exemple :**  
> *"Ce gouvernement n'est pas corrompu"*  
> $\neq$  
> *"Ce gouvernement est corrompu, pas honnête"*

* **BiLSTM / GRU :** Lit le tenseur mot par mot (du 1ᵉʳ au 50ᵉ) dans les deux sens pour comprendre les négations et le contexte.
* **CNN 1D :** Applique des filtres glissants sur des fenêtres de mots successifs (n-grammes).
* Ces réseaux ont donc absolument besoin de la dimension temporelle (`max_len`).

---

## 3. Les Modes 2D (Pooling) : 'mean', 'max' et 'mean_max'

Les classifieurs classiques (Régression Logistique, SVM, Random Forest) **ne savent pas lire une séquence mot par mot**. Ils exigent un seul vecteur fixe par texte :

$$\text{Matrice 2D : } (N, \text{dim})$$

Pour réduire une phrase de plusieurs mots en un seul vecteur, on utilise le **Pooling (agrégation)**.

---

### Exemple illustratif

Soit une phrase de **3 mots**, avec une dimension d'embedding de **3** :

* **Mot 1 :** `[ 0.2,   0.8,  -0.4 ]`
* **Mot 2 :** `[ 0.6,   0.2,   0.0 ]`
* **Mot 3 :** `[ 0.1,  -0.1,   0.7 ]`

---

### A. Le mode 'mean' (Moyenne arithmétique)

On calcule la moyenne pour chaque colonne :

* **Dim 1 :** $(0.2 + 0.6 + 0.1) / 3 = 0.9 / 3 =$ **`0.3`**
* **Dim 2 :** $(0.8 + 0.2 - 0.1) / 3 = 0.9 / 3 =$ **`0.3`**
* **Dim 3 :** $(-0.4 + 0.0 + 0.7) / 3 = 0.3 / 3 =$ **`0.1`**

```text
Résultat : [0.3, 0.3, 0.1]
Forme    : (N, 300)
```

* **Intuition :** Il capture le **thème global** du texte. C'est le « centre de gravité » de la conversation.
* **Inconvénient :** Il dilue les signaux rares ou les mots isolés très forts.

---

### B. Le mode 'max' (Max-Pooling)

Pour chaque dimension, on retient uniquement la valeur maximale rencontrée parmi tous les mots :

* **Dim 1 :** $\max(0.2,\; 0.6,\; 0.1) =$ **`0.6`**
* **Dim 2 :** $\max(0.8,\; 0.2,\; -0.1) =$ **`0.8`**
* **Dim 3 :** $\max(-0.4,\; 0.0,\; 0.7) =$ **`0.7`**

```text
Résultat : [0.6, 0.8, 0.7]
Forme    : (N, 300)
```

* **Intuition :** C'est un **détecteur de pics / signaux saillants**.
* **Application à la persuasion :** Si un texte contient un seul mot extrêmement marquant ou agressif (*menace, complot, trahison, urgence vitale*), le *mean pooling* risque de le noyer dans la moyenne. Le *max pooling*, lui, retient ce pic immédiatement.

---

### C. Le mode 'mean_max' (Le meilleur des deux mondes)

On calcule le vecteur `mean` (300 nombres) et le vecteur `max` (300 nombres), puis on les concatène côte à côte :

```text
v_final = [  0.3,  0.3,  0.1  |  0.6,  0.8,  0.7  ]
             \_____________/     \_____________/
              Mean (sens global)   Max (mots-clés saillants)
```

* **Forme finale :** `(N, 600)` *(car $300 \times 2 = 600$)*
* **Pourquoi c'est très puissant ?**
  Le classifieur reçoit à la fois :
  1. Le contexte général du texte (via la moyenne).
  2. La présence de mots particulièrement marquants (via le maximum).
* C'est une baseline très réputée et performante en compétition NLP.

---

## 4. Tableau Récapitulatif : Lequel choisir selon l'étape du TP ?

| Étape de votre TP | Modèles testés | Mode recommandé | Forme obtenue | Pourquoi ce choix ? |
| :--- | :--- | :---: | :---: | :--- |
| **Partie A — Baselines classiques** | Régression Logistique, SVM, Random Forest | `'mean_max'` *(ou `'mean'`)* | `(N, 600)` *(ou `(N, 300)`)* | Fournit un vecteur fixe prêt pour scikit-learn. |
| **Partie C — Réseaux profonds** | BiLSTM, GRU, CNN 1D, Transformers | `'sequence'` | `(N, max_len, 300)` | Conserve la séquence temporelle indispensable aux réseaux de neurones. |

---

## 5. Bonnes Pratiques

1. **Masquage du Padding :** Pour `'mean'`, ignorer les zéros de padding pour ne pas fausser la moyenne vers le bas.
2. **Standardisation :** Pour `'mean_max'`, appliquer un `StandardScaler()` ou une normalisation $L_2$ aide les SVM et régressions logistiques à converger plus vite.
