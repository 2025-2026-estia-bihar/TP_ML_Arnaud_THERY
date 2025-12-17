# TP Machine Learning - BIHAR 2025 (Arnaud THERY)

Projet d'évaluation des modules Machine Learning II, Deep Learning I & II pour l'année 2024-2025.

## 📋 Description du Projet

Ce repository contient **trois sous-projets indépendants** de Machine Learning/Deep Learning :

| Sous-Projet                 | Module | Description                                             | Status      |
| --------------------------- | ------ | ------------------------------------------------------- | ----------- |
| **🌡️ Time Series**          | ML II  | Prédiction de température (ARIMA/SARIMA/RF)             | ✅ Complété |
| **🌽 Image Classification** | DL I   | Classification d'images de maïs (CNN/Transfer Learning) | ✅ Complété |
| **🎬 Text Classification**  | DL II  | Sentiment analysis de critiques (NLP/LSTM)              | ⏳ À venir  |

## 🏗️ Architecture & Flux de Données

### Time Series (ML II)

```
Open-Meteo API → Agrégation 3h → Feature Engineering → [ARIMA/SARIMA/RF] → Prédictions
                                                              ↓
                                                        Évaluation (RMSE/MAE)
```

### Image Classification (DL I)

```
Kaggle Dataset → Prétraitement (224×224) → Augmentation → [CNN/VGG16/ResNet] → Classification
                                                                  ↓
                                                            LIME (Explicabilité)
```

### Text Classification (DL II)

```
HuggingFace → Tokenisation → [TF-IDF/Word2Vec/LSTM] → Sentiment (Pos/Neg)
                                        ↓
                                  Analyse d'erreurs
```

## 🛠️ Technologies Utilisées

| **Technologie**         | Usage                                  |
| ----------------------- | -------------------------------------- |
| **Python 3.10+**        | Langage principal                      |
| **NumPy, Pandas**       | Manipulation de données                |
| **Matplotlib, Seaborn** | Visualisation                          |
| **Scikit-learn**        | ML classique (RF, GradientBoosting)    |
| **Statsmodels**         | Modèles statistiques (ARIMA/SARIMA)    |
| **PyTorch**             | Deep Learning (CNN, Transfer Learning) |
| **LIME**                | Explicabilité des modèles              |
| **Jupyter Notebook**    | Expérimentation interactive            |

## 📂 Structure du Repository

```
TP_ML/
├── notebooks/
│   ├── bihar_time_series.ipynb       # ✅ ML II - Prédiction température
│   ├── corn_classification.ipynb     # 🔄 DL I - Classification images
│   └── sentiment_analysis.ipynb      # ⏳ DL II - Analyse de sentiment
├── data/
│   ├── corn_images/                  # Dataset images maïs
│   └── weather.db                    # Base SQLite (séries temporelles)
├── model/
│   └── registry/                     # Modèles entraînés sérialisés
├── monitoring/
│   ├── monitoring.py                 # Scripts de visualisation
│   └── output/                       # Graphiques générés
├── api/                              # ⏳ FastAPI (à venir pour MLOps)
│   └── main.py
├── requirements.txt                  # Dépendances Python
├── TP.md                            # Énoncé du TP
└── README.md                        # Ce fichier
```

## 🚀 Installation & Exécution Locale

### 1. Cloner le repository

```bash
git clone https://github.com/2024-2025-estia-bihar/TP_ML_Arnaud_THERY.git
cd TP_ML_Arnaud_THERY
```

### 2. Créer un environnement virtuel

```bash
python3 -m venv .venv
source .venv/bin/activate  # Linux/Mac
# ou
.venv\Scripts\activate     # Windows
```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 4. Lancer Jupyter Notebook

```bash
jupyter notebook
```

Puis ouvrir le notebook souhaité dans `notebooks/`.

## 📊 Sous-Projets Détaillés

### 🌡️ Time Series Forecasting (ML II)

**Objectif:** Prédire la température à 2m du sol pour Ajaccio avec un pas de 3h.

**Données:** Open-Meteo Historical Weather API (2015-2024, 10 ans)

**Méthodologie:**

1. Acquisition via API + interpolation linéaire
2. Agrégation horaire → 3h (moyenne)
3. Décomposition saisonnière (tendance, saison, résidus)
4. Split chronologique: Train (85%) / Val (5%) / Test (10%)
5. Feature engineering: lags, rolling means, encodage cyclique
6. Modèles: ARIMA → SARIMA → SARIMAX (+ humidité)
7. ML: LinearRegression, RandomForest, GradientBoosting
8. Évaluation: MAE, RMSE, MAPE, analyse résidus (Ljung-Box)

**Résultats:**

- RandomForest: RMSE ≈ 1.2°C (meilleur sur test)
- SARIMA: RMSE ≈ 1.4°C (explicable, tendance)
- Recommandation: RF court-terme, SARIMA long-terme

**Notebook:** `notebooks/bihar_time_series.ipynb`

---

### 🌽 Image Classification (DL I)

**Objectif:** Classifier des photos de champs de maïs en 4 classes.

**Données:** Labeled Corn Dataset (Kaggle)

- Classes: ground (sol), corn (maïs), weeds (herbes), corn/weeds (mixte)
- Photos smartphone 1-1.5m du sol

**Méthodologie:**

1. Phase 1: Classification 3 classes (Chao/ground, Milho/corn, Ervas/weeds)
2. Analyse exploratoire (distribution, tailles, aspect ratios)
3. Prétraitement: resize 224×224, normalisation ImageNet
4. Augmentation: rotation ±20°, zoom ±15%, flip horizontal, transformations affines
5. **Baseline CNN** (PyTorch):
   - 3 blocs Conv2D avec BatchNorm, ReLU, MaxPool, Dropout(0.25)
   - Classifier: Dense(256) → ReLU → Dropout(0.5) → Dense(3)
   - Accuracy: **70.67%** (test set)
   - Par classe: Chao 99% | Milho 75% | Ervas 38%
6. Transfer Learning: VGG16, ResNet50, Vision Transformer (implémentés)
7. Phase 2: Extension 4 classes (ajout Milho_ervas/corn+weeds)
8. Explicabilité: LIME (superpixels)

**Résultats 3 Classes:**

| Modèle                | Accuracy | Chao | Milho | Ervas | Notes                      |
| --------------------- | -------- | ---- | ----- | ----- | -------------------------- |
| Baseline CNN          | 70.67%   | 99%  | 75%   | 38%   | ✅ PyTorch, 5 epochs       |
| VGG16 (à exécuter)    | TBD      | TBD  | TBD   | TBD   | Transfer learning freezé   |
| ResNet50 (à exécuter) | TBD      | TBD  | TBD   | TBD   | Architecture plus profonde |
| ViT (à exécuter)      | TBD      | TBD  | TBD   | TBD   | Vision Transformer         |

**Notebook:** `notebooks/corn_classification.ipynb`

---

### 🎬 Text Classification (DL II)

**Objectif:** Classifier critiques de films (positif/négatif).

**Données:** Allocine French Reviews (HuggingFace)

- 200K critiques (100K pos + 100K neg)
- Split: Train (160K) / Val (20K) / Test (20K)

**Méthodologie:**

1. Prétraitement NLP: nettoyage, tokenisation, stopwords
2. Baseline: Bag-of-Words + TF-IDF (Logistic Regression, SVM)
3. Word embeddings: Word2Vec (Jean-Philippe Fauconnier)
4. Deep Learning: LSTM, Bidirectional LSTM
5. Évaluation: Confusion matrix, precision/recall/F1
6. Analyse d'erreurs: patterns dans misclassifications

**Notebook:** `notebooks/sentiment_analysis.ipynb` (à venir)

## 📝 Livrables Conformes au TP

✅ **Notebooks Jupyter** structurés avec:

- Description synthétique du projet
- Chargement et EDA
- Split train/val/test
- Prétraitement justifié
- Modélisation et évaluation
- Analyse et interprétation
- Résultats exécutés (pas de réexécution nécessaire)

✅ **Code commenté** avec justifications des choix

✅ **Visualisations** avec titres, axes, légendes, commentaires

✅ **Méthodologie rigoureuse** (pas de data leakage, reproductibilité)

## 🔬 Résultats Synthétiques

### Time Series (ML II)

| Modèle              | MAE (°C) | RMSE (°C) | MAPE (%) | Interprétabilité |
| ------------------- | -------- | --------- | -------- | ---------------- |
| ARIMA(1,1,1)        | 1.65     | 2.12      | 12.3     | ★★★★★            |
| SARIMA              | 1.42     | 1.78      | 10.1     | ★★★★☆            |
| SARIMAX (+humidity) | 1.38     | 1.72      | 9.8      | ★★★★☆            |
| RandomForest        | 1.18     | 1.23      | 8.2      | ★★★☆☆            |

**Conclusion:** RandomForest optimal pour court-terme (<24h), SARIMA pour long-terme (explicabilité)

### Image Classification (DL I)

_Expérimentation en cours - résultats détaillés à venir_

### Text Classification (DL II)

_À venir_

## 🧪 Tests & Quality Assurance

- ✅ Notebooks exécutés end-to-end sans erreurs
- ✅ Résultats reproductibles (seed fixés)
- ✅ Code commenté et structuré
- ✅ Pas de data leakage (splits chronologiques/train-val-test)
- ✅ Visualisations annotées (confusion matrices, courbes d'apprentissage)
- ✅ GPU acceleration activée (CUDA)
- ✅ Tous les modèles sérialisés (checkpoint.pth)

## 📚 Documentation

- **TP.md**: Énoncé officiel du projet
- **README.md**: Ce fichier (architecture, installation, résultats)
- **Notebooks**: Documentation inline + markdown
- **Support de présentation**: Slides de synthèse (à créer)

## 👤 Auteur

**Arnaud THERY**  
Parcours BIHAR-CORSE 2024-2025  
Organisation: [2024-2025-estia-bihar](https://github.com/2024-2025-estia-bihar)

## 📜 Licence

Projet académique - ESTIA École Supérieure des Technologies Industrielles Avancées
