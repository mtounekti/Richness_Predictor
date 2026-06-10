# Richness Predictor

Système de classification binaire prédit si une personne gagne **plus ou moins de 50 000 $/an**
à partir du recensement américain (Adult Census Income Dataset, UCI).

Le projet couvre l'entraînement, l'évaluation éthique, le déploiement via API REST
et la supervision en production.

---

## Architecture

```
Richness_Predictor/
├── data/
│   └── adult.csv              Données brutes (48 842 individus, 15 colonnes)
├── model/
│   ├── preprocess.py          Pipeline de prétraitement (nettoyage, encodage)
│   ├── train.py               Entraînement et comparaison de modèles (MLflow)
│   └── evaluate.py            Évaluation des performances et de l'équité
├── api/
│   ├── main.py                API REST FastAPI + métriques Prometheus
│   ├── requirements.txt       Dépendances de l'image Docker API
│   └── Dockerfile             Image Python 3.11-slim
├── artifacts/
│   └── model.joblib           Pipeline sklearn sérialisé (meilleur modèle)
├── mlruns/                    Expériences MLflow (auto-généré)
├── main.py                    Script d'entraînement original (version simplifiée)
├── requirements.txt           Dépendances Python du projet
├── docker-compose.yml         Stack complète (API, MLflow, Prometheus, Grafana)
└── .evn                       Variables d'environnement (credentials Grafana)
```

---

## Démarrage rapide

### 1. Entraîner le modèle

```bash
cd Richness_Predictor
pip install -r requirements.txt
python model/train.py
```

### 2. Évaluer les performances et l'équité

```bash
python model/evaluate.py
```

### 3. Lancer la stack complète (Docker)

```bash
docker compose up --build
```

| Service    | URL                   | Rôle                        |
|------------|-----------------------|-----------------------------|
| API        | http://localhost:8080 | Prédictions REST             |
| MLflow     | http://localhost:5000 | Suivi des expériences        |
| Prometheus | http://localhost:9090 | Collecte des métriques       |
| Grafana    | http://localhost:3000 | Tableaux de bord (admin/admin) |

---

## Modules

### `model/preprocess.py` — Pipeline de prétraitement

Ce module définit les constantes et fonctions de transformation des données.
Il est importé par `train.py`, `evaluate.py` et `api/main.py`.

> **Choix éthique** : `race`, `sex` et `native.country` sont exclus des features
> pour éviter que le modèle n'apprenne des discriminations directes.

---

#### `load_data(csv_path)`

Charge le fichier CSV brut en DataFrame pandas.

| Argument   | Type  | Description                  |
|------------|-------|------------------------------|
| `csv_path` | `str` | Chemin vers le fichier CSV   |

**Retourne** : `pd.DataFrame`

---

#### `clean_data(df)`

Nettoie le DataFrame : remplace les valeurs manquantes encodées `" ?"` par `NaN`,
supprime les lignes incomplètes, supprime les espaces dans les valeurs texte.

| Argument | Type           | Description           |
|----------|----------------|-----------------------|
| `df`     | `pd.DataFrame` | DataFrame brut chargé |

**Retourne** : `pd.DataFrame` nettoyé

---

#### `encode_target(df)`

Convertit la colonne `income` en variable binaire : `>50K → 1`, `<=50K → 0`.

| Argument | Type           | Description                          |
|----------|----------------|--------------------------------------|
| `df`     | `pd.DataFrame` | DataFrame contenant la colonne income |

**Retourne** : `pd.Series` d'entiers (0 ou 1)

---

#### `prepare_features(df)`

Sélectionne les colonnes prédictives (numériques + catégorielles, sans attributs sensibles)
et appelle `encode_target` pour construire la cible.

| Argument | Type           | Description             |
|----------|----------------|-------------------------|
| `df`     | `pd.DataFrame` | DataFrame nettoyé       |

**Retourne** : tuple `(X, y)` — `pd.DataFrame`, `pd.Series`

---

#### `build_pipeline()`

Construit un `ColumnTransformer` sklearn :
- colonnes numériques → `StandardScaler`
- colonnes catégorielles → `OneHotEncoder(handle_unknown="ignore")`

`handle_unknown="ignore"` permet de gérer en production les catégories
absentes de l'entraînement sans lever d'erreur.

**Retourne** : `ColumnTransformer` (preprocesseur non entraîné)

---

#### `save_pipeline(pipeline, path)`

Sérialise un pipeline sklearn sur disque avec joblib.

| Argument   | Type     | Description                              |
|------------|----------|------------------------------------------|
| `pipeline` | Pipeline | Pipeline ou modèle sklearn à sauvegarder |
| `path`     | `str`    | Chemin de destination (défaut : `artifacts/pipeline.joblib`) |

---

#### `load_pipeline(path)`

Charge un pipeline sérialisé depuis le disque.

| Argument | Type  | Description                               |
|----------|-------|-------------------------------------------|
| `path`   | `str` | Chemin du fichier joblib (défaut : `artifacts/pipeline.joblib`) |

**Retourne** : Pipeline sklearn désérialisé

---

### `model/train.py` — Entraînement et comparaison de modèles

Compare trois architectures de classification, logue les résultats dans MLflow
et sauvegarde le meilleur pipeline.

**Modèles comparés** :
- `LogisticRegression` (baseline interprétable)
- `RandomForest-100` (100 arbres, max_depth=15)
- `LightGBM` (200 estimateurs, max_depth=8)

---

#### `init_db()`

Crée la table SQLite `predictions` si elle n'existe pas, pour historiser
les prédictions faites en production.

**Pas d'argument.** Crée le fichier `data/richness.db`.

---

#### `measure_resources(func, *args, **kwargs)`

Mesure le temps d'exécution et la consommation RAM d'une fonction.
Utilisé pour comparer les modèles sur leurs coûts opérationnels.

| Argument   | Type       | Description                        |
|------------|------------|------------------------------------|
| `func`     | `callable` | Fonction à mesurer                 |
| `*args`    | —          | Arguments positionnels de la fonction |
| `**kwargs` | —          | Arguments nommés de la fonction    |

**Retourne** : tuple `(résultat, durée_secondes, delta_ram_mb)`

---

#### `train_and_evaluate(name, model, preprocessor, X_train, X_test, y_train, y_test)`

Assemble un pipeline `preprocessor + model`, l'entraîne, prédit sur le jeu de test
et logue toutes les métriques dans MLflow (accuracy, F1-macro, précision, rappel,
temps d'entraînement, temps d'inférence, RAM).

| Argument      | Type           | Description                              |
|---------------|----------------|------------------------------------------|
| `name`        | `str`          | Nom du modèle (identifiant MLflow)       |
| `model`       | Estimateur     | Modèle sklearn ou compatible             |
| `preprocessor`| ColumnTransformer | Preprocesseur construit par `build_pipeline()` |
| `X_train`     | `pd.DataFrame` | Features d'entraînement                  |
| `X_test`      | `pd.DataFrame` | Features de test                         |
| `y_train`     | `pd.Series`    | Cibles d'entraînement                    |
| `y_test`      | `pd.Series`    | Cibles de test                           |

**Retourne** : tuple `(pipeline, accuracy, durée_train, ram_mb)`

---

#### `main()` *(train.py)*

Orchestre l'entraînement complet :
1. Initialise la base SQLite
2. Charge et nettoie les données
3. Prépare les features et effectue le split 80/20 stratifié
4. Lance `train_and_evaluate` pour chaque modèle candidat
5. Sélectionne le meilleur modèle (accuracy maximale)
6. Sauvegarde le pipeline dans `artifacts/model.joblib`
7. Affiche le tableau comparatif

---

### `model/evaluate.py` — Évaluation des performances et de l'équité

Évalue le modèle entraîné sur ses performances globales **et** sur son comportement
par groupe sensible (sexe, race, pays d'origine).

---

#### `evaluate_performance(pipeline, X_test, y_test)`

Calcule les métriques globales du modèle : accuracy, AUC-ROC, rapport de classification,
matrice de confusion.

| Argument   | Type           | Description                    |
|------------|----------------|--------------------------------|
| `pipeline` | Pipeline       | Pipeline sklearn entraîné      |
| `X_test`   | `pd.DataFrame` | Features du jeu de test        |
| `y_test`   | `pd.Series`    | Étiquettes réelles du jeu de test |

**Retourne** : dict `{accuracy, auc, confusion_matrix}`

---

#### `evaluate_fairness(pipeline, df_test, y_test)`

Analyse les biais du modèle par groupe sensible (`sex`, `race`, `native.country`).
Pour chaque groupe, calcule l'accuracy et le taux de prédictions positives.
Émet un avertissement si l'écart de taux positif dépasse 0.10 (seuil configurable).

> Les colonnes sensibles ne sont **pas** utilisées comme features du modèle,
> mais sont conservées dans `df_test` uniquement pour cet audit.

| Argument   | Type           | Description                                        |
|------------|----------------|----------------------------------------------------|
| `pipeline` | Pipeline       | Pipeline sklearn entraîné                          |
| `df_test`  | `pd.DataFrame` | DataFrame de test incluant les colonnes sensibles  |
| `y_test`   | `pd.Series`    | Étiquettes réelles                                 |

**Retourne** : `pd.DataFrame` avec colonnes `variable, groupe, accuracy, taux_positif, n`

---

#### `save_predictions_to_db(pipeline, X_test, y_test, df_test)`

Exécute les prédictions sur le jeu de test et les persiste dans la table SQLite
`predictions` avec horodatage.

| Argument   | Type           | Description                             |
|------------|----------------|-----------------------------------------|
| `pipeline` | Pipeline       | Pipeline sklearn entraîné               |
| `X_test`   | `pd.DataFrame` | Features du jeu de test                 |
| `y_test`   | `pd.Series`    | Étiquettes réelles (pour alignement)    |
| `df_test`  | `pd.DataFrame` | DataFrame complet (pour récupérer age, workclass…) |

---

#### `main()` *(evaluate.py)*

Charge le pipeline sauvegardé, reconstitue le jeu de test (même split que l'entraînement),
puis enchaîne `evaluate_performance`, `evaluate_fairness` et `save_predictions_to_db`.

---

### `api/main.py` — API REST

Expose le modèle via HTTP. Chaque requête est loguée, mesurée et persistée dans SQLite.
Les métriques Prometheus sont exposées sur `/metrics` pour Grafana.

---

#### Schéma d'entrée — `PredictionInput`

Champs attendus dans le corps de la requête `POST /predict` :

| Champ            | Type    | Description                                  |
|------------------|---------|----------------------------------------------|
| `age`            | `int`   | Âge de la personne                           |
| `workclass`      | `str`   | Secteur d'activité (ex. `Private`)           |
| `education`      | `str`   | Niveau d'éducation (ex. `Bachelors`)         |
| `education_num`  | `int`   | Niveau d'éducation numérique (1–16)          |
| `marital_status` | `str`   | Situation matrimoniale                       |
| `occupation`     | `str`   | Profession (ex. `Tech-support`)              |
| `relationship`   | `str`   | Lien familial (ex. `Husband`)                |
| `capital_gain`   | `float` | Gains en capital                             |
| `capital_loss`   | `float` | Pertes en capital                            |
| `hours_per_week` | `float` | Heures travaillées par semaine               |
| `fnlwgt`         | `float` | Poids de recensement (défaut : 0.0)          |

> `race`, `sex` et `native.country` sont **absents** du schéma d'entrée — le modèle
> ne les utilise pas.

---

#### `input_to_dataframe(data)`

Convertit un objet `PredictionInput` en `pd.DataFrame` avec les noms de colonnes
attendus par le pipeline sklearn (ex. `education.num`, `hours.per.week`).

| Argument | Type              | Description                  |
|----------|-------------------|------------------------------|
| `data`   | `PredictionInput` | Données validées par Pydantic |

**Retourne** : `pd.DataFrame` d'une ligne

---

#### `save_to_db(data, prediction, confidence)`

Insère une prédiction dans la table SQLite `predictions` avec horodatage ISO.

| Argument     | Type              | Description                        |
|--------------|-------------------|------------------------------------|
| `data`       | `PredictionInput` | Données d'entrée de la requête     |
| `prediction` | `int`             | Classe prédite (0 ou 1)            |
| `confidence` | `float`           | Probabilité de la classe prédite   |

---

#### Endpoints

| Méthode | Route      | Description                                        |
|---------|------------|----------------------------------------------------|
| `GET`   | `/`        | Vérifie que l'API est en ligne                     |
| `GET`   | `/health`  | Retourne le statut et le chemin du modèle chargé   |
| `GET`   | `/metrics` | Métriques Prometheus (scrapé automatiquement)      |
| `POST`  | `/predict` | Retourne `prediction` (0/1), `label` et `confidence` |

---

## Variables d'environnement

Fichier `.evn` (à la racine du projet) :

| Variable                | Valeur par défaut        | Usage                             |
|-------------------------|--------------------------|-----------------------------------|
| `GRAFANA_ADMIN_USER`    | `admin`                  | Identifiant Grafana               |
| `GRAFANA_ADMIN_PASSWORD`| `admin`                  | Mot de passe Grafana              |
| `MODEL_PATH`            | `artifacts/model.joblib` | Chemin du pipeline pour l'API     |
| `DB_PATH`               | `data/richness.db`       | Chemin de la base SQLite          |

---

## Dépendances principales

| Package          | Rôle                                  |
|------------------|---------------------------------------|
| scikit-learn     | Pipeline, modèles, métriques          |
| lightgbm         | Modèle gradient boosting              |
| mlflow           | Suivi des expériences                 |
| fastapi          | Framework API REST                    |
| pydantic         | Validation des données d'entrée       |
| prometheus_client| Exposition des métriques              |
| loguru           | Journalisation structurée             |
| joblib           | Sérialisation des modèles             |
| psutil           | Mesure de la consommation RAM         |
