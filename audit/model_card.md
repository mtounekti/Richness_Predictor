# Model Card — Richness Predictor

## Informations générales

| Champ | Valeur |
|---|---|
| Nom du modèle | Richness Predictor  |
| Type | Classification binaire (>50K / <=50K) |
| Algorithme | LightGBM (LGBMClassifier) |
| Version | 2.0 |
| Date | Juin 2026 |
| Équipe | FastIA — Module 7 Brief 1 |

## Usage prévu

Ce modèle prédit si une personne gagne plus ou moins de 50 000 $/an à partir de données socio-économiques. Il est conçu à des fins de **recherche et d'expérimentation** dans le cadre d'une formation MLOps

**Usages autorisés**
- Démonstration de pipeline MLOps
- Recherche sur la détection de biais algorithmiques
- Formation et éducation

**Usages interdits**
- Décisions de crédit ou d'accès à des services financiers
- Recrutement ou évaluation de candidats
- Ciblage commercial basé sur le revenu prédit
- Tout système à haut risque au sens de l'AI Act

## Données d'entraînement

| Champ | Valeur |
|---|---|
| Dataset | Adult Census Income (UCI, 1994) |
| Source | Recensement américain — Yann LeCun |
| Taille | 32 561 individus |
| Période | 1994 |
| Split | 80% train / 20% test (stratifié) |

**Variables utilisées (features)**

| Variable | Type | Description |
|---|---|---|
| age | Numérique | Âge de l'individu |
| education.num | Numérique | Niveau d'éducation (1-16) |
| capital.gain | Numérique | Plus-values en capital |
| capital.loss | Numérique | Moins-values en capital |
| hours.per.week | Numérique | Heures travaillées par semaine |
| workclass | Catégorielle | Type d'employeur |
| education | Catégorielle | Niveau d'éducation |
| marital.status | Catégorielle | Situation matrimoniale |
| occupation | Catégorielle | Profession |
| relationship | Catégorielle | Relation familiale |

**Variables exclues pour raisons éthiques**

| Variable | Raison |
|---|---|
| race | Donnée sensible — RGPD Art.9 |
| sex | Donnée sensible — RGPD Art.9 |
| native.country | Donnée sensible — RGPD Art.9 |
| fnlwgt | Poids de recensement, pas une feature individuelle |

## Performances globales

| Métrique | Valeur |
|---|---|
| Accuracy | 86.83% |
| AUC-ROC | 92.23% |
| F1-macro | 81% |
| Temps d'entraînement | 1.25s |
| RAM utilisée | 15.4MB |

## Analyse d'équité (Fairness)

Même en excluant race, sex et native.country des features, le modèle reproduit des biais via des variables corrélées (proxy discrimination). Les résultats ci-dessous sont mesurés sur le jeu de test

### Par sexe

| Groupe | Accuracy | Taux prédiction >50K | Disparate Impact |
|---|---|---|---|
| Female | 93.68% | 8.45% | 0.299 ❌ |
| Male | 83.44% | 26.15% | référence |

### Par race

| Groupe | Accuracy | Taux prédiction >50K |
|---|---|---|
| White | 86.26% | 21.43% |
| Black | 92.06% | 11.74% |
| Asian-Pac-Islander | 84.18% | 20.41% |
| Amer-Indian-Eskimo | 89.55% | 8.96% |

**Disparate Impact Black/White : 0.498 ❌** — en dessous du seuil légal de 0.8

## Limites connues

- Le dataset date de 1994 — il reflète les inégalités socio-économiques de l'époque
- Le Disparate Impact est en dessous du seuil légal de 0.8 pour plusieurs groupes
- Le proxy discrimination persiste malgré la suppression des variables sensibles
- Fairlearn n'a pas été implémenté — le debiasing reste à faire
- Les performances baisseront sur des données plus récentes (dérive temporelle)

## Recommandations avant déploiement en production

- Implémenter Fairlearn (ExponentiatedGradient + DemographicParity) pour corriger le Disparate Impact
- Produire une DPIA (analyse d'impact RGPD) avant tout déploiement
- Limiter l'usage aux contextes non décisionnels
- Mettre en place un monitoring continu des métriques d'équité
- Réentraîner sur des données plus récentes

## Conformité réglementaire

| Réglementation | Statut | Commentaire |
|---|---|---|
| RGPD Art.9 | ✅ Partiel | Variables sensibles supprimées des features |
| RGPD Art.22 | ⚠️ Non conforme | Pas de mécanisme d'explication des décisions |
| AI Act Art.9-15 | ⚠️ Non conforme | Pas de documentation complète ni de journalisation |
| Règle des 4/5 (DI ≥ 0.8) | ❌ Non conforme | DI = 0.299 (sex), 0.498 (race) |
