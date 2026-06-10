# Model Card — Richness Predictor v2.0

## Informations générales

| Champ | Valeur |
|---|---|
| Nom du modèle | Richness Predictor v2.0 |
| Type | Classification binaire (>50K / <=50K) |
| Algorithme | LightGBM (LGBMClassifier) |
| Version | 2.0 |
| Date | Juin 2026 |
| Équipe | FastIA — Module 7 Brief 1 |

## Usage prévu

Ce modèle prédit si une personne gagne plus ou moins de 50 000 $/an à partir de données socio-économiques. Il est utilisé dans un contexte de **ciblage commercial** — identifier des clients potentiels pour des offres de services.

**Usages autorisés**
- Démonstration de pipeline MLOps
- Recherche sur la détection de biais algorithmiques
- Formation et éducation

**Usages interdits**
- Ciblage commercial tant que le Disparate Impact n'est pas corrigé (DI < 0.8)
- Décisions de crédit ou d'accès à des services financiers
- Recrutement ou évaluation de candidats
- Tout système à haut risque au sens de l'AI Act sans supervision humaine

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

### Pourquoi la mesure d'équité dépend de l'usage

La mesure de fairness à appliquer dépend directement de l'usage du modèle :

| Usage | Mesure appropriée | Question posée |
|---|---|---|
| Ciblage commercial (notre cas) | **Demographic Parity** | Tout le monde a-t-il la même chance d'être ciblé ? |
| Détection de fraude / maladie | **Equalized Odds** | Le modèle est-il aussi précis pour tous les groupes ? |
| Décision individuelle | **Individual Fairness** | Deux individus similaires reçoivent-ils la même prédiction ? |

Dans notre cas — ciblage commercial — la mesure appropriée est le **Demographic Parity** et son indicateur le **Disparate Impact (DI)**.

```
DI = taux de prédiction positive du groupe défavorisé
     ─────────────────────────────────────────────────
     taux de prédiction positive du groupe de référence
```

Le seuil de 0.8 vient de la règle des 4/5 définie par l'EEOC (Equal Employment Opportunity Commission) en 1978, issue de la jurisprudence américaine sur la discrimination à l'embauche. Ce seuil n'est pas codifié en droit européen — en Europe, le RGPD et l'AI Act n'imposent pas de seuil chiffré mais exigent une évaluation qualitative au cas par cas. En pratique, la règle des 4/5 est reprise comme bonne pratique internationale par la communauté MLOps et les organismes comme le NIST. Un DI ≥ 0.8 est donc la référence la plus utilisée même en dehors des États-Unis.

### Résultats

Même en excluant race, sex et native.country des features, le modèle reproduit des biais via des variables corrélées — c'est le **proxy discrimination** : occupation, relationship et education révèlent indirectement le sexe et la race.

**Par sexe**

| Groupe | Accuracy | Taux prédiction >50K | Disparate Impact |
|---|---|---|---|
| Female | 93.68% | 8.45% | **0.299 ❌** |
| Male | 83.44% | 26.15% | référence |

**Par race**

| Groupe | Accuracy | Taux prédiction >50K | Disparate Impact |
|---|---|---|---|
| White | 86.26% | 21.43% | référence |
| Black | 92.06% | 11.74% | **0.498 ❌** |
| Asian-Pac-Islander | 84.18% | 20.41% | 0.952 ✅ |
| Amer-Indian-Eskimo | 89.55% | 8.96% | **0.418 ❌** |

Les deux groupes principaux sont en dessous du seuil légal de 0.8.

### Pourquoi Fairlearn ne suffit pas

Même avec Fairlearn (ExponentiatedGradient + DemographicParity), un biais résiduel persistera car :

- Les données elles-mêmes datent de 1994 et reflètent les inégalités de l'époque
- On ne peut pas corriger complètement un biais qui vient des données historiques
- Il existe un théorème d'impossibilité de fairness : on ne peut pas satisfaire simultanément Demographic Parity, Equalized Odds et Individual Fairness

### Solutions concrètes selon le problème

| Problème | Solution | Impact |
|---|---|---|
| Demographic Parity trop faible | Fairlearn ExponentiatedGradient + DemographicParity | Légère baisse d'accuracy |
| Proxy discrimination | Supprimer les variables corrélées (relationship, occupation) | Baisse de performance |
| Biais dans les données | Collecter de nouvelles données plus récentes et équilibrées | Recommandé en priorité |
| Décision à fort impact | Supervision humaine obligatoire | Non négociable |

### Recommandation principale

**Ce modèle ne doit pas être utilisé pour du ciblage commercial en l'état.** Le Disparate Impact de 0.299 sur le genre signifie que les femmes ont 3 fois moins de chances d'être ciblées que les hommes — ce qui constitue une discrimination algorithmique au sens de la règle des 4/5.

La vraie solution est de :
1. Collecter des données plus récentes et représentatives
2. Appliquer Fairlearn comme première correction technique
3. Maintenir une supervision humaine sur toute décision commerciale

## Limites connues

- Le dataset date de 1994 — dérive temporelle importante
- Fairlearn non implémenté — le debiasing reste à faire
- DPIA (analyse d'impact RGPD) non produite
- Les performances baisseront sur des données plus récentes

## Conformité réglementaire

| Réglementation | Statut | Commentaire |
|---|---|---|
| RGPD Art.9 | ✅ Partiel | Variables sensibles supprimées des features |
| RGPD Art.22 | ⚠️ Non conforme | Pas de mécanisme d'explication des décisions |
| AI Act Art.9-15 | ⚠️ Non conforme | Pas de documentation complète ni de journalisation |
| Règle des 4/5 (DI ≥ 0.8) | ❌ Non conforme | DI = 0.299 (sex), 0.498 (race) |