# Prédiction de l'accès FTTH au Togo

Projet réalisé par **Alfred Kwami BIAM** dans le cadre du programme **African Citizen Data Scientist (ACDS)**.

L'objectif est d'estimer, pour chaque ménage togolais, la probabilité d'avoir accès à la fibre optique
(*FTTH – Fiber to the Home*) à partir de données socio-démographiques (recensement) et de 4 000 features
géospatiales [MOSAIKS](https://www.mosaiks.org/) extraites d'images satellites.

## Résultats

Split officiel du challenge (80/20 stratifié, `random_state=45`), modèle choisi par validation croisée
sur le train puis évalué une seule fois sur le test :

| Modèle retenu | AUC test | F1 | Accuracy | Précision | Rappel |
|---|---|---|---|---|---|
| Random Forest | 0,874 | 0,830 | 0,828 | 0,801 | 0,862 |

> **À retenir sur la portée du modèle.** Les features MOSAIKS décrivent une tuile satellite, et les
> 30 558 ménages ne couvrent que **423 tuiles** : 99,9 % des ménages du test sont dans une tuile déjà vue à
> l'entraînement. Une simple moyenne du taux d'accès par tuile atteint déjà une AUC de **0,849**, et
> sur des tuiles jamais vues (validation croisée spatiale) l'AUC du modèle tombe à **0,632**
> (une simple régression logistique y fait un peu mieux : 0,662).
> Le modèle est donc fiable pour **prioriser des ménages dans des zones déjà enquêtées**, pas pour
> découvrir de nouvelles zones de déploiement. Détails dans le notebook (section XI) et dans
> [`docs/AUDIT.md`](docs/AUDIT.md).

## Installation

Prérequis : Python ≥ 3.10 et [Git LFS](https://git-lfs.com/) (le jeu de données `Data/Data.parquet`,
150 Mo, est stocké avec LFS).

```bash
git lfs install
git clone https://github.com/mysteriodataweb/ACDS_FBER_ACCESS_PRED.git
cd ACDS_FBER_ACCESS_PRED
git lfs pull                      # télécharge Data/Data.parquet

# Option 1 : pip (dans un environnement virtuel)
python -m venv .venv && source .venv/bin/activate   # Windows : .venv\Scripts\activate
pip install -r requirements.txt

# Option 2 : conda
conda env create -f environment.yml
conda activate FTTH_Togo
```

Le fichier CSV d'origine (~2,2 Go) n'est pas versionné. Si vous partez de lui, placez-le dans
`Data/Data.csv` : la version Parquet est créée automatiquement au premier chargement. Les données du
challenge sont disponibles sur
[Google Drive](https://drive.google.com/drive/folders/1Cu0tUeayegKHuMiAdLgKnpnaebbMBBZY?usp=drive_link).

## Utilisation

**Notebook** (analyse complète, commentée) :

```bash
jupyter lab notebooks/Fiber_Acces_pred.ipynb
```

L'exécution complète prend une vingtaine de minutes sur 4 cœurs et environ 4 Go de RAM.

**Ligne de commande** (même protocole que le notebook) :

```bash
ftth-train --tune 15              # compare les modèles, optimise le meilleur, exporte dans Output/
ftth-predict --input nouveaux_menages.parquet --output predictions.csv
```

`ftth-predict` attend des données au même format que `Data/Data.parquet` (la colonne cible est facultative)
et produit, pour chaque `ID`, la probabilité d'accès, la prédiction (seuil 0,5, modifiable avec
`--threshold`) et un niveau de potentiel (Faible / Moyen / Élevé / Très élevé).

**Tests et lint** :

```bash
pytest
ruff check src tests
```

## Méthodologie

1. **Nettoyage** (`data.clean`) : libellés aux accents corrompus (`�`, `?`) ramenés à leur forme
   correcte, colonnes MOSAIKS renommées `mosaiks_0` … `mosaiks_3999`, équipements `Oui`/`Non` → 1/0.
   Les `ID` ne sont jamais modifiés.
2. **Features** : 5 variables nominales (one-hot), taille du ménage, 33 équipements, et les 4 000 features
   MOSAIKS réduites par PCA (6 composantes, 95 % de variance). `Connexion` est exclue car c'est une fuite
   de la cible (manquante si et seulement si le ménage n'a pas accès) ; `BoxLabel` est constante.
3. **Modèles comparés** : régression logistique, Random Forest, XGBoost, LightGBM, SVM, MLP.
4. **Sélection** : AUC moyenne en validation croisée 5 plis sur le train (prétraitement réappris dans chaque
   pli), puis recherche aléatoire d'hyperparamètres du meilleur modèle.
5. **Évaluation** : test utilisé une seule fois ; validation croisée spatiale groupée par tuile ; calibration ;
   interprétabilité SHAP.
6. **Export** : pipeline complet (prétraitement + modèle) réutilisable sur de nouvelles données.

## Structure du dépôt

```
ACDS_FBER_ACCESS_PRED/
├── Data/
│   ├── Data.parquet              # jeu de données (Git LFS)
│   └── metadonnees.csv           # dictionnaire des variables
├── docs/
│   ├── AUDIT.md                  # limites de la version précédente et corrections
│   └── images/
├── notebooks/
│   └── Fiber_Acces_pred.ipynb    # analyse complète
├── Output/                       # figures, comparaison des modèles, prédictions, rapport JSON
│   └── models/                   # pipeline entraîné (.joblib, non versionné)
├── src/ftth_access/
│   ├── config.py                 # chemins, colonnes, constantes (random_state=45…)
│   ├── data.py                   # chargement et nettoyage
│   ├── features.py               # pipeline de prétraitement
│   ├── models.py                 # modèles et espaces de recherche
│   ├── evaluation.py             # validation croisée, métriques, seuil
│   ├── train.py                  # entraînement de bout en bout (ftth-train)
│   └── predict.py                # prédictions sur de nouvelles données (ftth-predict)
├── tests/                        # tests pytest (données synthétiques)
├── RAPPORT_PRESENTATION.md       # synthèse pour décideurs, opérateurs et data scientists
├── pyproject.toml                # dépendances et configuration
├── requirements.txt
└── environment.yml
```

## Données

| Catégorie | Variables | Description |
|---|---|---|
| Identifiant | `ID` | Identifiant du ménage |
| Logement | `TypeLogmt_1`, `TypeLogmt_2`, `TypeLogmt_3` | Modernité, nombre de logements, nombre de niveaux |
| Énergie | `H08_Impute`, `H09_Impute` | Mode d'éclairage, énergie de cuisson |
| Ménage | `TAILLE_MENAGE` | Nombre de personnes |
| Équipements | `H17*`, `H18*`, `H20*`, `H21*` | Transport, électroménager, audiovisuel, agricole (Oui/Non) |
| Opérateur | `Connexion` | Exclue (fuite de la cible) |
| MOSAIKS | 4 000 colonnes | Features d'images satellites, identiques pour tous les ménages d'une tuile |
| Cible | `Accès internet` | Présence de la fibre dans le ménage (0/1, 48,8 % de 1) |

Le détail de chaque variable est dans [`Data/metadonnees.csv`](Data/metadonnees.csv).

## Dépannage

- **`RuntimeError: ... est un pointeur Git LFS`** : Git LFS n'était pas installé lors du clone ; lancez
  `git lfs install && git lfs pull`.
- **Le kernel n'apparaît pas dans Jupyter / VS Code** :
  `python -m ipykernel install --user --name FTTH_Togo --display-name "Python (FTTH_Togo)"`.
