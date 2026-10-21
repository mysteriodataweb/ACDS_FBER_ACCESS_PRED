"""Constantes du projet : chemins, colonnes et paramètres de reproductibilité."""

import os
from pathlib import Path

# Racine du dépôt (surchargeable via la variable d'environnement FTTH_ROOT).
PROJECT_ROOT = Path(os.environ.get("FTTH_ROOT", Path(__file__).resolve().parents[2]))

DATA_DIR = PROJECT_ROOT / "Data"
PARQUET_PATH = DATA_DIR / "Data.parquet"
CSV_PATH = DATA_DIR / "Data.csv"
METADATA_PATH = DATA_DIR / "metadonnees.csv"
OUTPUT_DIR = PROJECT_ROOT / "Output"
MODELS_DIR = OUTPUT_DIR / "models"

# Contraintes du challenge : split 80/20 stratifié avec random_state=45.
RANDOM_STATE = 45
TEST_SIZE = 0.20
CV_FOLDS = 5

ID_COL = "ID"
TARGET_COL = "Accès internet"

# `Connexion` (opérateur) est manquante si et seulement si la cible vaut 0 :
# c'est une fuite de la cible, elle ne doit jamais servir de feature.
LEAKAGE_COLS = ("Connexion",)
# `BoxLabel` ne prend qu'une seule valeur ("Box0") : aucune information.
CONSTANT_COLS = ("BoxLabel",)

NOMINAL_COLS = ("TypeLogmt_1", "TypeLogmt_2", "TypeLogmt_3", "H08_Impute", "H09_Impute")
NUMERIC_COLS = ("TAILLE_MENAGE",)
BINARY_PREFIXES = ("H17", "H18", "H20", "H21")
MOSAIKS_PREFIX = "mosaiks_"

# Part de variance MOSAIKS conservée par la PCA.
PCA_VARIANCE = 0.95
