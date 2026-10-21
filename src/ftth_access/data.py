"""Chargement et nettoyage du jeu de données ménages + MOSAIKS."""

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import (
    BINARY_PREFIXES,
    CSV_PATH,
    ID_COL,
    METADATA_PATH,
    MOSAIKS_PREFIX,
    NOMINAL_COLS,
    NUMERIC_COLS,
    PARQUET_PATH,
    TARGET_COL,
)

_LFS_POINTER_PREFIX = b"version https://git-lfs"

# Les libellés bruts contiennent des accents corrompus ("�" ou "?") selon les
# lignes, ce qui crée de fausses modalités ("Logement � un niveau" et
# "Logement ? un niveau"). On calcule une clé alphanumérique commune à toutes
# les variantes, puis on la ramène au libellé français correct.
CANONICAL_LABELS = {
    # TypeLogmt_1
    "Logementmoderne": "Logement moderne",
    "Logementsemimoderne": "Logement semi-moderne",
    "Logementtraditionnel": "Logement traditionnel",
    # TypeLogmt_2
    "Plusieurslogement": "Plusieurs logements",
    "Logementunique": "Logement unique",
    # TypeLogmt_3
    "Logementunniveauplainpied": "Logement à un niveau (plain-pied)",
    "Logementplusieursniveauxtage": "Logement à plusieurs niveaux (à étage)",
    # H08_Impute (éclairage)
    "ElectricitCEETcompteurdanslaconcession": "Électricité CEET (compteur dans la concession)",
    "ElectricitCEETcompteurhorsconcession": "Électricité CEET (compteur hors concession)",
    "Cblagedlectricitduvoisinnage": "Câblage d'électricité du voisinage",
    "Torchebougie": "Torche/bougie",
    "Lampeptrole": "Lampe à pétrole",
    "Energiesolaire": "Énergie solaire",
    "Lampion": "Lampion",
    "Groupelectrogne": "Groupe électrogène",
    "Lampegaz": "Lampe à gaz",
    # H09_Impute (cuisson)
    "Charbondebois": "Charbon de bois",
    "Gaz": "Gaz",
    "Boisdechauffe": "Bois de chauffe",
    "lectricit": "Électricité",
    "Necuisinepas": "Ne cuisine pas",
    "Ptrole": "Pétrole",
    "Rsidusvgtauxsciuredebois": "Résidus végétaux/sciure de bois",
    "Bousedanimaux": "Bouse d'animaux",
    # Modalités communes
    "Autre": "Autre",
    "Autreprciser": "Autre",
}

_CANONICAL_VALUES = frozenset(CANONICAL_LABELS.values())
_BINARY_VALUES = {"Oui": 1, "Non": 0}


@dataclass(frozen=True)
class FeatureGroups:
    """Colonnes de features regroupées par type de prétraitement."""

    nominal: list
    numeric: list
    binary: list
    mosaiks: list

    @property
    def all(self):
        return self.nominal + self.numeric + self.binary + self.mosaiks


def _is_lfs_pointer(path):
    with open(path, "rb") as f:
        return f.read(len(_LFS_POINTER_PREFIX)) == _LFS_POINTER_PREFIX


def load_raw(path=None):
    """Charge le jeu de données brut (Parquet de préférence, sinon CSV).

    Si seul le CSV est disponible, une copie Parquet est créée à côté pour
    accélérer les chargements suivants.
    """
    path = Path(path) if path is not None else PARQUET_PATH
    if not path.exists() and path == PARQUET_PATH and CSV_PATH.exists():
        df = _read_csv(CSV_PATH)
        df.to_parquet(PARQUET_PATH, compression="snappy", index=False)
        return df
    if not path.exists():
        raise FileNotFoundError(
            f"Fichier de données introuvable : {path}. "
            "Lancez `git lfs pull` ou placez Data.csv dans le dossier Data/."
        )
    if _is_lfs_pointer(path):
        raise RuntimeError(
            f"{path} est un pointeur Git LFS, pas le fichier de données. "
            "Installez Git LFS puis lancez `git lfs pull`."
        )
    if path.suffix == ".csv":
        return _read_csv(path)
    return pd.read_parquet(path)


def _read_csv(path):
    try:
        return pd.read_csv(path, encoding="utf-8", low_memory=False)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin-1", low_memory=False)


def load_variable_labels(path=METADATA_PATH):
    """Dictionnaire code de variable -> description française (ex. H18C -> Réfrigérateur/Congélateur)."""
    meta = pd.read_csv(path, encoding="utf-8")
    return dict(zip(meta["Colonnes"], meta["Descriptions (Fr)"].str.strip(), strict=True))


def normalize_label(value):
    """Ramène un libellé brut (accents corrompus, ponctuation) à sa forme canonique."""
    if not isinstance(value, str) or value in _CANONICAL_VALUES:
        return value
    key = re.sub(r"[^A-Za-z0-9]", "", value)
    return CANONICAL_LABELS.get(key, key)


def _mosaiks_column_names(columns):
    """Associe les colonnes MOSAIKS brutes à des noms explicites `mosaiks_<i>`.

    Dans le CSV d'origine les 4 000 en-têtes MOSAIKS sont vides : pandas les a
    dédoublonnés en " ", " .1", ..., " .3999". La colonne " " est donc la
    feature n°0 et doit être traitée comme les autres.
    """
    mapping = {}
    blank = [c for c in columns if c.strip() == "" or re.fullmatch(r"Unnamed: \d+", c)]
    if len(blank) > 1:
        raise ValueError(f"Plusieurs colonnes sans nom inattendues : {blank!r}")
    for col in blank:
        mapping[col] = f"{MOSAIKS_PREFIX}0"
    for col in columns:
        match = re.fullmatch(r"\s*\.(\d+)", col)
        if match:
            mapping[col] = f"{MOSAIKS_PREFIX}{match.group(1)}"
    return mapping


def binary_columns(columns):
    return [c for c in columns if c.startswith(BINARY_PREFIXES)]


def mosaiks_columns(columns):
    cols = [c for c in columns if c.startswith(MOSAIKS_PREFIX)]
    return sorted(cols, key=lambda c: int(c[len(MOSAIKS_PREFIX) :]))


def clean(df):
    """Nettoie le jeu de données brut. Idempotent, ne modifie jamais les `ID`."""
    df = df.rename(columns=_mosaiks_column_names(df.columns))

    for col in NOMINAL_COLS:
        if col in df.columns:
            df[col] = df[col].map(normalize_label).astype("object")

    for col in binary_columns(df.columns):
        if pd.api.types.is_numeric_dtype(df[col]):
            continue
        mapped = df[col].map(_BINARY_VALUES)
        unexpected = df.loc[mapped.isna() & df[col].notna(), col].unique()
        if len(unexpected):
            raise ValueError(f"Valeurs inattendues dans {col} : {list(unexpected)[:5]}")
        df[col] = mapped.astype("float32") if mapped.isna().any() else mapped.astype("int8")

    mosaiks = mosaiks_columns(df.columns)
    if mosaiks:
        df[mosaiks] = df[mosaiks].astype("float32")

    if TARGET_COL in df.columns and df[TARGET_COL].notna().all():
        df[TARGET_COL] = df[TARGET_COL].astype("int8")
    return df


def feature_groups(columns):
    """Liste explicite des features : tout le reste (ID, cible, fuites) est exclu."""
    columns = list(columns)
    missing = [c for c in (*NOMINAL_COLS, *NUMERIC_COLS) if c not in columns]
    if missing:
        raise KeyError(f"Colonnes attendues absentes : {missing}")
    groups = FeatureGroups(
        nominal=list(NOMINAL_COLS),
        numeric=list(NUMERIC_COLS),
        binary=binary_columns(columns),
        mosaiks=mosaiks_columns(columns),
    )
    if not groups.mosaiks:
        raise KeyError("Aucune colonne MOSAIKS trouvée (appelez `clean` avant).")
    return groups


def get_X_y(df):
    """Sépare features, cible et identifiants d'un DataFrame nettoyé."""
    groups = feature_groups(df.columns)
    return df[groups.all], df[TARGET_COL], df[ID_COL]


def tile_ids(df):
    """Identifiant de tuile satellite : les ménages d'une même tuile partagent
    exactement le même vecteur MOSAIKS. Sert de groupe pour la validation
    croisée spatiale."""
    mosaiks = mosaiks_columns(df.columns)
    hashes = pd.util.hash_pandas_object(df[mosaiks], index=False)
    codes, _ = pd.factorize(hashes)
    return pd.Series(codes.astype(np.int32), index=df.index, name="tile_id")
