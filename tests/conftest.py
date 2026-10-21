"""Jeu de données synthétique reproduisant le format brut de Data.parquet."""

import numpy as np
import pandas as pd
import pytest

from ftth_access.config import TARGET_COL

N_MOSAIKS = 40
BINARY = [f"H17{c}" for c in "ABY"] + [f"H18{c}" for c in "AGY"] + ["H20A", "H21Y"]

RAW_LABELS = {
    "TypeLogmt_1": ["Logement moderne", "Logement semi-moderne", "Autre"],
    "TypeLogmt_2": ["Plusieurs logement", "Logement unique"],
    "TypeLogmt_3": [
        "Logement \ufffd un niveau (plain-pied)",
        "Logement ? un niveau (plain-pied)",
        "Logement \ufffd plusieurs niveaux (\ufffd \ufffdtage)",
    ],
    "H08_Impute": [
        "Electricit\ufffd (CEET) compteur dans la concession",
        "Electricit? (CEET) compteur dans la concession",
        "Torche/bougie",
    ],
    "H09_Impute": ["Gaz", "Charbon de bois", "\ufffdlectricit\ufffd", "?lectricit?"],
}


def make_raw(n_rows=400, n_tiles=20, seed=0):
    rng = np.random.default_rng(seed)
    tile = rng.integers(0, n_tiles, n_rows)
    tile_vectors = rng.normal(size=(n_tiles, N_MOSAIKS))
    tile_rate = rng.uniform(0.1, 0.9, n_tiles)

    df = pd.DataFrame({"ID": [f"ID_{i:08X}" for i in range(n_rows)]})
    for col, labels in RAW_LABELS.items():
        df[col] = rng.choice(labels, n_rows)
    df["TAILLE_MENAGE"] = rng.integers(1, 12, n_rows).astype(float)
    for col in BINARY:
        df[col] = rng.choice(["Oui", "Non"], n_rows)
    y = (rng.uniform(size=n_rows) < tile_rate[tile]).astype(float)
    df["Connexion"] = np.where(y == 1, rng.choice(["GVA", "Togocom"], n_rows), None)
    df["BoxLabel"] = "Box0"
    mosaiks = pd.DataFrame(tile_vectors[tile], columns=[" "] + [f" .{i}" for i in range(1, N_MOSAIKS)])
    df = pd.concat([df, mosaiks], axis=1)
    df[TARGET_COL] = y
    return df


@pytest.fixture
def raw_df():
    return make_raw()


@pytest.fixture
def raw_parquet(tmp_path, raw_df):
    path = tmp_path / "Data.parquet"
    raw_df.to_parquet(path, index=False)
    return path
