"""Prédictions sur de nouvelles données : `python -m ftth_access.predict` (ou `ftth-predict`).

Exemple :
    ftth-predict --input Data/nouveaux_menages.parquet --output predictions.csv
"""

import argparse
from pathlib import Path

import joblib

from .config import ID_COL, MODELS_DIR
from .data import clean, load_raw
from .train import MODEL_FILENAME, prediction_table

DEFAULT_MODEL_PATH = MODELS_DIR / MODEL_FILENAME


def load_bundle(path=DEFAULT_MODEL_PATH):
    bundle = joblib.load(path)
    if not isinstance(bundle, dict) or "pipeline" not in bundle:
        raise ValueError(f"{path} n'est pas un modèle exporté par ftth_access.train")
    return bundle


def predict_frame(bundle, df_raw, threshold=None):
    """Applique le pipeline exporté à un DataFrame brut (même format que Data.parquet)."""
    df = clean(df_raw)
    missing = [c for c in bundle["feature_columns"] if c not in df.columns]
    if missing:
        raise KeyError(f"{len(missing)} colonnes attendues absentes, ex. : {missing[:5]}")
    proba = bundle["pipeline"].predict_proba(df[bundle["feature_columns"]])[:, 1]
    threshold = bundle["threshold"] if threshold is None else threshold
    return prediction_table(df[ID_COL], proba, threshold)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", type=Path, required=True, help="Parquet ou CSV de ménages")
    parser.add_argument("--output", type=Path, required=True, help="CSV de sortie")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--threshold", type=float, default=None, help="Seuil de décision")
    args = parser.parse_args(argv)

    predictions = predict_frame(load_bundle(args.model), load_raw(args.input), args.threshold)
    predictions.to_csv(args.output, index=False)
    print(f"{len(predictions):,} prédictions écrites dans {args.output}")


if __name__ == "__main__":
    main()
