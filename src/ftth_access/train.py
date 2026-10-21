"""Entraînement de bout en bout : `python -m ftth_access.train` (ou `ftth-train`).

Protocole :
1. split officiel du challenge (80/20 stratifié, random_state=45) ;
2. comparaison des modèles par validation croisée 5 plis sur le train
   uniquement (le test n'intervient jamais dans le choix du modèle) ;
3. optimisation optionnelle des hyperparamètres du meilleur modèle ;
4. validation croisée spatiale (groupée par tuile satellite) pour estimer
   la performance sur des zones jamais vues ;
5. entraînement final sur tout le train, évaluation unique sur le test ;
6. export du pipeline complet (prétraitement + modèle), du rapport et des
   prédictions.
"""

import argparse
import json
import platform
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import joblib
import lightgbm
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline

from . import __version__
from .config import CV_FOLDS, MODELS_DIR, OUTPUT_DIR, RANDOM_STATE, TEST_SIZE
from .data import clean, feature_groups, get_X_y, load_raw, tile_ids
from .evaluation import (
    best_f1_threshold,
    classification_metrics,
    cross_validate_model,
    cross_validate_models,
    group_rate_scores,
    make_folds,
    potential_level,
    random_search,
)
from .features import build_preprocessor
from .models import PARAM_DISTRIBUTIONS, get_models

MODEL_FILENAME = "ftth_pipeline.joblib"
DEFAULT_THRESHOLD = 0.5


@dataclass
class Dataset:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    ids_train: pd.Series
    ids_test: pd.Series
    groups_train: pd.Series
    groups_test: pd.Series


@dataclass
class TrainingResult:
    best_model_name: str
    pipeline: Pipeline
    cv_summary: pd.DataFrame
    spatial_cv_summary: pd.DataFrame
    search_results: pd.DataFrame | None
    train_metrics: dict
    test_metrics: dict
    predictions: pd.DataFrame
    report: dict


def split_dataset(df):
    """Découpe un DataFrame nettoyé selon le split officiel du challenge."""
    X, y, ids = get_X_y(df)
    groups = tile_ids(df)
    parts = train_test_split(X, y, ids, groups, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
    return Dataset(
        X_train=parts[0],
        X_test=parts[1],
        y_train=parts[2],
        y_test=parts[3],
        ids_train=parts[4],
        ids_test=parts[5],
        groups_train=parts[6],
        groups_test=parts[7],
    )


def prepare_dataset(data_path=None):
    """Charge, nettoie et découpe les données."""
    return split_dataset(clean(load_raw(data_path)))


def with_probabilities(model):
    """Garantit `predict_proba` (nécessaire aux niveaux de potentiel)."""
    if hasattr(model, "predict_proba"):
        return model
    return CalibratedClassifierCV(model, cv=3)


def build_pipeline(preprocessor, model):
    return Pipeline([("preprocess", clone(preprocessor)), ("model", with_probabilities(clone(model)))])


def evaluate(pipeline, X, y, threshold=DEFAULT_THRESHOLD):
    proba = pipeline.predict_proba(X)[:, 1]
    return classification_metrics(y, proba, (proba >= threshold).astype(int))


def prediction_table(ids, proba, threshold=DEFAULT_THRESHOLD, y_true=None):
    table = pd.DataFrame({"ID": np.asarray(ids)})
    if y_true is not None:
        table["Valeur_Réelle"] = np.asarray(y_true)
    table["Probabilité_Accès"] = proba
    table["Prédiction"] = (proba >= threshold).astype(int)
    table["Niveau_Potentiel"] = potential_level(proba)
    return table


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and np.isnan(value):
        return None
    return value


def library_versions():
    return {
        "python": platform.python_version(),
        "ftth_access": __version__,
        "scikit-learn": sklearn.__version__,
        "xgboost": xgboost.__version__,
        "lightgbm": lightgbm.__version__,
        "pandas": pd.__version__,
        "numpy": np.__version__,
    }


def run(
    data_path=None,
    output_dir=OUTPUT_DIR,
    model_names=None,
    tune_iter=0,
    cv_folds=CV_FOLDS,
    n_jobs=-1,
    log=print,
):
    ds = prepare_dataset(data_path)
    log(f"Train : {len(ds.X_train):,} ménages | Test : {len(ds.X_test):,} ménages")

    preprocessor = build_preprocessor(feature_groups(ds.X_train.columns))
    candidates = get_models(n_jobs=n_jobs)
    if model_names:
        candidates = {name: candidates[name] for name in model_names}

    log(f"Validation croisée {cv_folds} plis sur le train ({len(candidates)} modèles)...")
    cv = StratifiedKFold(cv_folds, shuffle=True, random_state=RANDOM_STATE)
    folds = make_folds(preprocessor, ds.X_train, ds.y_train, cv)
    cv_summary, oof_scores = cross_validate_models(candidates, folds)
    log(cv_summary[["AUC (mean)", "AUC (std)", "F1-Score (mean)"]].round(4).to_string())

    best_name = cv_summary.index[0]
    best_model = candidates[best_name]
    best_params = None
    search_results = None
    if tune_iter and best_name in PARAM_DISTRIBUTIONS:
        log(f"Recherche aléatoire ({tune_iter} essais) pour {best_name}...")
        search_results = random_search(
            best_model, PARAM_DISTRIBUTIONS[best_name], folds, tune_iter, RANDOM_STATE
        )
        if search_results.loc[0, "AUC (mean)"] > cv_summary.loc[best_name, "AUC (mean)"]:
            best_params = search_results.loc[0, "params"]
            best_model = clone(best_model).set_params(**best_params)
            _, oof_scores[best_name] = cross_validate_model(best_model, folds)
    log(f"Modèle retenu (meilleure AUC en CV) : {best_name}")

    threshold_f1 = None
    if hasattr(best_model, "predict_proba"):
        threshold_f1 = best_f1_threshold(ds.y_train, oof_scores[best_name])

    log("Validation croisée spatiale (groupes = tuiles satellite)...")
    spatial_cv = StratifiedGroupKFold(cv_folds, shuffle=True, random_state=RANDOM_STATE)
    spatial_folds = make_folds(preprocessor, ds.X_train, ds.y_train, spatial_cv, ds.groups_train)
    spatial_summary, _ = cross_validate_models({best_name: best_model}, spatial_folds)

    pipeline = build_pipeline(preprocessor, best_model).fit(ds.X_train, ds.y_train)
    train_metrics = evaluate(pipeline, ds.X_train, ds.y_train)
    test_metrics = evaluate(pipeline, ds.X_test, ds.y_test)
    test_proba = pipeline.predict_proba(ds.X_test)[:, 1]
    predictions = prediction_table(ds.ids_test, test_proba, y_true=ds.y_test)

    baseline_auc = tile_baseline_auc(ds)
    log(f"Test AUC : {test_metrics['AUC']:.4f} | F1 : {test_metrics['F1-Score']:.4f}")

    report = build_report(
        ds,
        best_name,
        best_params,
        cv_summary,
        spatial_summary,
        train_metrics,
        test_metrics,
        threshold_f1,
        baseline_auc,
        pipeline.named_steps["preprocess"].get_feature_names_out().tolist(),
        cv_folds,
    )
    if output_dir is not None:
        save_artifacts(output_dir, pipeline, best_name, ds.X_train.columns, cv_summary, predictions, report)
        log(f"Artefacts enregistrés dans {output_dir}")

    return TrainingResult(
        best_model_name=best_name,
        pipeline=pipeline,
        cv_summary=cv_summary,
        spatial_cv_summary=spatial_summary,
        search_results=search_results,
        train_metrics=train_metrics,
        test_metrics=test_metrics,
        predictions=predictions,
        report=report,
    )


def tile_baseline_auc(ds):
    """AUC test de la référence « taux d'accès moyen de la tuile » (sans ML)."""
    scores = group_rate_scores(ds.groups_train, ds.y_train, ds.groups_test)
    return classification_metrics(ds.y_test, scores, (scores >= DEFAULT_THRESHOLD).astype(int))["AUC"]


def build_report(
    ds,
    best_name,
    best_params,
    cv_summary,
    spatial_summary,
    train_metrics,
    test_metrics,
    threshold_f1,
    baseline_auc,
    final_features,
    cv_folds=CV_FOLDS,
):
    return _jsonable(
        {
            "projet": "Prédiction de l'accès FTTH - Togo",
            "date_execution": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "protocole": {
                "split": (
                    f"train/test {1 - TEST_SIZE:.0%}/{TEST_SIZE:.0%} stratifié, random_state={RANDOM_STATE}"
                ),
                "selection_modele": f"AUC moyenne en validation croisée {cv_folds} plis sur le train",
                "validation_spatiale": f"StratifiedGroupKFold {cv_folds} plis, groupes = tuiles MOSAIKS",
            },
            "meilleur_modele": best_name,
            "hyperparametres_optimises": best_params,
            "comparaison_cv": cv_summary.to_dict(orient="index"),
            "validation_spatiale": spatial_summary.to_dict(orient="index"),
            "metriques_entrainement": train_metrics,
            "metriques_test": test_metrics,
            "seuil_decision": DEFAULT_THRESHOLD,
            "seuil_f1_optimal_oof": threshold_f1,
            "reference_taux_par_tuile_auc_test": baseline_auc,
            "donnees": {
                "observations": len(ds.X_train) + len(ds.X_test),
                "train": len(ds.X_train),
                "test": len(ds.X_test),
                "tuiles_satellite": int(pd.concat([ds.groups_train, ds.groups_test]).nunique()),
                "part_test_dans_tuiles_vues_au_train": float(np.isin(ds.groups_test, ds.groups_train).mean()),
                "features_brutes": ds.X_train.shape[1],
                "features_apres_pretraitement": len(final_features),
                "noms_features": final_features,
            },
            "versions": library_versions(),
        }
    )


def save_artifacts(output_dir, pipeline, model_name, feature_columns, cv_summary, predictions, report):
    """Écrit le modèle (pipeline complet), la comparaison, les prédictions et le rapport."""
    output_dir = Path(output_dir)
    models_dir = output_dir / MODELS_DIR.name
    models_dir.mkdir(parents=True, exist_ok=True)
    bundle = {
        "pipeline": pipeline,
        "model_name": model_name,
        "threshold": DEFAULT_THRESHOLD,
        "feature_columns": list(feature_columns),
        "trained_at": report["date_execution"],
        "versions": report["versions"],
    }
    joblib.dump(bundle, models_dir / MODEL_FILENAME)
    cv_summary.rename_axis("Modèle").to_csv(output_dir / "model_comparison.csv")
    predictions.to_csv(output_dir / "predictions_test.csv", index=False)
    with open(output_dir / "rapport_modele.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, default=None, help="Parquet ou CSV (défaut : Data/Data.parquet)")
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR, help="Dossier de sortie")
    parser.add_argument("--models", nargs="+", default=None, help="Sous-ensemble de modèles à comparer")
    parser.add_argument("--tune", type=int, default=0, metavar="N", help="Essais de recherche aléatoire")
    parser.add_argument("--cv", type=int, default=CV_FOLDS, help="Nombre de plis")
    parser.add_argument("--n-jobs", type=int, default=-1)
    args = parser.parse_args(argv)
    run(args.data, args.output, args.models, args.tune, args.cv, args.n_jobs)


if __name__ == "__main__":
    main()
