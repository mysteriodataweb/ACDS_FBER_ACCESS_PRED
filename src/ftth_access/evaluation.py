"""Métriques, validation croisée sans fuite et outils d'aide à la décision."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import ParameterSampler

POTENTIAL_BINS = [0.0, 0.25, 0.5, 0.75, 1.0]
POTENTIAL_LABELS = ["Faible", "Moyen", "Élevé", "Très élevé"]


@dataclass
class Fold:
    """Un pli de validation croisée, prétraité avec un préprocesseur appris sur
    sa seule partie entraînement."""

    X_train: pd.DataFrame
    y_train: pd.Series
    X_valid: pd.DataFrame
    y_valid: pd.Series
    valid_index: np.ndarray


def predict_scores(model, X):
    """Score de la classe positive : probabilité si disponible, sinon marge."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def classification_metrics(y_true, y_score, y_pred):
    return {
        "AUC": roc_auc_score(y_true, y_score),
        "Average Precision": average_precision_score(y_true, y_score),
        "F1-Score": f1_score(y_true, y_pred, zero_division=0),
        "Accuracy": accuracy_score(y_true, y_pred),
        "Précision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
    }


def make_folds(preprocessor, X, y, cv, groups=None):
    """Prétraite chaque pli une seule fois (PCA comprise) pour pouvoir y
    évaluer ensuite autant de modèles que voulu, sans fuite entre plis."""
    folds = []
    for train_idx, valid_idx in cv.split(X, y, groups):
        prep = clone(preprocessor).fit(X.iloc[train_idx], y.iloc[train_idx])
        folds.append(
            Fold(
                X_train=prep.transform(X.iloc[train_idx]),
                y_train=y.iloc[train_idx],
                X_valid=prep.transform(X.iloc[valid_idx]),
                y_valid=y.iloc[valid_idx],
                valid_index=valid_idx,
            )
        )
    return folds


def select_features(folds, columns):
    """Mêmes plis restreints à un sous-ensemble de features prétraitées."""
    return [Fold(f.X_train[columns], f.y_train, f.X_valid[columns], f.y_valid, f.valid_index) for f in folds]


def cross_validate_model(model, folds):
    """Retourne les métriques par pli et les scores hors-pli (out-of-fold)."""
    rows = []
    oof = np.full(sum(len(f.valid_index) for f in folds), np.nan)
    for fold in folds:
        fitted = clone(model).fit(fold.X_train, fold.y_train)
        scores = predict_scores(fitted, fold.X_valid)
        rows.append(classification_metrics(fold.y_valid, scores, fitted.predict(fold.X_valid)))
        oof[fold.valid_index] = scores
    return pd.DataFrame(rows), oof


def cross_validate_models(models, folds):
    """Compare plusieurs modèles sur les mêmes plis.

    Retourne un tableau (moyenne et écart-type par métrique), trié par AUC
    moyenne, et le dictionnaire des scores hors-pli par modèle.
    """
    summaries, oof_scores = {}, {}
    for name, model in models.items():
        per_fold, oof_scores[name] = cross_validate_model(model, folds)
        summaries[name] = pd.concat({"mean": per_fold.mean(), "std": per_fold.std()}, axis=0)
    summary = pd.DataFrame(summaries).T
    summary.columns = [f"{metric} ({stat})" for stat, metric in summary.columns]
    return summary.sort_values("AUC (mean)", ascending=False), oof_scores


def random_search(model, param_distributions, folds, n_iter, random_state):
    """Recherche aléatoire d'hyperparamètres évaluée par AUC sur les plis.

    La colonne `params` conserve chaque combinaison avec ses types d'origine
    (les colonnes individuelles servent à l'affichage).
    """
    rows = []
    for params in ParameterSampler(param_distributions, n_iter, random_state=random_state):
        per_fold, _ = cross_validate_model(clone(model).set_params(**params), folds)
        rows.append(
            {
                **params,
                "AUC (mean)": per_fold["AUC"].mean(),
                "AUC (std)": per_fold["AUC"].std(),
                "params": params,
            }
        )
    return pd.DataFrame(rows).sort_values("AUC (mean)", ascending=False, ignore_index=True)


def best_f1_threshold(y_true, proba):
    """Seuil maximisant le F1 (à estimer sur des scores hors-pli, jamais sur le test)."""
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    return float(thresholds[np.argmax(f1[:-1])])


def group_rate_scores(groups_train, y_train, groups_eval, smoothing=10.0):
    """Taux d'accès moyen (lissé) de la tuile de chaque ménage, appris sur le train.

    Référence naïve : mesure ce qu'un modèle obtient en « reconnaissant » la
    tuile satellite sans rien apprendre d'autre.
    """
    prior = float(np.mean(y_train))
    stats = pd.DataFrame({"g": np.asarray(groups_train), "y": np.asarray(y_train)})
    stats = stats.groupby("g")["y"].agg(["sum", "count"])
    rate = (stats["sum"] + smoothing * prior) / (stats["count"] + smoothing)
    return pd.Series(np.asarray(groups_eval)).map(rate).fillna(prior).to_numpy()


def potential_level(proba):
    """Classe une probabilité d'accès en niveau de potentiel (bornes incluses)."""
    return pd.cut(proba, bins=POTENTIAL_BINS, labels=POTENTIAL_LABELS, include_lowest=True)
