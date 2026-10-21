import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from ftth_access import data, evaluation
from ftth_access.features import build_preprocessor


def test_potential_level_covers_bounds():
    levels = evaluation.potential_level(np.array([0.0, 0.25, 0.26, 0.5, 0.75, 1.0]))
    assert list(levels) == ["Faible", "Faible", "Moyen", "Moyen", "Élevé", "Très élevé"]
    assert not pd.isna(levels).any()


def test_best_f1_threshold_separates_perfectly_ranked_scores():
    y = np.array([0, 0, 0, 1, 1, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.6, 0.7, 0.8])
    assert 0.3 < evaluation.best_f1_threshold(y, scores) <= 0.6


def test_group_rate_scores_uses_train_rate_and_prior_for_unseen_groups():
    scores = evaluation.group_rate_scores([0, 0, 1, 1], [1, 1, 0, 0], [0, 1, 2], smoothing=0)
    np.testing.assert_allclose(scores, [1.0, 0.0, 0.5])


def test_make_folds_fits_preprocessing_per_fold_and_respects_groups(raw_df):
    df = data.clean(raw_df)
    X, y, _ = data.get_X_y(df)
    groups = data.tile_ids(df)
    prep = build_preprocessor(data.feature_groups(X.columns))
    folds = evaluation.make_folds(prep, X, y, StratifiedGroupKFold(3), groups)
    for fold in folds:
        train_groups = set(groups.iloc[fold.X_train.index])
        valid_groups = set(groups.iloc[fold.valid_index])
        assert not train_groups & valid_groups


def test_cross_validate_models_returns_sorted_summary_and_oof(raw_df):
    X, y, _ = data.get_X_y(data.clean(raw_df))
    prep = build_preprocessor(data.feature_groups(X.columns))
    folds = evaluation.make_folds(prep, X, y, StratifiedKFold(3, shuffle=True, random_state=0))
    models = {"weak": LogisticRegression(C=1e-6), "lr": LogisticRegression(max_iter=1000)}
    summary, oof = evaluation.cross_validate_models(models, folds)
    assert summary["AUC (mean)"].is_monotonic_decreasing
    assert {"AUC (mean)", "AUC (std)", "F1-Score (mean)"} <= set(summary.columns)
    assert not np.isnan(oof["lr"]).any() and len(oof["lr"]) == len(X)
