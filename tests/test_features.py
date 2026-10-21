import numpy as np

from ftth_access import data
from ftth_access.features import build_preprocessor


def _fitted(raw_df):
    X, y, _ = data.get_X_y(data.clean(raw_df))
    prep = build_preprocessor(data.feature_groups(X.columns)).fit(X, y)
    return prep, X


def test_preprocessor_output_is_numeric_and_named(raw_df):
    prep, X = _fitted(raw_df)
    out = prep.transform(X)
    assert len(out) == len(X)
    assert not out.isna().any().any()
    assert all(np.issubdtype(dtype, np.number) for dtype in out.dtypes)
    assert list(out.columns) == list(prep.get_feature_names_out())
    assert any(c.startswith("mosaiks_pc") for c in out.columns)
    # One-hot : une colonne par modalité, pas un code ordinal.
    assert "TypeLogmt_1_Logement moderne" in out.columns


def test_mosaiks_pca_keeps_requested_variance(raw_df):
    prep, X = _fitted(raw_df)
    pca = prep.named_transformers_["mosaiks"].named_steps["pca"]
    assert pca.explained_variance_ratio_.sum() >= 0.95
    assert pca.n_components_ < len(data.mosaiks_columns(X.columns))


def test_preprocessor_ignores_unknown_categories(raw_df):
    prep, X = _fitted(raw_df)
    X_new = X.head(3).copy()
    X_new["H09_Impute"] = "Modalité jamais vue"
    out = prep.transform(X_new)
    h09 = [c for c in out.columns if c.startswith("H09_Impute_")]
    assert (out[h09].sum(axis=1) == 0).all()
