import numpy as np
import pandas as pd
import pytest

from ftth_access import data
from ftth_access.config import TARGET_COL


def test_clean_keeps_ids_untouched(raw_df):
    cleaned = data.clean(raw_df.copy())
    pd.testing.assert_series_equal(cleaned["ID"], raw_df["ID"])
    assert cleaned["ID"].str.startswith("ID_").all()


def test_clean_merges_corrupted_accent_variants(raw_df):
    cleaned = data.clean(raw_df.copy())
    assert set(cleaned["TypeLogmt_3"]) <= {
        "Logement à un niveau (plain-pied)",
        "Logement à plusieurs niveaux (à étage)",
    }
    assert set(cleaned["H08_Impute"]) <= {
        "Électricité CEET (compteur dans la concession)",
        "Torche/bougie",
    }
    assert "Électricité" in set(cleaned["H09_Impute"])


def test_normalize_label_falls_back_to_key_for_unknown_values():
    assert data.normalize_label("Nouvelle modalité !") == "Nouvellemodalit"
    assert data.normalize_label(np.nan) is np.nan


def test_clean_renames_all_mosaiks_columns_including_blank_one(raw_df):
    cleaned = data.clean(raw_df.copy())
    mosaiks = data.mosaiks_columns(cleaned.columns)
    assert mosaiks[0] == "mosaiks_0"
    assert len(mosaiks) == sum(c.strip() == "" or c.startswith(" .") for c in raw_df.columns)
    assert all(cleaned[c].dtype == np.float32 for c in mosaiks)


def test_clean_encodes_binary_columns(raw_df):
    cleaned = data.clean(raw_df.copy())
    for col in data.binary_columns(cleaned.columns):
        assert set(cleaned[col].unique()) <= {0, 1}
        assert (cleaned[col] == 1).sum() == (raw_df[col] == "Oui").sum()


def test_clean_rejects_unexpected_binary_values(raw_df):
    raw_df.loc[0, "H17A"] = "Peut-être"
    with pytest.raises(ValueError, match="H17A"):
        data.clean(raw_df)


def test_clean_is_idempotent(raw_df):
    once = data.clean(raw_df.copy())
    pd.testing.assert_frame_equal(data.clean(once.copy()), once)


def test_features_exclude_id_target_and_leaking_columns(raw_df):
    X, y, ids = data.get_X_y(data.clean(raw_df))
    for col in ["ID", TARGET_COL, "Connexion", "BoxLabel"]:
        assert col not in X.columns
    assert len(X) == len(y) == len(ids)


def test_tile_ids_group_identical_mosaiks_vectors(raw_df):
    cleaned = data.clean(raw_df)
    tiles = data.tile_ids(cleaned)
    mosaiks = data.mosaiks_columns(cleaned.columns)
    assert tiles.nunique() == cleaned[mosaiks].drop_duplicates().shape[0]
    for _, members in cleaned.groupby(tiles):
        assert members[mosaiks].nunique().max() == 1


def test_load_raw_detects_git_lfs_pointer(tmp_path):
    pointer = tmp_path / "Data.parquet"
    pointer.write_text("version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 1\n")
    with pytest.raises(RuntimeError, match="git lfs pull"):
        data.load_raw(pointer)


def test_variable_labels_cover_all_socio_columns(raw_df):
    labels = data.load_variable_labels()
    groups = data.feature_groups(data.clean(raw_df).columns)
    assert labels["H18C"] == "Réfrigérateur/Congélateur"
    assert all(col in labels for col in groups.nominal + groups.numeric + groups.binary)


def test_load_raw_reads_parquet(raw_parquet, raw_df):
    loaded = data.load_raw(raw_parquet)
    assert loaded.shape == raw_df.shape
