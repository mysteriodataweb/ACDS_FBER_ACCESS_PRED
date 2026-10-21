"""Prétraitement des features, encapsulé dans un ColumnTransformer scikit-learn.

Tout le prétraitement (encodage, standardisation, PCA MOSAIKS) est appris
uniquement sur les données d'entraînement et sauvegardé avec le modèle : le
pipeline exporté s'applique donc directement à de nouvelles données brutes
nettoyées par `data.clean`.
"""

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import PCA_VARIANCE, RANDOM_STATE


class MosaiksPCA(PCA):
    """PCA dont les composantes sont nommées `mosaiks_pc1`, `mosaiks_pc2`, ..."""

    def get_feature_names_out(self, input_features=None):
        return np.array([f"mosaiks_pc{i + 1}" for i in range(self.n_components_)], dtype=object)


def build_preprocessor(groups, pca_variance=PCA_VARIANCE, random_state=RANDOM_STATE):
    """Construit le prétraitement à partir des groupes de colonnes (`data.feature_groups`).

    - variables nominales (type de logement, éclairage, cuisson) : one-hot, car
      un encodage ordinal 0..n imposerait un ordre artificiel aux modèles linéaires ;
    - taille du ménage : imputation médiane + standardisation ;
    - équipements Oui/Non : déjà encodés 0/1, conservés tels quels ;
    - 4 000 features MOSAIKS : standardisation + PCA (95 % de variance).
      Le solveur `covariance_eigh` donne la même PCA que `full` en ~30x moins
      de temps et de mémoire (n_samples >> n_features). Les composantes ne
      sont pas blanchies : le MLP est nettement meilleur ainsi ; la régression
      logistique standardise ses entrées elle-même (voir `models.py`).
    """
    mosaiks = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "pca",
                MosaiksPCA(
                    n_components=pca_variance,
                    svd_solver="covariance_eigh",
                    random_state=random_state,
                ),
            ),
        ]
    )
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    preprocessor = ColumnTransformer(
        [
            ("nominal", OneHotEncoder(handle_unknown="ignore", sparse_output=False), groups.nominal),
            ("numeric", numeric, groups.numeric),
            ("binary", "passthrough", groups.binary),
            ("mosaiks", mosaiks, groups.mosaiks),
        ],
        verbose_feature_names_out=False,
    )
    return preprocessor.set_output(transform="pandas")
