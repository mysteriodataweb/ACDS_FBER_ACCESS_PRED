"""Catalogue des modèles comparés.

Les classes sont équilibrées (48,8 % / 51,2 %) : aucune pondération de classe
n'est nécessaire.
"""

import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .config import RANDOM_STATE


def get_models(random_state=RANDOM_STATE, n_jobs=-1):
    """Retourne les modèles non entraînés, indexés par leur nom d'affichage."""
    return {
        # La 1re composante MOSAIKS a un écart-type ~50x plus grand que les autres
        # features : sans standardisation, lbfgs ne converge pas.
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000)),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features="sqrt",
            n_jobs=n_jobs,
            random_state=random_state,
        ),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            gamma=0.1,  # gain minimal pour créer un split
            reg_alpha=0.1,  # régularisation L1
            reg_lambda=1.0,  # régularisation L2
            eval_metric="auc",
            n_jobs=n_jobs,
            random_state=random_state,
        ),
        "LightGBM": lgb.LGBMClassifier(
            n_estimators=200,
            max_depth=10,
            learning_rate=0.1,
            num_leaves=31,
            subsample=0.8,
            subsample_freq=1,  # sans ce paramètre, `subsample` est ignoré par LightGBM
            colsample_bytree=0.8,
            min_child_samples=20,
            reg_alpha=0.1,
            reg_lambda=1.0,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=-1,
        ),
        # Sans `probability=True` (5 validations croisées internes, très lent) :
        # l'AUC se calcule sur `decision_function`.
        "SVM": SVC(C=1.0, kernel="rbf", gamma="scale", random_state=random_state),
        "MLP Neural Network": MLPClassifier(
            hidden_layer_sizes=(256, 128),
            alpha=1e-3,
            learning_rate_init=1e-3,
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
            random_state=random_state,
        ),
    }


# Espaces de recherche aléatoire des hyperparamètres (modèles à base d'arbres).
PARAM_DISTRIBUTIONS = {
    "Random Forest": {
        "n_estimators": [200, 300, 500],
        "max_depth": [10, 15, 20, 30, None],
        "min_samples_split": [2, 5, 10, 20],
        "min_samples_leaf": [1, 2, 4, 8],
        "max_features": ["sqrt", "log2", 0.3, 0.5],
    },
    "XGBoost": {
        "n_estimators": [200, 400, 600],
        "max_depth": [4, 6, 8, 10],
        "learning_rate": [0.03, 0.05, 0.1],
        "min_child_weight": [1, 3, 5, 10],
        "subsample": [0.7, 0.8, 1.0],
        "colsample_bytree": [0.5, 0.8, 1.0],
    },
    "LightGBM": {
        "n_estimators": [200, 400, 600],
        "num_leaves": [15, 31, 63, 127],
        "learning_rate": [0.03, 0.05, 0.1],
        "min_child_samples": [5, 10, 20, 50],
        "colsample_bytree": [0.5, 0.8, 1.0],
    },
}
