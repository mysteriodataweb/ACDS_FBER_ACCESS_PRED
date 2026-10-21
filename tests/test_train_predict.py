import json

import pandas as pd

from ftth_access import predict, train


def test_end_to_end_training_and_prediction(raw_parquet, raw_df, tmp_path):
    out = tmp_path / "Output"
    result = train.run(
        data_path=raw_parquet,
        output_dir=out,
        model_names=["Logistic Regression", "Random Forest"],
        tune_iter=2,
        cv_folds=3,
        n_jobs=1,
        log=lambda *_: None,
    )

    assert result.best_model_name in {"Logistic Regression", "Random Forest"}
    assert 0.5 <= result.test_metrics["AUC"] <= 1.0
    for name in ["model_comparison.csv", "predictions_test.csv", "rapport_modele.json"]:
        assert (out / name).exists()
    report = json.loads((out / "rapport_modele.json").read_text(encoding="utf-8"))
    assert report["meilleur_modele"] == result.best_model_name
    assert report["donnees"]["train"] + report["donnees"]["test"] == len(raw_df)

    exported = pd.read_csv(out / "predictions_test.csv")
    assert exported["ID"].str.startswith("ID_").all()
    assert exported["Niveau_Potentiel"].notna().all()

    # Le modèle exporté s'applique directement à des données brutes.
    bundle = predict.load_bundle(out / "models" / train.MODEL_FILENAME)
    new = raw_df.drop(columns=["Accès internet"]).head(25)
    preds = predict.predict_frame(bundle, new)
    assert list(preds.columns) == ["ID", "Probabilité_Accès", "Prédiction", "Niveau_Potentiel"]
    assert preds["Probabilité_Accès"].between(0, 1).all()
    assert (preds["ID"] == new["ID"].to_numpy()).all()


def test_predict_cli(raw_parquet, tmp_path):
    out = tmp_path / "Output"
    train.run(raw_parquet, out, ["Logistic Regression"], cv_folds=3, n_jobs=1, log=lambda *_: None)
    target = tmp_path / "preds.csv"
    predict.main(
        [
            "--input",
            str(raw_parquet),
            "--output",
            str(target),
            "--model",
            str(out / "models" / train.MODEL_FILENAME),
        ]
    )
    assert len(pd.read_csv(target)) == 400
