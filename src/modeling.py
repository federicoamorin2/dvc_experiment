"""Module containing Machiene learning pipeline
"""

import re
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import typer
import yaml
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve,
)
from sklearn.model_selection import train_test_split

PARAMS = yaml.safe_load(open("params.yaml"))
app = typer.Typer()


@app.command()
def split_data(
    raw_db_path: Path,
    train_df_path: Path,
    test_df_path: Path,
) -> None:
    """Split data in to train and test sets.

    Args:
        raw_db_path (Path): Path to the raw database.
    """
    raw_db = pd.read_csv(raw_db_path)

    params = PARAMS["split"]
    features, targets = raw_db.drop("target", axis=1), raw_db["target"]

    train_df, test_df, y_train, y_test = train_test_split(
        features, targets, random_state=305, test_size=params["test_size"]
    )
    train_df["target"] = y_train.map({"F": 1, "M": 0})
    test_df["target"] = y_test.map({"F": 1, "M": 0})
    train_df.to_csv(train_df_path)
    test_df.to_csv(test_df_path)


@app.command()
def make_model(train_df_path: Path, output_path: Path) -> None:
    """Recieves dataframe and model parameters, then trains model and persists
    it.

    Args:
        train_df_path (Path): Location of train dataframe.
        model_params_path (Path): Location of post hp-tunning parameters file.
        output_path (Path): Path to place final model in.
    """
    # Read train dataset and exctract features and target.
    train_df = pd.read_csv(train_df_path)
    x_train, y_train = train_df.drop("target", axis=1), train_df["target"]
    mask = (x_train.dtypes == "int64").tolist()
    x_train = x_train.loc[:, mask]

    # Instanciate model with best hyperparams found.
    model = LGBMClassifier()
    x_train = x_train.rename(columns=lambda x: re.sub("[^A-Za-z0-9_]+", "", x))
    model = model.fit(x_train, y_train)

    # Persist model...
    with open(output_path, "wb") as fd:
        pickle.dump(model, fd, pickle.HIGHEST_PROTOCOL)


@app.command()
def predict(model_path: Path, test_df_path: Path, predict_path: Path) -> None:
    """Recieves trained model and test dataset and persists predictions.

    Args:
        model_path (Path): Path to where trained model is.
        test_df_path (Path): Location of test dataframe.
        predict_path (Path): Path to store predictions.
    """
    # Read test dataset.
    x_test = pd.read_csv(test_df_path).drop("target", axis=1)
    mask = (x_test.dtypes == "int64").tolist()
    x_test = x_test.loc[:, mask]

    # Read trained model and predict
    with open(model_path, "rb") as fd:
        model = pickle.load(fd)
    preds = model.predict_proba(x_test)

    # Persist predictions...
    with open(predict_path, "wb") as fd:
        pickle.dump(preds, fd, pickle.HIGHEST_PROTOCOL)


@app.command()
def evaluate(
    predict_file: Path, test_df_path: Path, scores_file: Path, plots_path: Path
) -> None:
    """Compute and persist model metrics.

    Args:
        predict_file (Path): File where predicts are stored.
        test_df_path (Path): Location of test dataframe.
        thresh (float): Classification threshold to use in experiment.
    """

    # Load params, predicted and true test values.
    params = PARAMS["evaluate"]
    with open(predict_file, "rb") as fd:
        predicts = pickle.load(fd)
    y_test = pd.read_csv(test_df_path).loc[:, "target"]
    y_test_dummies = pd.get_dummies(y_test)

    # Set threshhold and manipulate prediction array to leave it in the right
    # format.
    binary_preds = (predicts > params["thresh"]).astype(int)
    final_preds = (
        binary_preds[:, 0] * 0 + binary_preds[:, 1] * 1 + binary_preds[:, 2] * 2
    )

    # Compute metrics and persist...
    confsn_matrix = confusion_matrix(y_test, final_preds)
    roc_auc = roc_auc_score(y_test_dummies, binary_preds)
    num_cats = y_test_dummies.columns.shape[0]
    persist_scores(scores_file, confsn_matrix, roc_auc, num_cats)

    # Create and persist plots...
    persist_pr_curves(y_test_dummies, predicts, plots_path)


def persist_scores(
    scores_file: Path, confsn_matrix: np.array, roc_auc: np.array, num_cats: int
):
    """Persist model scores in json format"""
    with open(scores_file, "w") as fd:
        true_positives_dict = {
            f"true_pos_{i}": str(confsn_matrix[i][i]) for i in range(num_cats)
        }
        true_positives_dict["roc_auc"] = roc_auc
        json.dump(true_positives_dict, fd, indent=4)


def persist_pr_curves(
    y_test_dummies: pd.DataFrame, predicts: np.array, plots_path: Path
):
    """Make and persist precision recall curve for each category in json."""
    for category in y_test_dummies.columns:
        file_name = f"pr_curve_{category}.json"
        with open(plots_path / file_name, "w") as fd:
            precision, recall, prc_thresholds = precision_recall_curve(
                y_test_dummies.loc[:, category], predicts[:, category]
            )
            json.dump(
                {
                    "prc": [
                        {"precision": p, "recall": r, "threshold": t}
                        for p, r, t in zip(precision, recall, prc_thresholds)
                    ]
                },
                fd,
                indent=4,
            )


if __name__ == "__main__":
    app()
