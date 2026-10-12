"""Save and load everything the API needs: trained models, features and purchase history."""
from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd

from .exceptions import DataNotFoundError
from .recommender import ItemItemRecommender

FILES = ["classifier.joblib", "regressor.joblib", "customer_features.csv",
         "purchases.csv", "catalog.csv"]


@dataclass
class ArtifactBundle:
    classifier: object
    regressor: object
    features: pd.DataFrame
    recommender: ItemItemRecommender
    catalog: dict


def save_artifacts(directory, classifier, regressor, features, history, sales) -> None:
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    joblib.dump(classifier, d / "classifier.joblib")
    joblib.dump(regressor, d / "regressor.joblib")
    features.to_csv(d / "customer_features.csv")
    history[["customer_id", "stock_code"]].drop_duplicates().to_csv(d / "purchases.csv", index=False)
    sales.drop_duplicates("stock_code")[["stock_code", "description"]].to_csv(
        d / "catalog.csv", index=False)


def load_artifacts(directory) -> ArtifactBundle:
    d = Path(directory)
    missing = [f for f in FILES if not (d / f).exists()]
    if missing:
        raise DataNotFoundError(
            f"Missing {missing} in '{d}'. Run the pipeline first: "
            "python -m retail_ml.pipeline --data-dir data --output-dir reports")
    features = pd.read_csv(d / "customer_features.csv", index_col="customer_id")
    purchases = pd.read_csv(d / "purchases.csv", dtype={"stock_code": str})
    catalog_df = pd.read_csv(d / "catalog.csv", dtype=str).fillna("")
    return ArtifactBundle(
        classifier=joblib.load(d / "classifier.joblib"),
        regressor=joblib.load(d / "regressor.joblib"),
        features=features,
        recommender=ItemItemRecommender().fit(purchases),
        catalog=dict(zip(catalog_df["stock_code"], catalog_df["description"])),
    )