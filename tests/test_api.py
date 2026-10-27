import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression, Ridge

from retail_ml.api import create_app
from retail_ml.artifacts import save_artifacts
from retail_ml.exceptions import DataNotFoundError
from retail_ml.features import FEATURE_COLUMNS


@pytest.fixture()
def client(tmp_path):
    rng = np.random.default_rng(0)
    index = pd.Index(range(1000, 1060), name="customer_id")
    X = pd.DataFrame(rng.uniform(1, 50, (60, len(FEATURE_COLUMNS))),
                     columns=FEATURE_COLUMNS, index=index)
    clf = LogisticRegression(max_iter=500).fit(X, (X["frequency"] > 25).astype(int))
    reg = Ridge().fit(X, np.log1p(X["monetary"]))
    history = pd.DataFrame({"customer_id": [1000, 1000, 1001, 1001, 1001, 1002],
                            "stock_code": ["11111", "22222", "11111", "22222", "33333", "11111"]})
    sales = pd.DataFrame({"stock_code": ["11111", "22222", "33333"],
                          "description": ["MUG", "LAMP", "CUP"]})
    save_artifacts(tmp_path, clf, reg, X, history, sales)
    with TestClient(create_app(tmp_path)) as c:
        yield c


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["customers"] == 60 and body["products"] == 3


def test_repurchase_prediction(client):
    body = client.get("/customers/1000/repurchase").json()
    assert 0.0 <= body["repurchase_probability_90d"] <= 1.0
    assert body["predicted_spend_90d_gbp"] >= 0.0


def test_recommendations_exclude_seen_items(client):
    recs = client.get("/customers/1000/recommendations?k=5").json()["recommendations"]
    codes = [r["stock_code"] for r in recs]
    assert codes == ["33333"] and recs[0]["description"] == "CUP"


def test_unknown_customer_and_bad_k(client):
    assert client.get("/customers/9/repurchase").status_code == 404
    assert client.get("/customers/1010/recommendations").status_code == 404  # no history
    assert client.get("/customers/1000/recommendations?k=0").status_code == 422


def test_inference_routes_require_api_key_when_configured(client, monkeypatch):
    monkeypatch.setenv("API_KEY", "demo-secret")

    assert client.get("/health").status_code == 200  # health check stays public
    assert client.get("/customers/1000/repurchase").status_code == 401
    assert client.get("/customers/1000/recommendations?k=3").status_code == 401

    headers = {"X-API-Key": "demo-secret"}
    assert client.get("/customers/1000/repurchase", headers=headers).status_code == 200
    assert client.get("/customers/1000/recommendations?k=3", headers=headers).status_code == 200


def test_missing_artifacts_fail_with_clear_error(tmp_path):
    with pytest.raises(DataNotFoundError):
        with TestClient(create_app(tmp_path / "empty")):
            pass