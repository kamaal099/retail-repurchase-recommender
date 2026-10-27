"""FastAPI service: repurchase probability, predicted spend and product recommendations."""
import hmac
import os
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request

from . import __version__
from .artifacts import load_artifacts
from .features import FEATURE_COLUMNS


def create_app(artifact_dir=None) -> FastAPI:
    directory = Path(artifact_dir or os.getenv("ARTIFACT_DIR", "artifacts"))

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.bundle = load_artifacts(directory)
        yield

    app = FastAPI(title="Retail Repurchase & Recommendation API",
                  version=__version__, lifespan=lifespan)

    def require_api_key(x_api_key: str | None = Header(default=None)):
        expected = os.getenv("API_KEY")
        # Local development remains frictionless when API_KEY is unset.
        if expected and (not x_api_key or not hmac.compare_digest(x_api_key, expected)):
            raise HTTPException(status_code=401, detail="Missing or invalid API key")

    def customer_row(request: Request, customer_id: int):
        features = request.app.state.bundle.features
        if customer_id not in features.index:
            raise HTTPException(status_code=404, detail=f"Unknown customer {customer_id}")
        return features.loc[[customer_id], FEATURE_COLUMNS]

    @app.get("/health")
    def health(request: Request):
        bundle = request.app.state.bundle
        return {"status": "ok", "customers": int(len(bundle.features)),
                "products": int(len(bundle.recommender.items))}

    @app.get("/customers/{customer_id}/repurchase", dependencies=[Depends(require_api_key)])
    def repurchase(customer_id: int, request: Request):
        bundle = request.app.state.bundle
        row = customer_row(request, customer_id)
        probability = float(bundle.classifier.predict_proba(row)[0, 1])
        spend = float(max(np.expm1(bundle.regressor.predict(row)[0]), 0.0))
        return {"customer_id": customer_id,
                "repurchase_probability_90d": round(probability, 4),
                "predicted_spend_90d_gbp": round(spend, 2)}

    @app.get("/customers/{customer_id}/recommendations", dependencies=[Depends(require_api_key)])
    def recommendations(customer_id: int, request: Request,
                        k: int = Query(10, ge=1, le=50)):
        bundle = request.app.state.bundle
        customer_row(request, customer_id)
        recs = bundle.recommender.recommend([customer_id], k).get(customer_id)
        if recs is None:
            raise HTTPException(status_code=404, detail="No purchase history for this customer")
        return {"customer_id": customer_id,
                "recommendations": [{"stock_code": code,
                                     "description": bundle.catalog.get(code, "")}
                                    for code in recs]}

    return app


app = create_app()