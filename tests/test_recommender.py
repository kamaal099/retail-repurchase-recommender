import pandas as pd
import pytest

from retail_ml.exceptions import NotFittedError
from retail_ml.recommender import (ItemItemRecommender, PopularityRecommender,
                                   evaluate_recommender)


def _history():
    rows = [(1, "A"), (1, "B"), (2, "A"), (2, "B"), (2, "C"), (3, "A"), (4, "D")]
    return pd.DataFrame(rows, columns=["customer_id", "stock_code"])


def test_item_item_recommends_co_purchased_unseen_item_first():
    model = ItemItemRecommender().fit(_history())
    recs = model.recommend([3], k=2)[3]
    assert "A" not in recs               # already bought
    assert recs[0] == "B"                # most similar to A


def test_popularity_excludes_seen_items():
    model = PopularityRecommender().fit(_history())
    assert "A" not in model.recommend([3], k=2)[3]


def test_unknown_customer_is_skipped_and_unfitted_raises():
    assert ItemItemRecommender().fit(_history()).recommend([999], k=2) == {}
    with pytest.raises(NotFittedError):
        ItemItemRecommender().recommend([1])


def test_evaluate_hit_rate():
    model = ItemItemRecommender().fit(_history())
    future = pd.DataFrame([(3, "B")], columns=["customer_id", "stock_code"])
    metrics = evaluate_recommender(model, _history(), future, k=2)
    assert metrics["hit_rate@2"] == 1.0
