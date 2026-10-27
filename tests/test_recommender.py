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
    assert "A" not in recs
    assert recs[0] == "B"


def test_popularity_excludes_seen_items():
    model = PopularityRecommender().fit(_history())
    assert "A" not in model.recommend([3], k=2)[3]


def test_unknown_customer_is_skipped_and_unfitted_raises():
    assert ItemItemRecommender().fit(_history()).recommend([999], k=2) == {}
    with pytest.raises(NotFittedError):
        ItemItemRecommender().recommend([1])


@pytest.mark.parametrize("k", [0, -1, 1.5, True])
def test_recommend_rejects_invalid_k(k):
    model = ItemItemRecommender().fit(_history())
    with pytest.raises(ValueError, match="positive integer"):
        model.recommend([3], k=k)


def test_recommend_empty_customer_list():
    model = ItemItemRecommender().fit(_history())
    assert model.recommend([], k=2) == {}


def test_evaluate_hit_rate_and_coverage():
    model = ItemItemRecommender().fit(_history())
    future = pd.DataFrame([(3, "B")], columns=["customer_id", "stock_code"])
    metrics = evaluate_recommender(model, _history(), future, k=2)
    assert metrics["hit_rate@2"] == 1.0
    assert metrics["recall@2"] == 1.0
    assert 0 <= metrics["coverage@2"] <= 1
    assert metrics["catalog_items"] == 4
