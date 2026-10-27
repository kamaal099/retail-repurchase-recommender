"""Retail Intelligence Console: Streamlit front end for the Retail ML FastAPI service."""
from __future__ import annotations

import html
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

ROOT = Path(__file__).resolve().parent
METRICS_PATH = ROOT / "reports" / "metrics.json"

st.set_page_config(
    page_title="Retail Intelligence Console",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

COLORS = {
    "ink": "#E8EEF8",
    "muted": "#92A0B8",
    "cyan": "#62E6D5",
    "blue": "#8BA7FF",
    "violet": "#B7A0FF",
    "amber": "#F3BE73",
    "panel": "#111A2A",
    "line": "rgba(150, 175, 215, 0.14)",
}
PLOT_LAYOUT = {
    "template": "plotly_dark",
    "paper_bgcolor": "rgba(0,0,0,0)",
    "plot_bgcolor": "rgba(0,0,0,0)",
    "font": {"family": "Inter, ui-sans-serif, system-ui, sans-serif", "color": COLORS["ink"]},
    "margin": {"l": 12, "r": 12, "t": 38, "b": 8},
    "hoverlabel": {"bgcolor": "#172338", "bordercolor": "#26344A", "font": {"color": "#E8EEF8"}},
}

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Manrope:wght@400;500;600;700;800&display=swap');
    :root {
      --ri-bg: #090F1B; --ri-panel: #101929; --ri-panel-2: #131F31;
      --ri-line: rgba(151,174,214,.15); --ri-text: #E8EEF8; --ri-muted: #91A0B8;
      --ri-cyan: #62E6D5; --ri-blue: #8BA7FF; --ri-violet: #B7A0FF;
    }
    html, body, [class*="css"] { font-family: 'Manrope', ui-sans-serif, system-ui, sans-serif; }
    [data-testid="stAppViewContainer"] {
      background:
        radial-gradient(ellipse at 72% -8%, rgba(71, 111, 174, .18), transparent 36%),
        radial-gradient(ellipse at 5% 35%, rgba(39, 128, 130, .07), transparent 28%),
        var(--ri-bg);
      color: var(--ri-text);
    }
    [data-testid="stHeader"] { background: rgba(9,15,27,.72); }
    [data-testid="stSidebar"] {
      background: linear-gradient(180deg, #0D1524 0%, #0A111E 100%);
      border-right: 1px solid var(--ri-line);
    }
    [data-testid="stSidebar"] > div { padding-top: 1.1rem; }
    .block-container { max-width: 1500px; padding-top: 1.65rem; padding-bottom: 3rem; }
    h1, h2, h3 { letter-spacing: -.035em; }
    h1 { font-weight: 800 !important; }
    h2 { font-weight: 750 !important; font-size: 1.35rem !important; }
    h3 { font-weight: 700 !important; font-size: 1.02rem !important; }
    p, li { color: #B9C5D8; }
    [data-testid="stMetric"] {
      background: rgba(17,26,42,.86); border: 1px solid var(--ri-line);
      border-radius: 16px; padding: 17px 18px; min-height: 116px;
    }
    [data-testid="stMetricLabel"] p { color: var(--ri-muted) !important; font-size: .78rem !important; }
    [data-testid="stMetricValue"] { color: var(--ri-text) !important; font-weight: 800; }
    [data-testid="stMetricDelta"] { font-size: .76rem; }
    [data-testid="stVerticalBlockBorderWrapper"] {
      border-color: var(--ri-line) !important; background: rgba(16,25,41,.72);
      border-radius: 18px !important;
    }
    [data-testid="stTextInput"] input, [data-testid="stNumberInput"] input,
    [data-testid="stSelectbox"] div[data-baseweb="select"] > div,
    [data-testid="stSlider"] { border-radius: 10px; }
    [data-testid="stButton"] button, [data-testid="stFormSubmitButton"] button {
      border-radius: 10px; border: 1px solid rgba(98,230,213,.35);
      font-weight: 700; min-height: 42px; transition: all .15s ease;
    }
    [data-testid="stButton"] button[kind="primary"],
    [data-testid="stFormSubmitButton"] button[kind="primary"] {
      background: linear-gradient(110deg, #62E6D5, #81CDEB);
      color: #07131D; border: 0;
    }
    [data-testid="stButton"] button:hover { border-color: var(--ri-cyan); transform: translateY(-1px); }
    [data-testid="stPlotlyChart"] { border: 1px solid var(--ri-line); border-radius: 15px; overflow: hidden; }
    [data-testid="stDataFrame"] { border: 1px solid var(--ri-line); border-radius: 12px; overflow: hidden; }
    .brand-row { display:flex; align-items:center; gap:11px; padding: 4px 2px 22px; }
    .brand-mark {
      width:43px; height:43px; display:grid; place-items:center; border-radius:13px;
      color:#08151E; font-size:21px; font-weight:900;
      background: linear-gradient(145deg, #8BFFEA, #6EA9FF);
      box-shadow: 0 7px 24px rgba(98,230,213,.12);
    }
    .brand-title { color:var(--ri-text); font-weight:800; font-size:1rem; letter-spacing:-.04em; line-height:1.2; }
    .brand-sub { color:var(--ri-muted); font-size:.67rem; letter-spacing:.13em; margin-top:4px; }
    .eyebrow {
      color: var(--ri-cyan); font-family: 'DM Mono', monospace; letter-spacing: .16em;
      text-transform: uppercase; font-size: .67rem; font-weight: 500;
    }
    .hero { padding: 4px 0 18px; }
    .hero h1 { margin:.45rem 0 .5rem; font-size: clamp(1.9rem,3vw,2.7rem); line-height:1.08; }
    .hero p { margin:0; color:var(--ri-muted); font-size:.94rem; max-width:820px; line-height:1.65; }
    .status-pill { display:inline-flex; align-items:center; gap:7px; border:1px solid var(--ri-line);
      border-radius:100px; padding:6px 10px; color:#C6D4E7; font-size:.73rem; background:rgba(17,26,42,.65); }
    .status-dot { height:7px; width:7px; border-radius:100%; display:inline-block; background:var(--ri-cyan); }
    .section-heading { display:flex; align-items:center; justify-content:space-between; gap:12px; margin: 10px 0 12px; }
    .section-heading h2 { margin:0; }
    .section-note { color:var(--ri-muted); font-size:.72rem; font-family:'DM Mono',monospace; }
    .panel-title { color:var(--ri-text); font-weight:750; letter-spacing:-.025em; font-size:1.02rem; margin:0 0 4px; }
    .panel-subtitle { color:var(--ri-muted); font-size:.78rem; margin:0 0 14px; line-height:1.5; }
    .product-card {
      border:1px solid var(--ri-line); background:linear-gradient(150deg, rgba(20,32,51,.92), rgba(12,20,34,.92));
      border-radius:14px; padding:14px; min-height:107px; margin-bottom:10px;
    }
    .product-number { color:var(--ri-cyan); font: .67rem 'DM Mono',monospace; letter-spacing:.12em; }
    .product-code { color:#F0F4FC; font-weight:800; font-size:.9rem; margin:7px 0 3px; }
    .product-description { color:var(--ri-muted); font-size:.76rem; line-height:1.45; overflow-wrap:anywhere; }
    .signal {
      border-left:3px solid var(--ri-cyan); background:rgba(98,230,213,.055);
      padding:12px 14px; border-radius:0 10px 10px 0; color:#C8D5E5; font-size:.82rem; line-height:1.55;
    }
    .footer { border-top:1px solid var(--ri-line); margin-top:30px; padding-top:15px; color:#71819A;
      display:flex; justify-content:space-between; gap:12px; font: .68rem 'DM Mono',monospace; }
    @media(max-width:800px) {
      .block-container { padding:1rem .8rem 2rem; }
      .hero h1 { font-size:1.85rem; }
      .footer { display:block; line-height:1.9; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_metrics() -> dict[str, Any]:
    if not METRICS_PATH.exists():
        return {}
    try:
        return json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


@st.cache_data(ttl=15, show_spinner=False)
def check_api_health(base_url: str) -> tuple[bool, dict[str, Any] | str]:
    try:
        response = requests.get(f"{base_url.rstrip('/')}/health", timeout=2.5)
        if response.ok:
            payload = response.json()
            return payload.get("status") == "ok", payload
        return False, f"API returned HTTP {response.status_code}"
    except requests.RequestException as exc:
        return False, str(exc)
    except ValueError:
        return False, "API health endpoint returned invalid JSON."


def request_payload(base_url: str, path: str) -> tuple[dict[str, Any] | None, str | None]:
    try:
        response = requests.get(f"{base_url.rstrip('/')}{path}", timeout=8)
    except requests.RequestException as exc:
        return None, f"Cannot reach the API at {base_url}. Check that FastAPI is running and the URL is correct. Details: {exc}"
    try:
        payload = response.json()
    except ValueError:
        return None, f"API returned non-JSON content (HTTP {response.status_code})."
    if not response.ok:
        detail = payload.get("detail", payload)
        return None, f"API returned HTTP {response.status_code}: {detail}"
    return payload, None


def metric_card(label: str, value: str, detail: str, accent: str = "cyan") -> None:
    accent_color = {"cyan": COLORS["cyan"], "blue": COLORS["blue"], "violet": COLORS["violet"], "amber": COLORS["amber"]}.get(accent, COLORS["cyan"])
    st.markdown(
        f"""
        <div style="border:1px solid {COLORS['line']};border-top:2px solid {accent_color};
                    border-radius:15px;background:linear-gradient(145deg,rgba(18,29,47,.96),rgba(11,19,32,.94));
                    padding:17px 18px 16px;min-height:112px;">
          <div style="color:{COLORS['muted']};font-size:.75rem;letter-spacing:.025em;">{html.escape(label)}</div>
          <div style="color:{COLORS['ink']};font-size:1.7rem;font-weight:800;letter-spacing:-.055em;margin-top:7px;">{html.escape(value)}</div>
          <div style="color:{COLORS['muted']};font-size:.72rem;margin-top:5px;line-height:1.4;">{html.escape(detail)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_heading(title: str, note: str | None = None) -> None:
    right = f'<span class="section-note">{html.escape(note)}</span>' if note else ""
    st.markdown(
        f'<div class="section-heading"><h2>{html.escape(title)}</h2>{right}</div>',
        unsafe_allow_html=True,
    )


def page_header(kicker: str, title: str, description: str, right: str | None = None) -> None:
    right_html = f'<span class="status-pill"><span class="status-dot"></span>{html.escape(right)}</span>' if right else ""
    st.markdown(
        f"""
        <div class="hero">
          <div class="eyebrow">{html.escape(kicker)}</div>
          <div style="display:flex;justify-content:space-between;align-items:flex-start;gap:18px;flex-wrap:wrap;">
            <div style="flex:1;min-width:250px;">
              <h1>{html.escape(title)}</h1>
              <p>{html.escape(description)}</p>
            </div>
            <div style="padding-top:10px">{right_html}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def plot_layout(fig: go.Figure, height: int = 320) -> go.Figure:
    fig.update_layout(**PLOT_LAYOUT, height=height, legend={"orientation": "h", "y": 1.13, "x": 0})
    fig.update_xaxes(showgrid=False, linecolor=COLORS["line"], tickfont={"color": COLORS["muted"]})
    fig.update_yaxes(gridcolor="rgba(145,160,184,.12)", zeroline=False, tickfont={"color": COLORS["muted"]})
    return fig


def render_overview(metrics: dict[str, Any]) -> None:
    page_header(
        "RETAIL INTELLIGENCE / COMMAND CENTER",
        "Know the customer. Anticipate the next move.",
        "A unified view of retention propensity, customer value, and product discovery — powered by your trained models and historical evaluation.",
        "MODEL PIPELINE READY",
    )
    if not metrics:
        st.error("Could not load reports/metrics.json. Run the training pipeline first.")
        return

    data = metrics.get("data", {})
    classification = metrics.get("classification", {})
    regression = metrics.get("regression", {})
    recommendations = metrics.get("recommendation", {})
    cls_test = classification.get("final_test_metrics", {})
    reg_test = regression.get("final_test_metrics", {})
    cf = recommendations.get("item_item_cf", {})
    popularity = recommendations.get("popularity_baseline", {})

    cols = st.columns(4, gap="medium")
    with cols[0]:
        metric_card("CLEANED SALES LINES", f"{data.get('sales_rows', 0):,}", "Transactions retained after cleaning", "cyan")
    with cols[1]:
        metric_card("CUSTOMERS MODELLED", f"{data.get('customers_modelled', 0):,}", "Customer-level modelling universe", "blue")
    with cols[2]:
        metric_card("REPURCHASE RATE", f"{data.get('repurchase_rate', 0):.1%}", "Observed future-window label rate", "violet")
    with cols[3]:
        metric_card("CF HIT RATE @10", f"{cf.get('hit_rate@10', 0):.1%}", "Offline future-item hit rate", "amber")

    st.write("")
    left, right = st.columns([1.2, 1], gap="large")
    with left:
        section_heading("Model signal", "FINAL TEST · HOLDOUT")
        candidates = classification.get("validation_metrics_by_model", {})
        if candidates:
            frame = pd.DataFrame(
                [{"Model": name.replace("_", " ").title(),
                  "ROC-AUC": value.get("roc_auc", 0),
                  "PR-AUC": value.get("pr_auc", 0),
                  "Selected": name == classification.get("best_model")} for name, value in candidates.items()]
            )
            chart = go.Figure()
            chart.add_bar(name="ROC-AUC", x=frame["Model"], y=frame["ROC-AUC"],
                          marker_color=[COLORS["cyan"] if v else "#385069" for v in frame["Selected"]])
            chart.add_bar(name="PR-AUC", x=frame["Model"], y=frame["PR-AUC"],
                          marker_color=[COLORS["blue"] if v else "#38465E" for v in frame["Selected"]])
            chart.update_layout(barmode="group", yaxis={"range": [0, 1], "title": None},
                                xaxis={"title": None}, title=None)
            st.plotly_chart(plot_layout(chart, 315), use_container_width=True, config={"displayModeBar": False})
            st.caption(f"Selected on validation ROC-AUC: {classification.get('best_model', 'not reported').replace('_', ' ').title()}. Final test ROC-AUC: {cls_test.get('roc_auc', float('nan')):.3f}.")
        else:
            st.info("Validation metrics are not present in this report.")
    with right:
        section_heading("Recommendation lift", "OFFLINE BENCHMARK")
        labels = ["Hit rate @10", "Recall @10", "Catalog coverage @10"]
        keys = ["hit_rate@10", "recall@10", "coverage@10"]
        fig = go.Figure()
        fig.add_bar(name="Popularity", x=labels, y=[popularity.get(k, 0) for k in keys],
                    marker_color="#52617A", marker_line_width=0)
        fig.add_bar(name="Item-item CF", x=labels, y=[cf.get(k, 0) for k in keys],
                    marker_color=COLORS["cyan"], marker_line_width=0)
        fig.update_layout(barmode="group", yaxis={"tickformat": ".0%", "title": None},
                          xaxis={"title": None}, title=None)
        st.plotly_chart(plot_layout(fig, 315), use_container_width=True, config={"displayModeBar": False})
        st.markdown(
            f'<div class="signal">Item-item CF improves hit rate by <b>{(cf.get("hit_rate@10", 0) - popularity.get("hit_rate@10", 0)):.1%}</b> over popularity on this historical holdout. This is offline ranking performance, not a causal revenue estimate.</div>',
            unsafe_allow_html=True,
        )

    st.write("")
    section_heading("Commercial modelling snapshot", "VALIDATION-SELECTED MODELS")
    c1, c2, c3 = st.columns(3, gap="medium")
    with c1:
        metric_card("REPURCHASE CLASSIFIER", classification.get("best_model", "—").replace("_", " ").title(),
                    f"ROC-AUC {cls_test.get('roc_auc', 0):.3f} · PR-AUC {cls_test.get('pr_auc', 0):.3f}", "cyan")
    with c2:
        metric_card("SPEND REGRESSOR", regression.get("best_model", "—").replace("_", " ").title(),
                    f"RMSE {reg_test.get('rmse', 0):.3f} · R² {reg_test.get('r2', 0):.3f} on log-spend target", "blue")
    with c3:
        metric_card("EVALUATED CUSTOMERS", f"{cf.get('n_customers_evaluated', 0):,}",
                    f"Catalog size {cf.get('catalog_items', 0):,} items · top 10 recommendations", "violet")
    st.caption(
        f"Temporal evaluation window: {data.get('cutoff_date', '—')} to {data.get('last_future_date', '—')}. "
        "Metrics are offline historical results; model performance may vary across time."
    )


def render_customer_workspace(api_url: str) -> None:
    page_header(
        "CUSTOMER INTELLIGENCE / 360",
        "One customer. Three decisions.",
        "Look up a known customer to retrieve their modelled repurchase propensity, projected spend, and personalised product suggestions.",
        "LIVE API WORKSPACE",
    )
    health_ok, health = check_api_health(api_url)
    if health_ok:
        st.success(f"Prediction API online · {health.get('customers', '—'):,} customers · {health.get('products', '—'):,} catalogue items")
    else:
        st.warning(f"Prediction API is unavailable at {api_url}. Start FastAPI in another terminal or update the API URL in the sidebar. {health}")

    with st.form("customer_analysis_form", border=True):
        col1, col2, col3 = st.columns([1.15, 1, 1.1], gap="medium")
        with col1:
            customer_id = st.number_input("Customer ID", min_value=1, value=12346, step=1, help="Enter a customer ID present in the trained customer features.")
        with col2:
            top_k = st.slider("Products to recommend", min_value=1, max_value=50, value=10)
        with col3:
            st.markdown('<div style="padding-top:27px"></div>', unsafe_allow_html=True)
            submitted = st.form_submit_button("Run customer analysis", type="primary", use_container_width=True)

    if not submitted:
        st.markdown(
            '<div class="signal">Choose a customer ID and recommendation depth, then run the analysis. The dashboard calls the FastAPI model service; it does not fabricate predictions when the service is offline.</div>',
            unsafe_allow_html=True,
        )
        return
    if not health_ok:
        st.error("Cannot analyse a customer until the API reports healthy.")
        return

    with st.spinner("Retrieving model outputs and ranking products..."):
        prediction, prediction_error = request_payload(api_url, f"/customers/{int(customer_id)}/repurchase")
        product_result, product_error = request_payload(api_url, f"/customers/{int(customer_id)}/recommendations?k={int(top_k)}")

    if prediction_error:
        st.error(prediction_error)
    if product_error:
        st.error(product_error)
    if not prediction or not product_result:
        return

    probability = float(prediction.get("repurchase_probability_90d", 0))
    spend = float(prediction.get("predicted_spend_90d_gbp", 0))
    section_heading("Customer signal", f"CUSTOMER {int(customer_id)}")
    col1, col2 = st.columns([1, 1], gap="large")
    with col1:
        gauge = go.Figure(go.Indicator(
            mode="gauge+number", value=probability * 100,
            number={"suffix": "%", "font": {"size": 35, "color": COLORS["ink"]}, "valueformat": ".1f"},
            title={"text": "Modelled repurchase propensity", "font": {"size": 14, "color": COLORS["muted"]}},
            gauge={
                "axis": {"range": [0, 100], "ticksuffix": "%", "tickcolor": COLORS["muted"],
                         "tickfont": {"color": COLORS["muted"], "size": 10}},
                "bar": {"color": COLORS["cyan"], "thickness": .22},
                "bgcolor": "rgba(0,0,0,0)",
                "borderwidth": 0,
                "steps": [{"range": [0, 33], "color": "rgba(255,141,154,.10)"},
                          {"range": [33, 66], "color": "rgba(243,190,115,.10)"},
                          {"range": [66, 100], "color": "rgba(98,230,213,.10)"}],
                "threshold": {"line": {"color": COLORS["blue"], "width": 3},
                              "thickness": .8, "value": probability * 100},
            },
        ))
        gauge.update_layout(**{**PLOT_LAYOUT, "height": 250, "margin": {"l": 20, "r": 20, "t": 50, "b": 8}})
        st.plotly_chart(gauge, use_container_width=True, config={"displayModeBar": False})
        band = "Higher modelled propensity" if probability >= .66 else "Middle modelled propensity" if probability >= .33 else "Lower modelled propensity"
        st.caption(f"{band} · heuristic band only; probability calibration and campaign thresholds should be validated before operational use.")
    with col2:
        st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)
        metric_card("PREDICTED SPEND · NEXT 90 DAYS", f"£{spend:,.2f}",
                    "Model output transformed back from log-spend units", "blue")
        st.write("")
        metric_card("REPURCHASE PROBABILITY", f"{probability:.1%}",
                    "Model score for the defined future window; not a guarantee of purchase", "cyan")
        st.write("")
        st.markdown(
            '<div class="signal">Use this score alongside customer context and campaign economics. It is not an individual causal treatment effect and should not be used alone to decide eligibility for discounts.</div>',
            unsafe_allow_html=True,
        )

    section_heading("Recommended products", f"TOP {int(top_k)} · ITEM-ITEM COLLABORATIVE FILTERING")
    products = product_result.get("recommendations", [])
    if not products:
        st.info("The API returned no recommendations for this customer.")
    else:
        for start in range(0, len(products), 3):
            row = st.columns(3, gap="medium")
            for slot, product in enumerate(products[start:start + 3]):
                code = html.escape(str(product.get("stock_code", "—")))
                description = html.escape(str(product.get("description") or "Description unavailable"))
                rank = start + slot + 1
                with row[slot]:
                    st.markdown(
                        f'<div class="product-card"><div class="product-number">RECOMMENDATION {rank:02d}</div>'
                        f'<div class="product-code">{code}</div><div class="product-description">{description}</div></div>',
                        unsafe_allow_html=True,
                    )
        st.caption("Ranked from historical purchase co-occurrence. Recommendations do not currently account for stock, margin, price, or product metadata beyond description.")


def render_model_lab(metrics: dict[str, Any]) -> None:
    page_header(
        "MODEL INTELLIGENCE / EVALUATION",
        "Understand the model before trusting the score.",
        "Compare candidate models on validation data and inspect the final untouched test results. The test partition is not used to select the winning candidate.",
        "VALIDATION-BASED SELECTION",
    )
    classification = metrics.get("classification", {})
    regression = metrics.get("regression", {})
    candidates = classification.get("validation_metrics_by_model", {})
    selected = classification.get("best_model")
    test_metrics = classification.get("final_test_metrics", {})
    reg_test = regression.get("final_test_metrics", {})
    if not candidates:
        st.error("Model validation metrics are missing from reports/metrics.json.")
        return

    col1, col2, col3 = st.columns(3, gap="medium")
    with col1:
        metric_card("SELECTED CLASSIFIER", str(selected or "—").replace("_", " ").title(),
                    f"Final test ROC-AUC {test_metrics.get('roc_auc', 0):.3f}", "cyan")
    with col2:
        metric_card("SELECTED REGRESSOR", str(regression.get("best_model", "—")).replace("_", " ").title(),
                    f"Final test RMSE {reg_test.get('rmse', 0):.3f} (log-spend target)", "blue")
    with col3:
        protocol = metrics.get("evaluation_protocol", {})
        metric_card("HOLDOUT DESIGN", "60 / 20 / 20",
                    f"Train / validation / test · horizon {protocol.get('temporal_horizon_days', '—')} days", "violet")

    section_heading("Classification candidates", "SORTED BY VALIDATION ROC-AUC")
    frame = pd.DataFrame([
        {"Model": name.replace("_", " ").title(),
         "Validation ROC-AUC": val.get("roc_auc", 0),
         "Validation PR-AUC": val.get("pr_auc", 0),
         "Validation F1": val.get("f1", 0),
         "Selected": name == selected}
        for name, val in candidates.items()
    ]).sort_values("Validation ROC-AUC", ascending=False)
    st.dataframe(
        frame.style.format({"Validation ROC-AUC": "{:.4f}", "Validation PR-AUC": "{:.4f}", "Validation F1": "{:.4f}"}),
        use_container_width=True, hide_index=True,
    )
    fig = go.Figure()
    fig.add_bar(name="ROC-AUC", x=frame["Model"], y=frame["Validation ROC-AUC"],
                marker_color=[COLORS["cyan"] if v else "#3B5069" for v in frame["Selected"]])
    fig.add_bar(name="PR-AUC", x=frame["Model"], y=frame["Validation PR-AUC"],
                marker_color=[COLORS["blue"] if v else "#39465E" for v in frame["Selected"]])
    fig.update_layout(barmode="group", yaxis={"range": [0, 1], "title": None}, xaxis={"title": None})
    st.plotly_chart(plot_layout(fig, 330), use_container_width=True, config={"displayModeBar": False})

    st.write("")
    section_heading("Final holdout results", "UNTOUCHED TEST PARTITION")
    a, b = st.columns(2, gap="large")
    with a:
        st.markdown('<p class="panel-title">Classification</p><p class="panel-subtitle">Scored after candidate selection and refit.</p>', unsafe_allow_html=True)
        cls_rows = [{"Metric": k.replace("_", " ").upper(), "Final test": v} for k, v in test_metrics.items()]
        st.dataframe(pd.DataFrame(cls_rows).style.format({"Final test": "{:.4f}"}), use_container_width=True, hide_index=True)
    with b:
        st.markdown('<p class="panel-title">Spend regression</p><p class="panel-subtitle">Target is log(1 + spend), not raw GBP.</p>', unsafe_allow_html=True)
        reg_rows = [{"Metric": k.upper(), "Final test": v} for k, v in reg_test.items()]
        st.dataframe(pd.DataFrame(reg_rows).style.format({"Final test": "{:.4f}"}), use_container_width=True, hide_index=True)

    st.markdown(
        '<div class="signal">Interpretation note: AUC and regression results are historical holdout estimates, not production guarantees. Monitor drift and validate on later periods before operational deployment.</div>',
        unsafe_allow_html=True,
    )


metrics = load_metrics()

with st.sidebar:
    st.markdown(
        """
        <div class="brand-row">
          <div class="brand-mark">R◈</div>
          <div>
            <div class="brand-title">Retail Intelligence</div>
            <div class="brand-sub">CUSTOMER & COMMERCE LAB</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<div class="eyebrow">WORKSPACE</div>', unsafe_allow_html=True)
    page = st.radio(
        "Navigation",
        ["Command Center", "Customer 360", "Model Lab"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown('<div class="eyebrow">INFERENCE SERVICE</div>', unsafe_allow_html=True)
    default_api_url = os.getenv("RETAIL_API_URL", "http://127.0.0.1:8000")
    api_url = st.text_input("FastAPI base URL", value=default_api_url, help="Set RETAIL_API_URL when the API runs on a separate host.")
    api_url = api_url.strip().rstrip("/")
    if api_url:
        api_ok, api_state = check_api_health(api_url)
    else:
        api_ok, api_state = False, "Enter a valid API URL."
    st.markdown(
        f'<div class="status-pill"><span class="status-dot" style="background:{"#62E6D5" if api_ok else "#FF8D9A"}"></span>{"API CONNECTED" if api_ok else "API DISCONNECTED"}</div>',
        unsafe_allow_html=True,
    )
    if api_ok and isinstance(api_state, dict):
        st.caption(f"{api_state.get('customers', 0):,} customer profiles · {api_state.get('products', 0):,} products")
    else:
        st.caption("Overview and model diagnostics remain available; customer inference requires the API.")
    st.markdown("---")
    st.caption("OFFLINE MODEL EVALUATION")
    st.caption(f"Window: {metrics.get('data', {}).get('cutoff_date', '—')} → {metrics.get('data', {}).get('last_future_date', '—')}")


if page == "Command Center":
    render_overview(metrics)
elif page == "Customer 360":
    render_customer_workspace(api_url)
else:
    render_model_lab(metrics)

st.markdown(
    """
    <div class="footer">
      <span>RETAIL INTELLIGENCE CONSOLE · OFFLINE MODEL METRICS</span>
      <span>FASTAPI + STREAMLIT · VALIDATION-BASED MODEL SELECTION</span>
    </div>
    """,
    unsafe_allow_html=True,
)
