import streamlit as st
import numpy as np
import pandas as pd
import joblib
import json
import os
import plotly.graph_objects as go

# ── Load model & metadata ──────────────────────────────────────────────────────
BASE = os.path.dirname(__file__)

@st.cache_resource
def load_model():
    return joblib.load(os.path.join(BASE, "model.joblib"))

@st.cache_data
def load_meta():
    with open(os.path.join(BASE, "model_metadata.json")) as f:
        return json.load(f)

model = load_model()
meta  = load_meta()

COLS        = meta["nutrient_cols"]
DISP        = meta["display_names"]   # "Carbohydrate (g)", etc.
FI          = meta["feature_importances"]
CAT_MEANS   = meta["category_means"]
CATEGORIES  = meta["categories"]
METRICS     = meta["global_metrics"]

# Typical per-100g ranges for slider bounds (generous)
RANGES = {
    "Data.Carbohydrate":               (0.0, 100.0, 20.0),
    "Data.Fat.Total Lipid":            (0.0, 100.0, 10.0),
    "Data.Protein":                    (0.0,  90.0, 10.0),
    "Data.Sugar Total":                (0.0, 100.0,  5.0),
    "Data.Fiber":                      (0.0,  80.0,  2.0),
    "Data.Water":                      (0.0, 100.0, 50.0),
    "Data.Fat.Saturated Fat":          (0.0,  90.0,  3.0),
    "Data.Fat.Monosaturated Fat":      (0.0,  80.0,  3.0),
    "Data.Fat.Polysaturated Fat":      (0.0,  80.0,  2.0),
    "Data.Major Minerals.Sodium":      (0.0,5000.0,100.0),
    "Data.Major Minerals.Calcium":     (0.0,2000.0, 50.0),
    "Data.Major Minerals.Potassium":   (0.0,2000.0,150.0),
    "Data.Major Minerals.Phosphorus":  (0.0,2000.0,100.0),
    "Data.Cholesterol":                (0.0,2000.0, 30.0),
    "Data.Ash":                        (0.0,  30.0,  1.5),
}

FOOD_PRESETS = {
    "— choose a preset —": None,
    "Grilled Chicken Breast": {
        "Data.Carbohydrate": 0.0, "Data.Fat.Total Lipid": 3.6, "Data.Protein": 31.0,
        "Data.Sugar Total": 0.0, "Data.Fiber": 0.0, "Data.Water": 65.0,
        "Data.Fat.Saturated Fat": 1.0, "Data.Fat.Monosaturated Fat": 1.2,
        "Data.Fat.Polysaturated Fat": 0.8, "Data.Major Minerals.Sodium": 74.0,
        "Data.Major Minerals.Calcium": 15.0, "Data.Major Minerals.Potassium": 358.0,
        "Data.Major Minerals.Phosphorus": 220.0, "Data.Cholesterol": 85.0, "Data.Ash": 1.0,
    },
    "White Rice (cooked)": {
        "Data.Carbohydrate": 28.2, "Data.Fat.Total Lipid": 0.3, "Data.Protein": 2.7,
        "Data.Sugar Total": 0.1, "Data.Fiber": 0.4, "Data.Water": 68.4,
        "Data.Fat.Saturated Fat": 0.08, "Data.Fat.Monosaturated Fat": 0.09,
        "Data.Fat.Polysaturated Fat": 0.08, "Data.Major Minerals.Sodium": 1.0,
        "Data.Major Minerals.Calcium": 10.0, "Data.Major Minerals.Potassium": 35.0,
        "Data.Major Minerals.Phosphorus": 43.0, "Data.Cholesterol": 0.0, "Data.Ash": 0.2,
    },
    "Cheddar Cheese": {
        "Data.Carbohydrate": 1.3, "Data.Fat.Total Lipid": 33.1, "Data.Protein": 24.9,
        "Data.Sugar Total": 0.5, "Data.Fiber": 0.0, "Data.Water": 37.0,
        "Data.Fat.Saturated Fat": 21.1, "Data.Fat.Monosaturated Fat": 9.4,
        "Data.Fat.Polysaturated Fat": 0.9, "Data.Major Minerals.Sodium": 621.0,
        "Data.Major Minerals.Calcium": 721.0, "Data.Major Minerals.Potassium": 98.0,
        "Data.Major Minerals.Phosphorus": 512.0, "Data.Cholesterol": 105.0, "Data.Ash": 3.7,
    },
    "Olive Oil": {
        "Data.Carbohydrate": 0.0, "Data.Fat.Total Lipid": 100.0, "Data.Protein": 0.0,
        "Data.Sugar Total": 0.0, "Data.Fiber": 0.0, "Data.Water": 0.0,
        "Data.Fat.Saturated Fat": 13.8, "Data.Fat.Monosaturated Fat": 73.0,
        "Data.Fat.Polysaturated Fat": 10.5, "Data.Major Minerals.Sodium": 2.0,
        "Data.Major Minerals.Calcium": 1.0, "Data.Major Minerals.Potassium": 1.0,
        "Data.Major Minerals.Phosphorus": 0.0, "Data.Cholesterol": 0.0, "Data.Ash": 0.0,
    },
    "Raw Broccoli": {
        "Data.Carbohydrate": 6.6, "Data.Fat.Total Lipid": 0.4, "Data.Protein": 2.8,
        "Data.Sugar Total": 1.7, "Data.Fiber": 2.6, "Data.Water": 89.3,
        "Data.Fat.Saturated Fat": 0.04, "Data.Fat.Monosaturated Fat": 0.03,
        "Data.Fat.Polysaturated Fat": 0.19, "Data.Major Minerals.Sodium": 33.0,
        "Data.Major Minerals.Calcium": 47.0, "Data.Major Minerals.Potassium": 316.0,
        "Data.Major Minerals.Phosphorus": 66.0, "Data.Cholesterol": 0.0, "Data.Ash": 0.9,
    },
}

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Food Calorie Predictor",
    page_icon="🥗",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  .big-number { font-size: 4rem; font-weight: 800; color: #E8654A; line-height: 1.1; }
  .kcal-label { font-size: 1.1rem; color: #888; margin-top: -6px; }
  .metric-box { background: #1E1E2E; border-radius: 12px; padding: 18px 24px; text-align: center; }
  .cat-badge  { display: inline-block; background: #2D4A3E; color: #7EC8A4;
                border-radius: 20px; padding: 4px 14px; font-size: 0.85rem; font-weight: 600; }
  .section-header { color: #aaa; font-size: 0.75rem; text-transform: uppercase;
                    letter-spacing: 1.5px; margin-bottom: 4px; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🥗 Food Calorie Predictor")
    st.caption(
        "Enter the nutrient profile of any food (per 100 g) and get an instant "
        "kilocalorie prediction from a Random Forest model trained on 7,400+ USDA foods."
    )
    st.divider()

    preset = st.selectbox("Load a food preset", list(FOOD_PRESETS.keys()))
    preset_vals = FOOD_PRESETS[preset]

    st.divider()
    st.markdown("**Model stats**")
    col1, col2 = st.columns(2)
    col1.metric("R²", f"{METRICS['r2']:.4f}")
    col2.metric("MAE", f"{METRICS['mae']} kcal")
    st.caption(f"Trained on {meta['n_total']:,} USDA food records · Random Forest (200 trees)")
    st.divider()
    st.caption(
        "Built by Saketh · Research project: *Predicting Food Kilocalories "
        "from Nutrient Profiles Using Machine Learning*"
    )

# ── Main layout ────────────────────────────────────────────────────────────────
st.markdown("# Predict Food Calories from Nutrients")
st.markdown(
    "Enter nutrient values per 100 g of food. Values update the prediction live."
)

input_col, result_col = st.columns([3, 2], gap="large")

with input_col:
    st.markdown("### Nutrient Inputs (per 100 g)")

    values = {}
    macros  = ["Data.Carbohydrate", "Data.Fat.Total Lipid", "Data.Protein",
                "Data.Sugar Total", "Data.Fiber", "Data.Water"]
    fats    = ["Data.Fat.Saturated Fat", "Data.Fat.Monosaturated Fat", "Data.Fat.Polysaturated Fat"]
    minerals= ["Data.Major Minerals.Sodium", "Data.Major Minerals.Calcium",
                "Data.Major Minerals.Potassium", "Data.Major Minerals.Phosphorus",
                "Data.Cholesterol", "Data.Ash"]

    def make_inputs(group):
        for col in group:
            lo, hi, default = RANGES[col]
            dv = float(preset_vals[col]) if preset_vals else default
            label = meta["short_names"][col]
            values[col] = st.slider(label, min_value=lo, max_value=hi, value=dv, step=0.1, key=col)

    with st.expander("Macronutrients", expanded=True):
        make_inputs(macros)

    with st.expander("Fat Breakdown", expanded=False):
        make_inputs(fats)

    with st.expander("Minerals & Other", expanded=False):
        make_inputs(minerals)

# ── Prediction ─────────────────────────────────────────────────────────────────
X_input = pd.DataFrame([{c: values[c] for c in COLS}])
pred_kcal = float(model.predict(X_input)[0])
pred_kcal = max(0.0, pred_kcal)

# Closest category by Euclidean distance (normalised)
def closest_category(vals):
    # Normalize by max range
    ranges_arr = np.array([RANGES[c][1] - RANGES[c][0] for c in COLS])
    ranges_arr[ranges_arr == 0] = 1
    v = np.array([vals[c] for c in COLS]) / ranges_arr
    best_cat, best_dist = None, float("inf")
    for cat, means in CAT_MEANS.items():
        m = np.array([means.get(c, 0) for c in COLS]) / ranges_arr
        d = np.linalg.norm(v - m)
        if d < best_dist:
            best_dist, best_cat = d, cat
    return best_cat

nearest_cat = closest_category(values)

with result_col:
    st.markdown("### Predicted Kilocalories")
    st.markdown(
        f'<div class="big-number">{pred_kcal:.0f}</div>'
        f'<div class="kcal-label">kcal per 100 g</div>',
        unsafe_allow_html=True,
    )

    st.markdown("")
    st.markdown(
        f'<span class="section-header">Closest food category</span><br>'
        f'<span class="cat-badge">{nearest_cat}</span>',
        unsafe_allow_html=True,
    )

    st.markdown("")
    st.markdown("**Atwater comparison**")
    atwater = values["Data.Carbohydrate"] * 4 + values["Data.Fat.Total Lipid"] * 9 + values["Data.Protein"] * 4
    diff = pred_kcal - atwater
    st.markdown(
        f"Atwater formula estimate: **{atwater:.0f} kcal**  \n"
        f"ML model difference: **{diff:+.0f} kcal** "
        f"({'higher' if diff > 0 else 'lower'} than Atwater)"
    )

    st.divider()

    # Top feature importances bar chart
    st.markdown("**What drives calorie prediction?**")
    fi_sorted = sorted(FI.items(), key=lambda x: x[1], reverse=True)[:8]
    labels = [x[0].replace(" (g)", "").replace(" (mg)", "") for x, _ in fi_sorted]
    vals_fi = [v for _, v in fi_sorted]

    fig = go.Figure(go.Bar(
        x=vals_fi,
        y=labels,
        orientation="h",
        marker_color=["#E8654A" if i < 3 else "#5B8DB8" for i in range(len(labels))],
        text=[f"{v:.3f}" for v in vals_fi],
        textposition="outside",
    ))
    fig.update_layout(
        margin=dict(l=10, r=40, t=10, b=10),
        height=280,
        xaxis_title="Feature Importance",
        yaxis=dict(autorange="reversed"),
        plot_bgcolor="#0E1117",
        paper_bgcolor="#0E1117",
        font=dict(color="#CCC", size=11),
        xaxis=dict(gridcolor="#333"),
    )
    st.plotly_chart(fig, use_container_width=True)

# ── Research context ───────────────────────────────────────────────────────────
st.divider()
with st.expander("About this project"):
    st.markdown("""
**Research:** *Predicting Food Kilocalories from Nutrient Profiles Using Machine Learning*

This app is the live demonstration of research analyzing **7,413 USDA food records** across
**48 nutrient attributes**. The core finding: the traditional Atwater factors (carbs=4, fat=9,
protein=4 kcal/g) are a good approximation, but **water content and fiber** are consistently
among the top predictors in a Random Forest model — not just the macronutrients.

**Model:** Random Forest Regressor (200 trees, per-category training)
**Global performance:** R² = {r2} · MAE = {mae} kcal · RMSE = {rmse} kcal
**Dataset:** USDA National Nutrient Database via Kaggle (food1.csv, 7,413 rows × 48 columns)

**Key findings:**
- Water is the #1 predictor in 10 of 15 food categories
- Total Fat dominates in meat, dairy, and candy categories
- Fiber is the top predictor for cereals (importance = 0.582)
- The ML model outperforms Atwater estimates by ~5 kcal MAE
""".format(**METRICS))
