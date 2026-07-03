import streamlit as st
import numpy as np
import pandas as pd
import joblib
import json
import os
import plotly.graph_objects as go

BASE = os.path.dirname(__file__)

@st.cache_resource
def load_model():
    return joblib.load(os.path.join(BASE, "model.joblib"))

@st.cache_data
def load_meta():
    with open(os.path.join(BASE, "model_metadata.json")) as f:
        return json.load(f)

@st.cache_data
def load_foods():
    # Build a searchable food list from category means + presets
    return FOOD_DB

model = load_model()
meta  = load_meta()

COLS     = meta["nutrient_cols"]
FI       = meta["feature_importances"]
CAT_MEANS= meta["category_means"]
METRICS  = meta["global_metrics"]
SHORT    = meta["short_names"]

# ── Food database (name → nutrients per 100g) ──────────────────────────────────
FOOD_DB = {
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
    "Whole Milk": {
        "Data.Carbohydrate": 4.8, "Data.Fat.Total Lipid": 3.3, "Data.Protein": 3.2,
        "Data.Sugar Total": 5.1, "Data.Fiber": 0.0, "Data.Water": 87.8,
        "Data.Fat.Saturated Fat": 1.9, "Data.Fat.Monosaturated Fat": 0.8,
        "Data.Fat.Polysaturated Fat": 0.2, "Data.Major Minerals.Sodium": 44.0,
        "Data.Major Minerals.Calcium": 113.0, "Data.Major Minerals.Potassium": 150.0,
        "Data.Major Minerals.Phosphorus": 84.0, "Data.Cholesterol": 10.0, "Data.Ash": 0.7,
    },
    "Salmon (baked)": {
        "Data.Carbohydrate": 0.0, "Data.Fat.Total Lipid": 13.4, "Data.Protein": 25.4,
        "Data.Sugar Total": 0.0, "Data.Fiber": 0.0, "Data.Water": 59.4,
        "Data.Fat.Saturated Fat": 2.0, "Data.Fat.Monosaturated Fat": 4.7,
        "Data.Fat.Polysaturated Fat": 5.2, "Data.Major Minerals.Sodium": 75.0,
        "Data.Major Minerals.Calcium": 15.0, "Data.Major Minerals.Potassium": 490.0,
        "Data.Major Minerals.Phosphorus": 371.0, "Data.Cholesterol": 85.0, "Data.Ash": 1.5,
    },
    "Banana": {
        "Data.Carbohydrate": 23.0, "Data.Fat.Total Lipid": 0.3, "Data.Protein": 1.1,
        "Data.Sugar Total": 12.2, "Data.Fiber": 2.6, "Data.Water": 74.9,
        "Data.Fat.Saturated Fat": 0.11, "Data.Fat.Monosaturated Fat": 0.03,
        "Data.Fat.Polysaturated Fat": 0.07, "Data.Major Minerals.Sodium": 1.0,
        "Data.Major Minerals.Calcium": 5.0, "Data.Major Minerals.Potassium": 358.0,
        "Data.Major Minerals.Phosphorus": 22.0, "Data.Cholesterol": 0.0, "Data.Ash": 0.8,
    },
    "White Bread": {
        "Data.Carbohydrate": 49.0, "Data.Fat.Total Lipid": 3.2, "Data.Protein": 8.9,
        "Data.Sugar Total": 5.0, "Data.Fiber": 2.7, "Data.Water": 36.0,
        "Data.Fat.Saturated Fat": 0.7, "Data.Fat.Monosaturated Fat": 0.6,
        "Data.Fat.Polysaturated Fat": 1.5, "Data.Major Minerals.Sodium": 490.0,
        "Data.Major Minerals.Calcium": 144.0, "Data.Major Minerals.Potassium": 115.0,
        "Data.Major Minerals.Phosphorus": 96.0, "Data.Cholesterol": 0.0, "Data.Ash": 1.6,
    },
    "Avocado": {
        "Data.Carbohydrate": 8.5, "Data.Fat.Total Lipid": 14.7, "Data.Protein": 2.0,
        "Data.Sugar Total": 0.7, "Data.Fiber": 6.7, "Data.Water": 73.2,
        "Data.Fat.Saturated Fat": 2.1, "Data.Fat.Monosaturated Fat": 9.8,
        "Data.Fat.Polysaturated Fat": 1.8, "Data.Major Minerals.Sodium": 7.0,
        "Data.Major Minerals.Calcium": 12.0, "Data.Major Minerals.Potassium": 485.0,
        "Data.Major Minerals.Phosphorus": 52.0, "Data.Cholesterol": 0.0, "Data.Ash": 1.6,
    },
    "Egg (boiled)": {
        "Data.Carbohydrate": 1.1, "Data.Fat.Total Lipid": 10.6, "Data.Protein": 13.0,
        "Data.Sugar Total": 1.1, "Data.Fiber": 0.0, "Data.Water": 74.6,
        "Data.Fat.Saturated Fat": 3.3, "Data.Fat.Monosaturated Fat": 4.1,
        "Data.Fat.Polysaturated Fat": 1.4, "Data.Major Minerals.Sodium": 124.0,
        "Data.Major Minerals.Calcium": 50.0, "Data.Major Minerals.Potassium": 126.0,
        "Data.Major Minerals.Phosphorus": 172.0, "Data.Cholesterol": 373.0, "Data.Ash": 1.0,
    },
    "Dark Chocolate (70%)": {
        "Data.Carbohydrate": 45.9, "Data.Fat.Total Lipid": 42.6, "Data.Protein": 7.8,
        "Data.Sugar Total": 24.0, "Data.Fiber": 10.9, "Data.Water": 1.3,
        "Data.Fat.Saturated Fat": 24.5, "Data.Fat.Monosaturated Fat": 12.8,
        "Data.Fat.Polysaturated Fat": 1.3, "Data.Major Minerals.Sodium": 20.0,
        "Data.Major Minerals.Calcium": 73.0, "Data.Major Minerals.Potassium": 715.0,
        "Data.Major Minerals.Phosphorus": 308.0, "Data.Cholesterol": 3.0, "Data.Ash": 2.3,
    },
    "Apple": {
        "Data.Carbohydrate": 13.8, "Data.Fat.Total Lipid": 0.2, "Data.Protein": 0.3,
        "Data.Sugar Total": 10.4, "Data.Fiber": 2.4, "Data.Water": 85.6,
        "Data.Fat.Saturated Fat": 0.03, "Data.Fat.Monosaturated Fat": 0.01,
        "Data.Fat.Polysaturated Fat": 0.05, "Data.Major Minerals.Sodium": 1.0,
        "Data.Major Minerals.Calcium": 6.0, "Data.Major Minerals.Potassium": 107.0,
        "Data.Major Minerals.Phosphorus": 11.0, "Data.Cholesterol": 0.0, "Data.Ash": 0.2,
    },
    "Butter": {
        "Data.Carbohydrate": 0.1, "Data.Fat.Total Lipid": 81.1, "Data.Protein": 0.9,
        "Data.Sugar Total": 0.1, "Data.Fiber": 0.0, "Data.Water": 17.4,
        "Data.Fat.Saturated Fat": 51.4, "Data.Fat.Monosaturated Fat": 21.0,
        "Data.Fat.Polysaturated Fat": 3.0, "Data.Major Minerals.Sodium": 576.0,
        "Data.Major Minerals.Calcium": 24.0, "Data.Major Minerals.Potassium": 24.0,
        "Data.Major Minerals.Phosphorus": 24.0, "Data.Cholesterol": 215.0, "Data.Ash": 2.1,
    },
    "Almonds": {
        "Data.Carbohydrate": 21.7, "Data.Fat.Total Lipid": 49.4, "Data.Protein": 21.2,
        "Data.Sugar Total": 3.9, "Data.Fiber": 12.5, "Data.Water": 4.4,
        "Data.Fat.Saturated Fat": 3.7, "Data.Fat.Monosaturated Fat": 31.6,
        "Data.Fat.Polysaturated Fat": 12.1, "Data.Major Minerals.Sodium": 1.0,
        "Data.Major Minerals.Calcium": 264.0, "Data.Major Minerals.Potassium": 705.0,
        "Data.Major Minerals.Phosphorus": 484.0, "Data.Cholesterol": 0.0, "Data.Ash": 3.0,
    },
}

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Food Calorie Predictor",
    page_icon="🥗",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
  .big-number   { font-size: 5rem; font-weight: 800; color: #E8654A; line-height: 1.0; }
  .kcal-label   { font-size: 1.1rem; color: #888; margin-top: -4px; margin-bottom: 20px; }
  .cat-badge    { display: inline-block; background: #2D4A3E; color: #7EC8A4;
                  border-radius: 20px; padding: 4px 14px; font-size: 0.9rem; font-weight: 600; }
  .section-hdr  { color: #aaa; font-size: 0.72rem; text-transform: uppercase;
                  letter-spacing: 1.5px; margin-bottom: 2px; }
  .nutrient-row { display: flex; justify-content: space-between; padding: 4px 0;
                  border-bottom: 1px solid #222; font-size: 0.9rem; }
  .nutrient-val { color: #E8654A; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────────────────────────
title_col, credit_col = st.columns([4, 1])
with title_col:
    st.markdown("# 🥗 How Many Calories Is That Food?")
    st.markdown(
        "Pick any food below and our AI will instantly tell you how many calories it has — "
        "and explain *why* in plain English."
    )
with credit_col:
    st.markdown("<div style='text-align:right; padding-top:18px; color:#aaa; font-size:0.85rem;'>Project by<br><strong style='color:#fff;'>Saketh Raj Kolar</strong></div>", unsafe_allow_html=True)
st.divider()

# ── Food selector ──────────────────────────────────────────────────────────────
food_names = list(FOOD_DB.keys())
selected = st.selectbox(
    "🔍 Pick a food",
    food_names,
    index=0,
    help="Choose any food from the list to see its calorie prediction",
)

nutrients = FOOD_DB[selected]

# ── Prediction ─────────────────────────────────────────────────────────────────
X_input   = pd.DataFrame([{c: nutrients[c] for c in COLS}])
pred_kcal = max(0.0, float(model.predict(X_input)[0]))
atwater   = (nutrients["Data.Carbohydrate"] * 4
             + nutrients["Data.Fat.Total Lipid"] * 9
             + nutrients["Data.Protein"] * 4)
diff      = pred_kcal - atwater

def closest_category(vals):
    best_cat, best_dist = None, float("inf")
    for cat, means in CAT_MEANS.items():
        d = sum((vals.get(c, 0) - means.get(c, 0)) ** 2 for c in COLS) ** 0.5
        if d < best_dist:
            best_dist, best_cat = d, cat
    return best_cat

nearest_cat = closest_category(nutrients)

# ── Layout: result left, breakdown right ───────────────────────────────────────
st.markdown("")
res_col, break_col = st.columns([1, 2], gap="large")

with res_col:
    st.markdown(
        f'<div class="big-number">{pred_kcal:.0f}</div>'
        f'<div class="kcal-label">calories per 100 g (about a handful)</div>',
        unsafe_allow_html=True,
    )

    # Fun calorie comparison
    apple_cals = 52
    equiv = pred_kcal / apple_cals
    st.info(f"💡 That's roughly the same calories as **{equiv:.1f} apples** (52 cal each)")

    st.markdown("")
    st.markdown("### What's in it?")
    m1, m2, m3 = st.columns(3)
    m1.metric("💪 Protein", f"{nutrients['Data.Protein']:.1f}g", help="Builds muscles")
    m2.metric("⚡ Carbs",   f"{nutrients['Data.Carbohydrate']:.1f}g", help="Your body's main fuel")
    m3.metric("🧈 Fat",     f"{nutrients['Data.Fat.Total Lipid']:.1f}g", help="Stores energy, helps absorb vitamins")

    water_pct = nutrients["Data.Water"]
    st.markdown(f"💧 **{water_pct:.0f}% is water** — the more water a food has, the fewer calories it packs in.")

    st.markdown("")
    st.markdown(
        f'<div class="section-hdr">This food is most similar to</div>'
        f'<span class="cat-badge">{nearest_cat}</span>',
        unsafe_allow_html=True,
    )

    st.markdown("")
    st.markdown("### 🤖 Why does our AI predict this?")

    if abs(diff) < 5:
        diff_explain = (
            f"The simple math formula also predicts **{atwater:.0f} cal** — "
            f"our AI agrees! For this food, the basic formula works well."
        )
    elif diff > 0:
        diff_explain = (
            f"A simple math formula (just multiplying protein, carbs, fat) "
            f"would guess **{atwater:.0f} cal**, but our AI says **{pred_kcal:.0f} cal** — "
            f"**{diff:+.0f} higher**, because it also picks up on other nutrients "
            f"like minerals and ash content."
        )
    else:
        diff_explain = (
            f"A simple math formula (just multiplying protein, carbs, fat) "
            f"would guess **{atwater:.0f} cal**, but our AI says **{pred_kcal:.0f} cal** — "
            f"**{abs(diff):.0f} fewer**, because it noticed this food has "
            f"a lot of water or fiber which dilute the calories."
        )
    st.markdown(diff_explain)

with break_col:
    # ── Macronutrient donut ────────────────────────────────────────────────────
    carb = nutrients["Data.Carbohydrate"]
    fat  = nutrients["Data.Fat.Total Lipid"]
    prot = nutrients["Data.Protein"]
    fib  = nutrients["Data.Fiber"]
    watr = nutrients["Data.Water"]
    other= max(0, 100 - carb - fat - prot - fib - watr)

    st.markdown("### 🥧 What is this food made of?")
    st.caption("Each slice shows how much of 100g is each nutrient")
    donut = go.Figure(go.Pie(
        labels=["Carbs", "Fat", "Protein", "Fiber", "Water", "Other"],
        values=[carb, fat, prot, fib, watr, other],
        hole=0.55,
        marker_colors=["#4E91D2", "#E8654A", "#7EC8A4", "#F5C842", "#96C8D8", "#888"],
        textinfo="label+percent",
        hovertemplate="%{label}: %{value:.1f}g out of 100g<extra></extra>",
    ))
    donut.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        height=280,
        showlegend=True,
        legend=dict(font=dict(color="#CCC", size=11)),
        paper_bgcolor="#0E1117",
        font=dict(color="#CCC", size=11),
    )
    st.plotly_chart(donut, use_container_width=True)

    # ── Feature importance bar ────────────────────────────────────────────────
    st.markdown("### 🧠 What does the AI look at most?")
    st.caption("The AI learned these are the most important clues for predicting calories")
    fi_sorted = sorted(FI.items(), key=lambda x: x[1], reverse=True)[:6]
    labels_fi = [x[0].replace(" (g)", "").replace(" (mg)", "") for x, _ in fi_sorted]
    vals_fi   = [v for _, v in fi_sorted]
    pct_fi    = [f"{v*100:.1f}%" for v in vals_fi]

    bar = go.Figure(go.Bar(
        x=vals_fi, y=labels_fi, orientation="h",
        marker_color=["#E8654A" if i < 2 else "#5B8DB8" for i in range(len(labels_fi))],
        text=pct_fi, textposition="outside",
        hovertemplate="%{y}: accounts for %{text} of the prediction<extra></extra>",
    ))
    bar.update_layout(
        margin=dict(l=10, r=60, t=10, b=10),
        height=250,
        xaxis_title="How much this nutrient matters (0% = ignored, 100% = everything)",
        yaxis=dict(autorange="reversed"),
        plot_bgcolor="#0E1117", paper_bgcolor="#0E1117",
        font=dict(color="#CCC", size=11),
        xaxis=dict(gridcolor="#333", tickformat=".0%"),
    )
    st.plotly_chart(bar, use_container_width=True)
    st.caption(
        "💡 Surprising fact: **Water** is the #1 clue in most foods — "
        "watery foods like broccoli are naturally low-calorie, "
        "dry foods like almonds are high-calorie."
    )

# ── Nutrient detail (collapsible) ──────────────────────────────────────────────
st.divider()
with st.expander("🔬 See all nutrient numbers (for the curious)"):
    st.caption("These are the exact values per 100g that the AI used to make its prediction")
    c1, c2, c3 = st.columns(3)
    items = [(SHORT[k].replace(" (g)", " g").replace(" (mg)", " mg"), nutrients[k])
             for k in COLS]
    third = len(items) // 3
    for i, (label, val) in enumerate(items):
        col = c1 if i < third else (c2 if i < 2 * third else c3)
        col.metric(label, f"{val:.2f}")

# ── About ──────────────────────────────────────────────────────────────────────
with st.expander("📖 About this project"):
    st.markdown(f"""
**What is this?**
This is a science project that uses AI to predict how many calories are in food.
Instead of just multiplying protein × 4 + fat × 9 + carbs × 4 (the old way),
our AI learned from **7,413 real foods** in the USDA database and figured out
that water content, fiber, and other nutrients also matter a lot.

**How accurate is it?**
Our AI is off by only **{METRICS['mae']} calories on average** — that's incredibly close
for predicting something as complex as food energy!

**The big surprise we found:**
Most people think fat and protein are the main drivers of calories.
But our AI discovered that **water content is actually the #1 clue** —
foods with lots of water (like vegetables) are almost always low-calorie,
and dry foods (like nuts or chocolate) are almost always high-calorie.

**Who made this?**
Built by **Saketh Raj Kolar** as part of an 8-week research project analyzing nutrition data.
*Model: Random Forest · 200 decision trees · R²={METRICS['r2']} accuracy*
""")
