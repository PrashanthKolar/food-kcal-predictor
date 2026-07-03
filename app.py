import streamlit as st
import numpy as np
import pandas as pd
import joblib
import json
import os
import plotly.graph_objects as go
import random

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

COLS      = meta["nutrient_cols"]
FI        = meta["feature_importances"]
CAT_MEANS = meta["category_means"]
METRICS   = meta["global_metrics"]
SHORT     = meta["short_names"]

# ── Food database ──────────────────────────────────────────────────────────────
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

def predict(nutrients):
    X = pd.DataFrame([{c: nutrients[c] for c in COLS}])
    return max(0.0, float(model.predict(X)[0]))

def closest_category(nutrients):
    best_cat, best_dist = None, float("inf")
    for cat, means in CAT_MEANS.items():
        d = sum((nutrients.get(c, 0) - means.get(c, 0)) ** 2 for c in COLS) ** 0.5
        if d < best_dist:
            best_dist, best_cat = d, cat
    return best_cat

FOOD_NAMES = list(FOOD_DB.keys())

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Food Calorie Predictor",
    page_icon="🥗",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
  .big-number  { font-size: 5rem; font-weight: 800; color: #E8654A; line-height: 1.0; }
  .kcal-label  { font-size: 1.1rem; color: #888; margin-top: -4px; margin-bottom: 16px; }
  .cat-badge   { display: inline-block; background: #2D4A3E; color: #7EC8A4;
                 border-radius: 20px; padding: 4px 14px; font-size: 0.9rem; font-weight: 600; }
  .section-hdr { color: #aaa; font-size: 0.72rem; text-transform: uppercase;
                 letter-spacing: 1.5px; margin-bottom: 2px; }
  .winner-box  { background: #2D4A3E; border-radius: 12px; padding: 14px 20px;
                 text-align: center; color: #7EC8A4; font-size: 1.1rem; font-weight: 700; }
</style>
""", unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────────────────────────
title_col, credit_col = st.columns([4, 1])
with title_col:
    st.markdown("# 🥗 How Many Calories Is That Food?")
    st.markdown("An AI trained on 7,400+ USDA food records predicts calories and explains *why*.")
with credit_col:
    st.markdown(
        "<div style='text-align:right; padding-top:18px; color:#aaa; font-size:0.85rem;'>"
        "Project by<br><strong style='color:#fff;'>Saketh Raj Kolar</strong></div>",
        unsafe_allow_html=True,
    )
st.divider()

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["🔍 Explore", "🎯 Guess the Calories", "⚖️ Compare Foods", "🏃 Burn it Off"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — EXPLORE
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    selected = st.selectbox("🔍 Pick a food", FOOD_NAMES, key="explore_select")
    nutrients = FOOD_DB[selected]
    pred_kcal = predict(nutrients)
    atwater   = nutrients["Data.Carbohydrate"]*4 + nutrients["Data.Fat.Total Lipid"]*9 + nutrients["Data.Protein"]*4
    diff      = pred_kcal - atwater
    nearest_cat = closest_category(nutrients)

    st.markdown("")
    res_col, break_col = st.columns([1, 2], gap="large")

    with res_col:
        st.markdown(
            f'<div class="big-number">{pred_kcal:.0f}</div>'
            f'<div class="kcal-label">calories per 100 g</div>',
            unsafe_allow_html=True,
        )
        equiv = pred_kcal / 52
        st.info(f"💡 Same calories as **{equiv:.1f} apples**")

        st.markdown("### What's in it?")
        m1, m2, m3 = st.columns(3)
        m1.metric("💪 Protein", f"{nutrients['Data.Protein']:.1f}g")
        m2.metric("⚡ Carbs",   f"{nutrients['Data.Carbohydrate']:.1f}g")
        m3.metric("🧈 Fat",     f"{nutrients['Data.Fat.Total Lipid']:.1f}g")
        st.markdown(f"💧 **{nutrients['Data.Water']:.0f}% is water** — more water = fewer calories")

        st.markdown("")
        st.markdown(
            f'<div class="section-hdr">Most similar USDA category</div>'
            f'<span class="cat-badge">{nearest_cat}</span>',
            unsafe_allow_html=True,
        )

        st.markdown("")
        st.markdown("### 🤖 Why this prediction?")
        if abs(diff) < 5:
            st.markdown(f"The simple math formula also gives **{atwater:.0f} cal** — our AI agrees!")
        elif diff < 0:
            st.markdown(
                f"Simple formula guesses **{atwater:.0f} cal**, but our AI says **{pred_kcal:.0f}** — "
                f"**{abs(diff):.0f} fewer**, because it noticed the high water/fiber content."
            )
        else:
            st.markdown(
                f"Simple formula guesses **{atwater:.0f} cal**, but our AI says **{pred_kcal:.0f}** — "
                f"**{diff:.0f} more**, because it also weighs minerals and other nutrients."
            )

    with break_col:
        carb = nutrients["Data.Carbohydrate"]
        fat  = nutrients["Data.Fat.Total Lipid"]
        prot = nutrients["Data.Protein"]
        fib  = nutrients["Data.Fiber"]
        watr = nutrients["Data.Water"]
        other = max(0, 100 - carb - fat - prot - fib - watr)

        st.markdown("### 🥧 What is it made of?")
        st.caption("Each slice = grams out of 100g")
        donut = go.Figure(go.Pie(
            labels=["Carbs", "Fat", "Protein", "Fiber", "Water", "Other"],
            values=[carb, fat, prot, fib, watr, other],
            hole=0.55,
            marker_colors=["#4E91D2", "#E8654A", "#7EC8A4", "#F5C842", "#96C8D8", "#888"],
            textinfo="label+percent",
            hovertemplate="%{label}: %{value:.1f}g<extra></extra>",
        ))
        donut.update_layout(
            margin=dict(l=10, r=10, t=10, b=10), height=270,
            paper_bgcolor="#0E1117", font=dict(color="#CCC", size=11),
            legend=dict(font=dict(color="#CCC", size=11)),
        )
        st.plotly_chart(donut, use_container_width=True)

        st.markdown("### 🧠 What does the AI look at most?")
        st.caption("Higher % = this nutrient matters more for predicting calories")
        fi_sorted = sorted(FI.items(), key=lambda x: x[1], reverse=True)[:6]
        labels_fi = [x[0].replace(" (g)","").replace(" (mg)","") for x, _ in fi_sorted]
        vals_fi   = [v for _, v in fi_sorted]
        bar = go.Figure(go.Bar(
            x=vals_fi, y=labels_fi, orientation="h",
            marker_color=["#E8654A" if i < 2 else "#5B8DB8" for i in range(len(labels_fi))],
            text=[f"{v*100:.1f}%" for v in vals_fi], textposition="outside",
        ))
        bar.update_layout(
            margin=dict(l=10, r=60, t=10, b=10), height=240,
            xaxis=dict(tickformat=".0%", gridcolor="#333"),
            yaxis=dict(autorange="reversed"),
            plot_bgcolor="#0E1117", paper_bgcolor="#0E1117",
            font=dict(color="#CCC", size=11),
        )
        st.plotly_chart(bar, use_container_width=True)
        st.caption("💡 Water is the #1 clue — watery foods are almost always low-calorie.")

    st.divider()
    with st.expander("🔬 See all nutrient numbers"):
        st.caption("Exact values per 100g used by the AI")
        c1, c2, c3 = st.columns(3)
        items = [(SHORT[k].replace(" (g)"," g").replace(" (mg)"," mg"), nutrients[k]) for k in COLS]
        third = len(items) // 3
        for i, (label, val) in enumerate(items):
            (c1 if i < third else c2 if i < 2*third else c3).metric(label, f"{val:.2f}")

    with st.expander("📖 About this project"):
        st.markdown(f"""
**What is this?**
An AI trained on **7,413 real foods** from the USDA database predicts calories more accurately
than the traditional formula (protein×4 + fat×9 + carbs×4) by also learning from water,
fiber, and mineral content.

**How accurate?** Off by only **{METRICS['mae']} calories on average** (R²={METRICS['r2']}).

**The big finding:** Water content is the #1 predictor — not fat or protein.
Watery foods are almost always low-calorie; dry foods are almost always high-calorie.

Built by **Saketh Raj Kolar** · Random Forest · 200 trees · 7,413 USDA records
""")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — GUESS THE CALORIES
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### 🎯 Can you guess the calories?")
    st.markdown("A random food is chosen. Guess how many calories it has per 100g, then reveal the answer!")

    if "game_food" not in st.session_state:
        st.session_state.game_food = random.choice(FOOD_NAMES)
        st.session_state.revealed  = False
        st.session_state.score     = 0
        st.session_state.rounds    = 0

    food_name = st.session_state.game_food
    game_nutrients = FOOD_DB[food_name]
    actual_kcal = predict(game_nutrients)

    col_game, col_score = st.columns([3, 1])

    with col_score:
        st.markdown("### 🏆 Score")
        st.metric("Correct guesses", st.session_state.score)
        st.metric("Total rounds", st.session_state.rounds)
        if st.session_state.rounds > 0:
            pct = int(st.session_state.score / st.session_state.rounds * 100)
            st.metric("Accuracy", f"{pct}%")

    with col_game:
        st.markdown(f"## 🍽️ **{food_name}**")
        st.caption("Per 100g — that's roughly a small handful")

        guess = st.number_input(
            "Your calorie guess:", min_value=0, max_value=1000, value=100, step=5,
            key="guess_input"
        )

        c1, c2 = st.columns(2)
        reveal_btn = c1.button("🎲 Reveal Answer", type="primary", use_container_width=True)
        next_btn   = c2.button("⏭️ Next Food", use_container_width=True)

        if reveal_btn and not st.session_state.revealed:
            st.session_state.revealed = True
            st.session_state.rounds  += 1
            error = abs(guess - actual_kcal)
            if error <= 20:
                st.session_state.score += 1

        if next_btn:
            st.session_state.game_food = random.choice(FOOD_NAMES)
            st.session_state.revealed  = False
            st.rerun()

        if st.session_state.revealed:
            error = abs(guess - actual_kcal)
            st.markdown("---")
            res1, res2 = st.columns(2)
            res1.markdown(
                f'<div style="text-align:center">'
                f'<div style="font-size:1rem;color:#aaa">Your guess</div>'
                f'<div style="font-size:3rem;font-weight:800;color:#5B8DB8">{guess}</div>'
                f'<div style="color:#aaa">kcal</div></div>',
                unsafe_allow_html=True,
            )
            res2.markdown(
                f'<div style="text-align:center">'
                f'<div style="font-size:1rem;color:#aaa">AI prediction</div>'
                f'<div style="font-size:3rem;font-weight:800;color:#E8654A">{actual_kcal:.0f}</div>'
                f'<div style="color:#aaa">kcal</div></div>',
                unsafe_allow_html=True,
            )

            st.markdown("")
            if error <= 10:
                st.success(f"🔥 Incredible! You were only {error:.0f} cal off — almost perfect!")
            elif error <= 20:
                st.success(f"✅ Great guess! Only {error:.0f} cal off. You win this round!")
            elif error <= 50:
                st.warning(f"👍 Not bad — you were {error:.0f} cal off. Getting warmer!")
            elif error <= 100:
                st.warning(f"😅 You were {error:.0f} cal off. Tricky one!")
            else:
                st.error(f"😲 You were {error:.0f} cal off — this food is surprising!")

            st.markdown(
                f"**Why {actual_kcal:.0f} cal?** {food_name} is "
                f"**{game_nutrients['Data.Water']:.0f}% water**, "
                f"has **{game_nutrients['Data.Fat.Total Lipid']:.1f}g fat** and "
                f"**{game_nutrients['Data.Carbohydrate']:.1f}g carbs** per 100g."
            )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — COMPARE FOODS
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### ⚖️ Which food has more calories?")
    st.markdown("Pick two foods and see them go head to head.")

    c1, c2 = st.columns(2)
    food_a = c1.selectbox("🅰️ Food A", FOOD_NAMES, index=0, key="compare_a")
    food_b = c2.selectbox("🅱️ Food B", FOOD_NAMES, index=2, key="compare_b")

    nut_a = FOOD_DB[food_a]
    nut_b = FOOD_DB[food_b]
    kcal_a = predict(nut_a)
    kcal_b = predict(nut_b)

    st.markdown("")

    # Calorie banner
    ba, mid, bb = st.columns([2, 1, 2])
    ba.markdown(
        f'<div style="text-align:center;background:#1E1E2E;border-radius:12px;padding:20px">'
        f'<div style="font-size:0.9rem;color:#aaa">{food_a}</div>'
        f'<div style="font-size:3.5rem;font-weight:800;color:#{"E8654A" if kcal_a >= kcal_b else "5B8DB8"}">{kcal_a:.0f}</div>'
        f'<div style="color:#aaa">kcal / 100g</div></div>',
        unsafe_allow_html=True,
    )
    mid.markdown(
        f'<div style="text-align:center;padding-top:40px;font-size:2rem;font-weight:800;color:#aaa">VS</div>',
        unsafe_allow_html=True,
    )
    bb.markdown(
        f'<div style="text-align:center;background:#1E1E2E;border-radius:12px;padding:20px">'
        f'<div style="font-size:0.9rem;color:#aaa">{food_b}</div>'
        f'<div style="font-size:3.5rem;font-weight:800;color:#{"E8654A" if kcal_b >= kcal_a else "5B8DB8"}">{kcal_b:.0f}</div>'
        f'<div style="color:#aaa">kcal / 100g</div></div>',
        unsafe_allow_html=True,
    )

    st.markdown("")
    if abs(kcal_a - kcal_b) < 5:
        st.info("🤝 It's basically a tie! These two foods have almost the same calories.")
    elif kcal_a > kcal_b:
        diff_pct = (kcal_a - kcal_b) / kcal_b * 100
        st.markdown(
            f'<div class="winner-box">🏆 {food_a} has {kcal_a - kcal_b:.0f} more calories '
            f'({diff_pct:.0f}% higher than {food_b})</div>',
            unsafe_allow_html=True,
        )
    else:
        diff_pct = (kcal_b - kcal_a) / kcal_a * 100
        st.markdown(
            f'<div class="winner-box">🏆 {food_b} has {kcal_b - kcal_a:.0f} more calories '
            f'({diff_pct:.0f}% higher than {food_a})</div>',
            unsafe_allow_html=True,
        )

    # Side-by-side bar comparison
    st.markdown("")
    st.markdown("### Nutrient comparison")
    nutrients_to_show = ["Data.Protein", "Data.Carbohydrate", "Data.Fat.Total Lipid",
                         "Data.Fiber", "Data.Water", "Data.Sugar Total"]
    labels_cmp = ["Protein", "Carbs", "Fat", "Fiber", "Water", "Sugar"]
    vals_a = [nut_a[n] for n in nutrients_to_show]
    vals_b = [nut_b[n] for n in nutrients_to_show]

    fig_cmp = go.Figure()
    fig_cmp.add_trace(go.Bar(name=food_a, x=labels_cmp, y=vals_a, marker_color="#E8654A"))
    fig_cmp.add_trace(go.Bar(name=food_b, x=labels_cmp, y=vals_b, marker_color="#5B8DB8"))
    fig_cmp.update_layout(
        barmode="group",
        margin=dict(l=10, r=10, t=10, b=10), height=300,
        yaxis_title="grams per 100g",
        plot_bgcolor="#0E1117", paper_bgcolor="#0E1117",
        font=dict(color="#CCC", size=11),
        legend=dict(font=dict(color="#CCC")),
        yaxis=dict(gridcolor="#333"),
    )
    st.plotly_chart(fig_cmp, use_container_width=True)

    # Plain English explanation
    st.markdown("### 💬 Why the difference?")
    higher_food = food_a if kcal_a >= kcal_b else food_b
    lower_food  = food_b if kcal_a >= kcal_b else food_a
    higher_nut  = nut_a  if kcal_a >= kcal_b else nut_b
    lower_nut   = nut_b  if kcal_a >= kcal_b else nut_a

    water_diff = lower_nut["Data.Water"] - higher_nut["Data.Water"]
    fat_diff   = higher_nut["Data.Fat.Total Lipid"] - lower_nut["Data.Fat.Total Lipid"]

    reasons = []
    if water_diff > 10:
        reasons.append(f"**{lower_food}** has {water_diff:.0f}% more water, which dilutes its calories")
    if fat_diff > 5:
        reasons.append(f"**{higher_food}** has {fat_diff:.0f}g more fat (fat = 9 cal/g, the most calorie-dense nutrient)")
    carb_diff = higher_nut["Data.Carbohydrate"] - lower_nut["Data.Carbohydrate"]
    if carb_diff > 10:
        reasons.append(f"**{higher_food}** has {carb_diff:.0f}g more carbs")

    if reasons:
        for r in reasons:
            st.markdown(f"- {r}")
    else:
        st.markdown("These foods have similar nutrient profiles, which is why their calorie counts are close.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — BURN IT OFF
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("### 🏃 How long to burn it off?")
    st.markdown("Pick a food and see how much exercise it takes to burn those calories.")

    burn_food = st.selectbox("🔍 Pick a food", FOOD_NAMES, key="burn_select")

    col_portion, col_weight = st.columns(2)
    portion_g = col_portion.slider("Portion size (grams)", min_value=10, max_value=500,
                                    value=100, step=10)
    weight_kg = col_weight.slider("Your body weight (kg)", min_value=30, max_value=150,
                                   value=70, step=5)

    burn_nutrients = FOOD_DB[burn_food]
    kcal_per_100g  = predict(burn_nutrients)
    total_kcal     = kcal_per_100g * portion_g / 100

    st.markdown("")
    st.markdown(
        f'<div class="big-number">{total_kcal:.0f}</div>'
        f'<div class="kcal-label">calories in {portion_g}g of {burn_food}</div>',
        unsafe_allow_html=True,
    )
    st.markdown("")

    # MET values × weight × time (hours) = calories burned
    # calories = MET × weight_kg × hours
    activities = {
        "🚶 Walking (casual)":        3.5,
        "🚴 Cycling (moderate)":      8.0,
        "🏃 Running (jogging)":       9.8,
        "🏊 Swimming (laps)":         8.3,
        "⚽ Playing soccer":          10.0,
        "🧘 Yoga":                    2.5,
        "💃 Dancing":                 5.0,
        "🏋️ Weight training":         5.0,
    }

    st.markdown("### ⏱️ Time needed to burn it off")
    cols = st.columns(4)
    for i, (activity, met) in enumerate(activities.items()):
        cal_per_min = met * weight_kg / 60
        minutes = total_kcal / cal_per_min
        hours   = int(minutes // 60)
        mins    = int(minutes % 60)
        time_str = f"{hours}h {mins}m" if hours > 0 else f"{mins} min"
        cols[i % 4].metric(activity, time_str)

    st.markdown("")

    # Bar chart of burn times
    st.markdown("### 📊 Exercise comparison")
    act_names = [a.split(" (")[0] for a in activities]
    burn_mins = [total_kcal / (met * weight_kg / 60) for met in activities.values()]

    fig_burn = go.Figure(go.Bar(
        x=act_names, y=burn_mins,
        marker_color=["#E8654A" if m > 60 else "#7EC8A4" for m in burn_mins],
        text=[f"{int(m)} min" for m in burn_mins],
        textposition="outside",
        hovertemplate="%{x}: %{y:.0f} minutes<extra></extra>",
    ))
    fig_burn.update_layout(
        margin=dict(l=10, r=10, t=10, b=10), height=320,
        yaxis_title="Minutes of exercise",
        plot_bgcolor="#0E1117", paper_bgcolor="#0E1117",
        font=dict(color="#CCC", size=11),
        yaxis=dict(gridcolor="#333"),
        xaxis=dict(tickangle=-20),
    )
    fig_burn.add_hline(y=60, line_dash="dot", line_color="#aaa",
                        annotation_text="1 hour", annotation_position="top right")
    st.plotly_chart(fig_burn, use_container_width=True)

    st.caption(
        f"Burn times estimated for a {weight_kg}kg person. "
        "Running burns calories ~3× faster than walking!"
    )
