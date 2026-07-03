# Food Calorie Predictor

A machine learning web app that predicts the kilocalories in any food from its nutrient profile (per 100g), trained on 7,413 USDA food records.

**Live app:** https://food-kcal-predictor.streamlit.app

## What it does

- Enter nutrient values (carbohydrates, fat, protein, water, fiber, minerals, etc.)
- Get an instant calorie prediction from a Random Forest model
- See which nutrients drive the prediction (feature importance)
- Compare against the traditional Atwater formula estimate
- Load presets for common foods (chicken, rice, cheese, olive oil, broccoli)

## Model performance

| Metric | Value |
|--------|-------|
| R²     | 0.9974 |
| MAE    | 5.02 kcal |
| RMSE   | 8.47 kcal |

## Dataset

USDA National Nutrient Database · 7,413 food records · 48 nutrient attributes

## Research

Built as part of an original research project: *Predicting Food Kilocalories from Nutrient Profiles Using Machine Learning*, analyzing how water content, fat, fiber, and minerals drive caloric variation across 29 food categories.

Key finding: Water is the #1 predictor in 10/15 food categories — not the macronutrients typically tracked by apps.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```
