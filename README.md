# California Housing Price Prediction

A machine learning project that predicts median house values across California districts using the 1990 California Census dataset.

---

## Overview

This project follows a repeatable 5-step ML workflow:
1. Load & inspect the data
2. Clean the data
3. Engineer features
4. Train the models
5. Read the output

Two models are trained and compared — Linear Regression and Random Forest Regressor.

---

## Dataset

**Source:** [California Census Housing Data (1990)](https://www.geeksforgeeks.org/machine-learning/dataset-for-linear-regression/)  
**Rows:** 20,640 districts  
**Columns:** 10 features including location, income, housing age, and ocean proximity  
**Target:** `median_house_value` — median house price per district in USD

---

## Features Used

| Feature | Description |
|---|---|
| `longitude` / `latitude` | Geographic location |
| `housing_median_age` | Age of houses in the district |
| `total_rooms` / `total_bedrooms` | Room counts per district |
| `population` / `households` | District population data |
| `median_income` | Median income of district residents |
| `ocean_proximity` | Distance category from the ocean |
| `rooms_per_household` | Engineered: average rooms per household |
| `bedroom_ratio` | Engineered: ratio of bedrooms to total rooms |
| `population_per_household` | Engineered: average people per household |

---

## Results

| Model | RMSE | R² |
|---|---|---|
| Linear Regression | $76,645 | 0.5517 |
| Random Forest | $50,253 | 0.8073 |

**Random Forest significantly outperforms Linear Regression** because house prices follow non-linear patterns that a straight line cannot capture.

---

## Key Insight

The strongest predictor of house price is **median income** (0.52 importance score) — not location. A wealthy neighborhood commands a price premium regardless of exact coordinates.


---

## How to Run

1. Clone this repository
2. Download the dataset: [California Housing Dataset]
3. Place `housing.csv` in the same folder as the script
4. Install dependencies:
```
pip install pandas numpy scikit-learn matplotlib seaborn
```
5. Run the script:
```
python california_housing_price_prediction.py
```

---

## Project Structure

```
├── california_housing_price_prediction.py   # Main script
├── housing.csv                              # Dataset
└── README.md                                # This file
```

---

