# 🏠 House Price Prediction — Analysis Dashboard

A machine learning web application that predicts house prices and provides
an interactive Power-BI-style analytics dashboard, built with **Python**,
**scikit-learn**, **Plotly**, and **Streamlit**.

---

## 📌 Project Overview

This project combines a **Linear Regression** model with an interactive,
filterable dashboard to analyze and predict house prices. Users can filter
the dataset by bedrooms, bathrooms, parking, area, and price range, view
live KPI cards and charts that respond to those filters, predict the price
of a custom house, and inspect the model's accuracy through diagnostic
charts — all inside a single Streamlit web application.

---

## 🎯 Objectives

- Build a regression model to predict house prices from structured data.
- Provide an interactive dashboard for exploring pricing trends by
  bedrooms, bathrooms, parking, and area.
- Allow real-time filtering of the dataset via sidebar slicers.
- Evaluate the model's accuracy using standard regression metrics.
- Provide an interactive, user-friendly interface for real-time price
  predictions.

---

## 🗂️ Dataset

- **File:** `HOUSE_PRICE_PREDICTION.csv`
- **Resource:** [Housing Prices Dataset (Kaggle)](https://www.kaggle.com/datasets/yasserh/housing-prices-dataset)
- **Records:** 545 houses
- **Features used:**

| Column      | Description                          |
|-------------|---------------------------------------|
| `area`      | Total area of the house (sq ft)       |
| `bedrooms`  | Number of bedrooms                    |
| `bathrooms` | Number of bathrooms                   |
| `parking`   | Number of parking spaces              |
| `price`     | Target variable — price of the house (₹) |

---

## 🛠️ Technologies Used

| Category            | Tools / Libraries            |
|----------------------|-------------------------------|
| Programming Language | Python 3.13                  |
| Web Framework         | Streamlit                    |
| Machine Learning       | scikit-learn (Linear Regression) |
| Data Handling          | pandas, numpy                |
| Visualization           | Plotly, statsmodels (for trendlines) |

---

## ⚙️ Project Workflow

1. **Data Loading** – The housing dataset is loaded using pandas.
2. **Preprocessing** – Features (`area`, `bedrooms`, `bathrooms`, `parking`)
   and target (`price`) are separated.
3. **Train-Test Split** – 80% of the data is used for training and 20% for
   testing (`random_state=42` for reproducibility).
4. **Model Training** – A Linear Regression model is trained on the
   training set.
5. **Evaluation** – Model performance is measured using:
   - **MAE** (Mean Absolute Error)
   - **RMSE** (Root Mean Squared Error)
   - **R² Score**
6. **Deployment** – The trained model and dashboard are served through a
   Streamlit web app with filters, KPIs, charts, and a prediction
   interface.

---

## 📊 Application Features

The web app is organized into three tabs:

### 📊 Dashboard
A Power-BI-style analytics view of the housing dataset:
- **Sidebar filters (slicers)** — filter by bedrooms, bathrooms, parking,
  area range, and price range, with a "Clear all filters" option.
- **KPI cards** — Average Price, Avg Bathrooms, Median Bedrooms, Average
  Parking, and Average Area, all recalculated live based on the current
  filters.
- **Charts** — Avg Price by Bedrooms, Avg Price by Bathrooms, Avg Price by
  Parking, House Area vs. Price scatter plot, Price Distribution
  histogram, Bedrooms Split donut chart, Parking Mix pie chart, and Avg
  Price by Bedrooms & Bathrooms (stacked bar) — all update instantly as
  filters change.

### 🔮 Predict
- Enter house details (area, bedrooms, bathrooms, parking) to get an
  estimated price.
- View how the input compares to the dataset average using a radar chart.
- See where the prediction falls within the overall price distribution.

### 🎯 Model Performance
- Evaluation metrics: MAE, RMSE, R² Score.
- Actual vs. Predicted price scatter plot.
- Residual plots (vs. predicted price, and residual distribution).
- Feature importance chart based on regression coefficients.

---

## 🚀 How to Run the Project

1. **Clone / download the project folder**, ensuring it contains:
   - `app.py`
   - `HOUSE_PRICE_PREDICTION.csv`
   - `requirements.txt`

2. **Create and activate a virtual environment (recommended)**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate      # Windows
   source .venv/bin/activate   # macOS / Linux
   ```

3. **Install the required dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Streamlit application**
   ```bash
   streamlit run app.py
   ```

5. **Open the app** in your browser at the local URL shown in the terminal
   (typically `http://localhost:8501`).

---

## 📈 Results

The Linear Regression model achieves a reasonable fit on the test set, with
performance metrics (MAE, RMSE, R²) displayed directly within the
**🎯 Model Performance** tab of the application. The **📊 Dashboard** tab
allows further exploration of how price varies with property features
across the full (or filtered) dataset.

---

## 🔮 Future Enhancements

- Add more features (location, house age, furnishing status,
  airconditioning, basement, etc.) if a fuller dataset becomes available.
- Compare multiple algorithms (Random Forest, Gradient Boosting) for
  improved accuracy.
- Deploy the application on a public platform such as Streamlit Community
  Cloud.
- Add data validation and outlier handling for more robust predictions.
- Add drill-through / export options to the dashboard charts.

---

## 👤 Author

Sourav Mahata— BCA student, Roll No :2507 ,College : Midnapore College (Autonomous), IBM SkillsBuild Data Analytics with AI Academic Internship (BharatCares x AICTE), 2026.

---

