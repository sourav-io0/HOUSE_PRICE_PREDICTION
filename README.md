# 🏠 House Price Prediction

A machine learning web application that predicts house prices based on key
property features, built with **Python**, **scikit-learn**, and **Streamlit**.

---

## 📌 Project Overview

This project uses a **Linear Regression** model trained on a housing dataset
to estimate the price of a house given its area, number of bedrooms,
bathrooms, and parking spaces. The model is served through an interactive
Streamlit web application that allows users to enter house details and view
the predicted price, along with supporting charts and model performance
metrics.

---

## 🎯 Objectives

- Build a regression model to predict house prices from structured data.
- Evaluate the model's accuracy using standard regression metrics.
- Provide an interactive, user-friendly interface for real-time predictions.
- Visualize the dataset and model performance through charts and graphs.

---

## 🗂️ Dataset

- **File:** `HOUSE_PRICE_PREDICTION.csv`
**Resource** https://www.kaggle.com/datasets/yasserh/housing-prices-dataset
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
6. **Deployment** – The trained model is served through a Streamlit web
   app with an interactive prediction interface.

---

## 📊 Application Features

The web app is organized into three tabs:

- **🔮 Predict** — Enter house details to get an estimated price, view how
  the input compares to the dataset average, and see where the prediction
  falls within the overall price distribution.
- **📈 Explore Data** — View the dataset, price distribution, feature vs.
  price scatter plots, and a correlation heatmap.
- **🎯 Model Performance** — View evaluation metrics, an actual vs.
  predicted price plot, residual plots, and feature importance
  (regression coefficients).

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
**Model Performance** tab of the application.

---

## 🔮 Future Enhancements

- Add more features (location, house age, furnishing status, etc.).
- Compare multiple algorithms (Random Forest, Gradient Boosting) for
  improved accuracy.
- Deploy the application on a public platform such as Streamlit Community
  Cloud.
- Add data validation and outlier handling for more robust predictions.

---

## 👤 Author

Sourav Mahata— BCA student, Roll No :2507 ,College : Midnapore College (Autonomous), IBM SkillsBuild Data Analytics with AI Academic Internship (BharatCares x AICTE), 2026.

---

