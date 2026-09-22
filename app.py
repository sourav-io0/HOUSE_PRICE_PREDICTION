import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

DATA_PATH = "HOUSE_PRICE_PREDICTION.csv"
FEATURES = ["area", "bedrooms", "bathrooms", "parking"]
TARGET = "price"


# ==========================================
# DATA LOADING & MODEL TRAINING (cached)
# ==========================================
# Cached so the CSV is read and the model trained once, instead of on
# every widget interaction (Streamlit reruns the whole script each time).

@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_resource
def train_model(df: pd.DataFrame):
    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    results = pd.DataFrame({
        "actual": y_test.reset_index(drop=True),
        "predicted": y_pred,
    })
    results["residual"] = results["actual"] - results["predicted"]

    coefs = pd.DataFrame({
        "feature": FEATURES,
        "coefficient": model.coef_,
    }).sort_values("coefficient", key=abs, ascending=False)

    return model, mae, rmse, r2, results, coefs, X_train.shape, X_test.shape


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="wide",
)

st.title("🏠 House Price Prediction")
st.write("Estimate a house's price and explore how the model arrived at it.")


# ==========================================
# LOAD DATA + TRAIN MODEL (with error handling)
# ==========================================

try:
    df = load_data(DATA_PATH)
except FileNotFoundError:
    st.error(
        f"Couldn't find **{DATA_PATH}**. Make sure the CSV is in the same "
        "folder as this app (or update `DATA_PATH`) and reload the page."
    )
    st.stop()

missing_cols = [c for c in FEATURES + [TARGET] if c not in df.columns]
if missing_cols:
    st.error(f"The dataset is missing required column(s): {', '.join(missing_cols)}")
    st.stop()

model, mae, rmse, r2, results, coefs, train_shape, test_shape = train_model(df)


# ==========================================
# TABS
# ==========================================

tab_predict, tab_explore, tab_performance = st.tabs(
    ["🔮 Predict", "📈 Explore Data", "🎯 Model Performance"]
)


# ------------------------------------------
# TAB 1: PREDICT
# ------------------------------------------
with tab_predict:
    st.header("Enter House Details")

    col1, col2 = st.columns(2)
    with col1:
        area = st.number_input("Area (sq ft)", min_value=100, max_value=20000, value=1200, step=100)
        bedrooms = st.number_input("Number of Bedrooms", min_value=1, max_value=10, value=3, step=1)
    with col2:
        bathrooms = st.number_input("Number of Bathrooms", min_value=1, max_value=10, value=2, step=1)
        parking = st.number_input("Number of Parking Spaces", min_value=0, max_value=10, value=1, step=1)

    if bathrooms > bedrooms + 2:
        st.warning("That's a lot of bathrooms relative to bedrooms — double check the numbers.")

    if st.button("🔮 Predict House Price", type="primary"):
        input_data = pd.DataFrame({
            "area": [area], "bedrooms": [bedrooms],
            "bathrooms": [bathrooms], "parking": [parking],
        })

        prediction = max(model.predict(input_data)[0], 0)

        st.success("Prediction completed!")
        st.subheader("🏠 Estimated House Price")
        st.write(f"## ₹{prediction:,.0f}")

        low, high = max(prediction - mae, 0), prediction + mae
        st.caption(f"Typical range given model error: ₹{low:,.0f} – ₹{high:,.0f}")
        st.info("This is an estimate from a linear regression model, not a formal valuation.")

        st.markdown("---")
        st.subheader("How your inputs compare to the dataset")

        # Radar-style comparison: your house's feature values vs dataset averages,
        # each normalized to 0-1 so wildly different scales (area vs bedrooms) are comparable.
        norm_rows = []
        your_values = {"area": area, "bedrooms": bedrooms, "bathrooms": bathrooms, "parking": parking}
        for feat in FEATURES:
            fmin, fmax = df[feat].min(), df[feat].max()
            span = (fmax - fmin) or 1
            norm_rows.append({
                "feature": feat,
                "You": (your_values[feat] - fmin) / span,
                "Dataset average": (df[feat].mean() - fmin) / span,
            })
        norm_df = pd.DataFrame(norm_rows)

        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=norm_df["You"], theta=norm_df["feature"], fill="toself", name="Your house"
        ))
        fig_radar.add_trace(go.Scatterpolar(
            r=norm_df["Dataset average"], theta=norm_df["feature"], fill="toself", name="Dataset average"
        ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=True, height=400,
        )
        st.plotly_chart(fig_radar, use_container_width=True)

        # Where does this prediction fall in the overall price distribution?
        fig_price_pos = px.histogram(df, x=TARGET, nbins=40, title="Where your estimate falls in the price distribution")
        fig_price_pos.add_vline(
            x=prediction, line_dash="dash", line_color="red",
            annotation_text="Your estimate", annotation_position="top right",
        )
        st.plotly_chart(fig_price_pos, use_container_width=True)
    else:
        st.caption("Fill in the details above and click **Predict** to see comparison charts.")


# ------------------------------------------
# TAB 2: EXPLORE DATA
# ------------------------------------------
with tab_explore:
    st.header("Dataset Overview")
    st.write(f"**{df.shape[0]} rows × {df.shape[1]} columns**")
    st.dataframe(df.head(10), use_container_width=True)

    st.subheader("Price distribution")
    fig_hist = px.histogram(df, x=TARGET, nbins=40, marginal="box")
    st.plotly_chart(fig_hist, use_container_width=True)

    st.subheader("Feature relationships with price")
    feat_choice = st.selectbox("Choose a feature to plot against price", FEATURES)
    fig_scatter = px.scatter(
        df, x=feat_choice, y=TARGET, trendline="ols",
        opacity=0.6, title=f"{feat_choice} vs {TARGET}",
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    st.subheader("Correlation heatmap")
    corr = df[FEATURES + [TARGET]].corr()
    fig_corr = px.imshow(
        corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
        title="Correlation between features and price",
    )
    st.plotly_chart(fig_corr, use_container_width=True)


# ------------------------------------------
# TAB 3: MODEL PERFORMANCE
# ------------------------------------------
with tab_performance:
    st.header("Model Performance")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("MAE", f"₹{mae:,.0f}")
    m2.metric("RMSE", f"₹{rmse:,.0f}")
    m3.metric("R² Score", f"{r2:.3f}")
    m4.metric("Test set size", f"{test_shape[0]} rows")

    st.subheader("Actual vs. Predicted prices (test set)")
    fig_avp = px.scatter(
        results, x="actual", y="predicted", opacity=0.6,
        title="Actual vs. Predicted — points closer to the diagonal are better",
    )
    min_v, max_v = results[["actual", "predicted"]].min().min(), results[["actual", "predicted"]].max().max()
    fig_avp.add_trace(go.Scatter(
        x=[min_v, max_v], y=[min_v, max_v], mode="lines",
        line=dict(dash="dash", color="gray"), name="Perfect prediction",
    ))
    st.plotly_chart(fig_avp, use_container_width=True)

    st.subheader("Residuals")
    col_a, col_b = st.columns(2)
    with col_a:
        fig_resid = px.scatter(
            results, x="predicted", y="residual", opacity=0.6,
            title="Residuals vs. Predicted price",
        )
        fig_resid.add_hline(y=0, line_dash="dash", line_color="red")
        st.plotly_chart(fig_resid, use_container_width=True)
    with col_b:
        fig_resid_hist = px.histogram(results, x="residual", nbins=30, title="Distribution of residuals")
        st.plotly_chart(fig_resid_hist, use_container_width=True)

    st.subheader("Feature importance (regression coefficients)")
    fig_coef = px.bar(
        coefs, x="coefficient", y="feature", orientation="h",
        color="coefficient", color_continuous_scale="RdBu",
        title="Effect of each feature on predicted price",
    )
    st.plotly_chart(fig_coef, use_container_width=True)
    st.caption(
        "A positive coefficient means increasing that feature tends to increase the "
        "predicted price (holding other features constant), and vice versa."
    )