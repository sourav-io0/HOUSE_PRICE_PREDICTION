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

NAVY = "#1F4E79"
RED = "#B23A48"
PALETTE = [NAVY, RED, "#4C8DAE", "#E8A33D"]


# ==========================================
# DATA LOADING & MODEL TRAINING (cached)
# ==========================================

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


def fmt_money(x: float) -> str:
    """Compact currency formatting, e.g. 4.77M / 950K."""
    if abs(x) >= 1_00_00_000:  # 1 crore+
        return f"₹{x/1_00_00_000:.2f}Cr"
    if abs(x) >= 1_000_000:
        return f"₹{x/1_000_000:.2f}M"
    if abs(x) >= 1_000:
        return f"₹{x/1_000:.0f}K"
    return f"₹{x:,.0f}"


# ==========================================
# PAGE CONFIGURATION + LIGHT STYLING
# ==========================================

st.set_page_config(
    page_title="Housing Price Analysis Dashboard",
    page_icon="🏠",
    layout="wide",
)

st.markdown(
    """
    <style>
        .block-container { padding-top: 1.5rem; }
        div[data-testid="stMetric"] {
            background-color: #ffffff;
            border: 1px solid #e6e6e6;
            border-radius: 10px;
            padding: 12px 10px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.06);
        }
        div[data-testid="stMetricValue"] { color: #1F4E79; }
        .dash-title {
            background-color: #eaf1f8;
            border-radius: 10px;
            padding: 14px 20px;
            border-left: 6px solid #1F4E79;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


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
# HEADER
# ==========================================

st.markdown(
    '<div class="dash-title"><h2 style="margin:0;color:#1F4E79;">'
    '🏠 Real Estate — Housing Price Analysis Dashboard</h2></div>',
    unsafe_allow_html=True,
)
st.write("")


# ==========================================
# SIDEBAR FILTERS (slicers)
# ==========================================

st.sidebar.header("🔎 Filters")

if st.sidebar.button("Clear all filters"):
    for key in ["f_bedrooms", "f_bathrooms", "f_parking", "f_area", "f_price"]:
        st.session_state.pop(key, None)
    st.rerun()

bedroom_opts = sorted(df["bedrooms"].unique())
bathroom_opts = sorted(df["bathrooms"].unique())
parking_opts = sorted(df["parking"].unique())

sel_bedrooms = st.sidebar.multiselect("Bedrooms", bedroom_opts, default=bedroom_opts, key="f_bedrooms")
sel_bathrooms = st.sidebar.multiselect("Bathrooms", bathroom_opts, default=bathroom_opts, key="f_bathrooms")
sel_parking = st.sidebar.multiselect("Parking spaces", parking_opts, default=parking_opts, key="f_parking")

area_min, area_max = int(df["area"].min()), int(df["area"].max())
sel_area = st.sidebar.slider("Area (sq ft)", area_min, area_max, (area_min, area_max), key="f_area")

price_min, price_max = int(df["price"].min()), int(df["price"].max())
sel_price = st.sidebar.slider(
    "Price range (₹)", price_min, price_max, (price_min, price_max),
    format="₹%d", key="f_price",
)

filtered = df[
    df["bedrooms"].isin(sel_bedrooms)
    & df["bathrooms"].isin(sel_bathrooms)
    & df["parking"].isin(sel_parking)
    & df["area"].between(*sel_area)
    & df["price"].between(*sel_price)
]

if filtered.empty:
    st.warning("No properties match the current filters. Try widening them from the sidebar.")
    st.stop()


# ==========================================
# TABS
# ==========================================

tab_dash, tab_predict, tab_performance = st.tabs(
    ["📊 Dashboard", "🔮 Predict", "🎯 Model Performance"]
)


# ------------------------------------------
# TAB 1: DASHBOARD (Power-BI style overview)
# ------------------------------------------
with tab_dash:
    st.caption(f"Showing **{len(filtered)}** of {len(df)} properties based on current filters")

    # ---- KPI cards ----
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Average Price", fmt_money(filtered["price"].mean()))
    k2.metric("Avg Bathrooms", f"{filtered['bathrooms'].mean():.2f}")
    k3.metric("Median Bedrooms", f"{filtered['bedrooms'].median():.0f}")
    k4.metric("Average Parking", f"{filtered['parking'].mean():.2f}")
    k5.metric("Average Area", f"{filtered['area'].mean():,.0f} sqft")

    st.write("")

    # ---- Row 1: bar charts by bedrooms / bathrooms / parking ----
    r1c1, r1c2, r1c3 = st.columns(3)

    with r1c1:
        avg_by_bed = filtered.groupby("bedrooms", as_index=False)["price"].mean()
        fig = px.bar(
            avg_by_bed, x="bedrooms", y="price", text_auto=".2s",
            title="Avg Price by Bedrooms", color_discrete_sequence=[NAVY],
        )
        fig.update_layout(height=300, margin=dict(t=40, b=10, l=10, r=10))
        st.plotly_chart(fig, use_container_width=True)

    with r1c2:
        avg_by_bath = filtered.groupby("bathrooms", as_index=False)["price"].mean()
        fig = px.bar(
            avg_by_bath, x="bathrooms", y="price", text_auto=".2s",
            title="Avg Price by Bathrooms", color_discrete_sequence=[RED],
        )
        fig.update_layout(height=300, margin=dict(t=40, b=10, l=10, r=10))
        st.plotly_chart(fig, use_container_width=True)

    with r1c3:
        avg_by_park = filtered.groupby("parking", as_index=False)["price"].mean()
        fig = px.bar(
            avg_by_park, x="parking", y="price", text_auto=".2s",
            title="Avg Price by Parking", color_discrete_sequence=["#4C8DAE"],
        )
        fig.update_layout(height=300, margin=dict(t=40, b=10, l=10, r=10))
        st.plotly_chart(fig, use_container_width=True)

    # ---- Row 2: scatter, distribution, donut ----
    r2c1, r2c2, r2c3 = st.columns(3)

    with r2c1:
        fig = px.scatter(
            filtered, x="area", y="price", color="bedrooms",
            title="House Area and Price", opacity=0.7,
            color_continuous_scale="Blues",
        )
        fig.update_layout(height=320, margin=dict(t=40, b=10, l=10, r=10))
        st.plotly_chart(fig, use_container_width=True)

    with r2c2:
        fig = px.histogram(
            filtered, x="price", nbins=30, title="Price Distribution",
            color_discrete_sequence=[NAVY],
        )
        fig.update_layout(height=320, margin=dict(t=40, b=10, l=10, r=10))
        st.plotly_chart(fig, use_container_width=True)

    with r2c3:
        bed_counts = filtered["bedrooms"].value_counts().reset_index()
        bed_counts.columns = ["bedrooms", "count"]
        fig = px.pie(
            bed_counts, names="bedrooms", values="count", hole=0.55,
            title="Bedrooms Split", color_discrete_sequence=PALETTE,
        )
        fig.update_layout(height=320, margin=dict(t=40, b=10, l=10, r=10))
        st.plotly_chart(fig, use_container_width=True)

    # ---- Row 3: parking pie + stacked bar (bedrooms x bathrooms) ----
    r3c1, r3c2 = st.columns(2)

    with r3c1:
        park_counts = filtered["parking"].value_counts().reset_index()
        park_counts.columns = ["parking", "count"]
        fig = px.pie(
            park_counts, names="parking", values="count",
            title="Avg Price Mix by Parking Spaces", color_discrete_sequence=PALETTE,
        )
        fig.update_layout(height=340, margin=dict(t=40, b=10, l=10, r=10))
        st.plotly_chart(fig, use_container_width=True)

    with r3c2:
        combo = filtered.groupby(["bedrooms", "bathrooms"], as_index=False)["price"].mean()
        fig = px.bar(
            combo, x="bedrooms", y="price", color="bathrooms",
            title="Avg Price by Bedrooms and Bathrooms",
            barmode="stack", color_continuous_scale="Blues",
        )
        fig.update_layout(height=340, margin=dict(t=40, b=10, l=10, r=10))
        st.plotly_chart(fig, use_container_width=True)


# ------------------------------------------
# TAB 2: PREDICT
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

        fig_price_pos = px.histogram(df, x=TARGET, nbins=40, title="Where your estimate falls in the price distribution")
        fig_price_pos.add_vline(
            x=prediction, line_dash="dash", line_color="red",
            annotation_text="Your estimate", annotation_position="top right",
        )
        st.plotly_chart(fig_price_pos, use_container_width=True)
    else:
        st.caption("Fill in the details above and click **Predict** to see comparison charts.")


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