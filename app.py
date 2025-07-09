import streamlit as st
import pandas as pd
import plotly.express as px
import io
import numpy as np
from utils.preprocessing import preprocess_data
from utils.forecasting import load_model_and_predict, estimate_feature_importance

st.set_page_config(page_title="AI Supply Chain Forecast", layout="wide")

# --- Sidebar ---
st.sidebar.image("assets/logo.png", use_container_width=True)
st.sidebar.title("🧠 Forecast Dashboard")
st.sidebar.markdown("Upload your SKU-Store-TimeSeries CSV file")

# File uploader
uploaded_file = st.sidebar.file_uploader("📤 Choose a CSV file", type=["csv"])

# Re-run Forecast button
if st.sidebar.button("🔁 Re-run Forecast"):
    st.rerun()

# Proceed only if file uploaded
if not uploaded_file:
    st.warning("👆 Please upload a CSV file to begin.")
    st.stop()

# Load data
df = pd.read_csv(uploaded_file)
df["date"] = pd.to_datetime(df["date"])

# --- Date Filter ---
st.sidebar.markdown("### 📅 Select Date Range")
min_date, max_date = df["date"].min(), df["date"].max()
date_range = st.sidebar.date_input("Filter Data By Date", [min_date, max_date])

if len(date_range) == 2:
    df = df[(df["date"] >= pd.to_datetime(date_range[0])) & (df["date"] <= pd.to_datetime(date_range[1]))]

# Preprocess and Predict
features, original = preprocess_data(df)
preds, metrics, model, X_tensor = load_model_and_predict(features)

# Add predictions to DataFrame
original["Predicted"] = preds

# --- Tabs ---
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Forecast Overview",
    "📅 Calendar Trends",
    "🌧️ Weather vs Demand",
    "📦 Inventory Impact"
])

# --- Tab 1: Forecast Overview ---
with tab1:
    st.header("📈 Forecast Overview")
    st.subheader("🔍 Preview of Uploaded Data")
    st.dataframe(original.head())

    st.subheader("📊 Predicted vs Actual Sales (First 50 Rows)")
    fig = px.line(original.head(50), x=original.index[:50], y=["sales_qty", "Predicted"],
                  labels={"value": "Sales Quantity", "index": "Time Step"},
                  title="Predicted vs Actual")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("📈 Model Performance")
    col1, col2, col3 = st.columns(3)
    col1.metric("MAE", f"{metrics['mae']:.2f}")
    col2.metric("RMSE", f"{metrics['rmse']:.2f}")
    col3.metric("MAPE", f"{metrics['mape']:.2f}%")

    # 📥 CSV Export
    st.subheader("📥 Export Predictions")
    csv = original.to_csv(index=False).encode("utf-8")
    st.download_button("Download Predictions as CSV", csv, "predictions.csv", "text/csv")


    # 📊 Feature Importance View
st.subheader("📊 Feature Importance (Estimated)")
try:
    if features.ndim == 2:
        features_reshaped = features[:, np.newaxis, :]
    else:
        features_reshaped = features

    y_true = features_reshaped[:, -1, 0]  # last step's actual sales
    input_features = features_reshaped[:, -1, 1:]  # exclude sales_qty

    from sklearn.linear_model import LinearRegression
    imp_model = LinearRegression()
    imp_model.fit(input_features, preds)

    feat_count = input_features.shape[1]
    feat_names = [f"Feature {i+1}" for i in range(feat_count)]
    importance = imp_model.coef_

    imp_df = pd.DataFrame({"Feature": feat_names, "Importance": importance})
    fig_imp = px.bar(imp_df, x="Feature", y="Importance", title="Estimated Feature Influence on Prediction")
    st.plotly_chart(fig_imp, use_container_width=True)

except Exception as e:
    st.warning(f"⚠️ Unable to compute feature importance.\n\n{e}")



# --- Tab 2: Calendar Trends ---
with tab2:
    st.header("📅 Calendar Trends")
    daywise = original.groupby("day_of_week")[["sales_qty", "Predicted"]].mean().reset_index()
    fig_day = px.bar(daywise, x="day_of_week", y=["sales_qty", "Predicted"], barmode="group",
                     labels={"value": "Avg Sales", "day_of_week": "Day of Week"})
    st.plotly_chart(fig_day, use_container_width=True)

# --- Tab 3: Weather vs Demand ---
with tab3:
    st.header("🌧️ Weather vs Demand")
    fig_weather = px.box(original, x="rainy", y="sales_qty", points="all",
                         labels={"rainy": "Rainy Day", "sales_qty": "Sales Quantity"},
                         title="Actual Sales on Rainy vs Non-Rainy Days")
    st.plotly_chart(fig_weather, use_container_width=True)

    fig_temp = px.scatter(original, x="temperature", y="sales_qty", color="store_id",
                          title="Sales vs Temperature")
    st.plotly_chart(fig_temp, use_container_width=True)

# --- Tab 4: Inventory Impact ---
with tab4:
    st.header("📦 Inventory Effects on Sales")
    fig_inv = px.scatter(original, x="inventory_level", y="sales_qty", color="store_id",
                         title="Sales Quantity vs Inventory Level")
    st.plotly_chart(fig_inv, use_container_width=True)

    fig_delay = px.box(original, x="supplier_delay", y="sales_qty", points="all",
                       labels={"supplier_delay": "Supplier Delay"},
                       title="Sales Distribution With/Without Supplier Delay")
    st.plotly_chart(fig_delay, use_container_width=True)

