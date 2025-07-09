import streamlit as st
import pandas as pd
import plotly.express as px
from utils.preprocessing import preprocess_data
from utils.forecasting import load_model_and_predict

st.set_page_config(page_title="AI Supply Chain Forecast", layout="wide")

# --- Sidebar ---
st.sidebar.image("assets/logo.png", use_container_width=True)
st.sidebar.title("🧠 Forecast Dashboard")
st.sidebar.markdown("Upload your SKU-Store-TimeSeries CSV file")

uploaded_file = st.sidebar.file_uploader("📤 Choose a CSV file", type=["csv"])

if not uploaded_file:
    st.warning("👆 Please upload a CSV file to begin.")
    st.stop()

# Load data
df = pd.read_csv(uploaded_file)

# Title and intro
st.title("📦 AI-Powered Demand Forecasting System")
st.markdown("Make smarter decisions using LSTM-based demand prediction with real-time calendar, weather, and inventory signals.")

# Preprocess & Predict
features, original = preprocess_data(df)
preds, metrics = load_model_and_predict(features)

# Add predictions to main dataframe
original["Predicted"] = preds

# --- Tabs ---
tab1, tab2, tab3, tab4 = st.tabs(["📈 Forecast Overview", "📅 Calendar Trends", "🌧️ Weather vs Demand", "📦 Inventory Impact"])

# Tab 1: Forecast
with tab1:
    st.header("📈 Forecast Overview")
    st.subheader("🔍 Preview of Uploaded Data")
    st.dataframe(original.head())

    st.subheader("📊 Predicted vs Actual Sales (First 50 Rows)")
    fig = px.line(original.head(50), x=original.index[:50], y=["sales_qty", "Predicted"],
                  labels={"value": "Sales Quantity", "index": "Time Step"}, title="Predicted vs Actual")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("📈 Model Performance")
    col1, col2, col3 = st.columns(3)
    col1.metric("MAE", f"{metrics['mae']:.2f}")
    col2.metric("RMSE", f"{metrics['rmse']:.2f}")
    col3.metric("MAPE", f"{metrics['mape']:.2f}%")

# Tab 2: Calendar Trends
with tab2:
    st.header("📅 Calendar Trends")
    daywise = original.groupby("day_of_week")[["sales_qty", "Predicted"]].mean().reset_index()
    fig_day = px.bar(daywise, x="day_of_week", y=["sales_qty", "Predicted"], barmode="group",
                     labels={"value": "Avg Sales", "day_of_week": "Day of Week"})
    st.plotly_chart(fig_day, use_container_width=True)

# Tab 3: Weather Trends
with tab3:
    st.header("🌧️ Weather vs Demand")
    fig_weather = px.box(original, x="rainy", y="sales_qty", points="all",
                         labels={"rainy": "Rainy Day", "sales_qty": "Sales Quantity"},
                         title="Actual Sales on Rainy vs Non-Rainy Days")
    st.plotly_chart(fig_weather, use_container_width=True)

    fig_temp = px.scatter(original, x="temperature", y="sales_qty", color="store_id",
                          title="Sales vs Temperature")
    st.plotly_chart(fig_temp, use_container_width=True)

# Tab 4: Inventory Impact
with tab4:
    st.header("📦 Inventory Effects on Sales")
    fig_inv = px.scatter(original, x="inventory_level", y="sales_qty", color="store_id",
                         title="Sales Quantity vs Inventory Level")
    st.plotly_chart(fig_inv, use_container_width=True)

    fig_delay = px.box(original, x="supplier_delay", y="sales_qty", points="all",
                       labels={"supplier_delay": "Supplier Delay"},
                       title="Sales Distribution With/Without Supplier Delay")
    st.plotly_chart(fig_delay, use_container_width=True)

