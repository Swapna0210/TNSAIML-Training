
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from backend import train_model, predict_sales, get_sales_data


# Page configuration
st.set_page_config(
    page_title="Retail Sales Forecasting",
    page_icon="📊",
    layout="wide"
)

# Title
st.title("Retail Sales Forecasting Using Machine Learning")

st.write("Analyze historical sales and predict future retail sales.")

# Train model if saved model does not exist
try:
    model = joblib.load("model.joblib")
except FileNotFoundError:
    with st.spinner("Training machine learning model..."):
        train_model()

# Load sales data
data = get_sales_data()

# Sidebar
st.sidebar.title("Navigation")

menu = st.sidebar.selectbox(
    "Select Option",
    ["Dashboard", "Sales Analysis", "Sales Prediction"]
)


# Dashboard
if menu == "Dashboard":

    st.subheader("Sales Dashboard")

    total_sales = data["sales"].sum()
    average_sales = data["sales"].mean()
    total_days = len(data)

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Sales", f"{total_sales:,.2f}")
    col2.metric("Average Daily Sales", f"{average_sales:,.2f}")
    col3.metric("Total Days", total_days)

    st.subheader("Daily Sales Trend")

    fig, ax = plt.subplots()

    ax.plot(data["date"], data["sales"])

    ax.set_xlabel("Date")
    ax.set_ylabel("Sales")
    ax.set_title("Daily Retail Sales")

    st.pyplot(fig)


# Sales Analysis
elif menu == "Sales Analysis":

    st.subheader("Sales Analysis")

    monthly_sales = data.copy()

    monthly_sales["month"] = (
        monthly_sales["date"].dt.to_period("M").astype(str)
    )

    monthly_data = monthly_sales.groupby(
        "month"
    )["sales"].sum()

    st.subheader("Monthly Sales")

    st.bar_chart(monthly_data)

    st.subheader("Sales Statistics")

    st.write(data["sales"].describe())


# Sales Prediction
elif menu == "Sales Prediction":

    st.subheader("Predict Future Retail Sales")

    last_date = joblib.load("last_date.joblib")

    st.write("Last available dataset date:", last_date.date())

    selected_date = st.date_input(
        "Select Prediction Date",
        value=(last_date + pd.Timedelta(days=1)).date(),
        min_value=(last_date + pd.Timedelta(days=1)).date()
    )

    promotion = st.number_input(
        "Number of Products on Promotion",
        min_value=0,
        value=0
    )

    if st.button("Predict Sales"):

        result = predict_sales(
            selected_date,
            promotion
        )

        st.success(
            f"Predicted Sales: {result:,.2f}"
        )
