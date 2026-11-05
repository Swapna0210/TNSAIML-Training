
import pandas as pd
import numpy as np
import joblib

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Load dataset
def load_data():

    data = pd.read_csv("dataset/train.csv")

    data["date"] = pd.to_datetime(data["date"])

    # Calculate daily sales
    daily_data = data.groupby("date").agg({
        "sales": "sum",
        "onpromotion": "sum"
    }).reset_index()

    return daily_data


# Prepare dataset
def prepare_data(data):

    data["year"] = data["date"].dt.year
    data["month"] = data["date"].dt.month
    data["day"] = data["date"].dt.day
    data["day_of_week"] = data["date"].dt.dayofweek

    return data


# Train model
def train_model():

    data = load_data()
    data = prepare_data(data)

    features = [
        "year",
        "month",
        "day",
        "day_of_week",
        "onpromotion"
    ]

    X = data[features]
    y = data["sales"]

    # Split training and testing data
    split = int(len(data) * 0.8)

    X_train = X.iloc[:split]
    X_test = X.iloc[split:]

    y_train = y.iloc[:split]
    y_test = y.iloc[split:]

    # Create Linear Regression model
    model = LinearRegression()

    model.fit(X_train, y_train)

    # Make predictions
    predictions = model.predict(X_test)

    # Calculate performance
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)

    # Train final model using complete data
    model.fit(X, y)

    # Save model
    joblib.dump(model, "model.joblib")
    joblib.dump(data["date"].max(), "last_date.joblib")

    return mae, mse, rmse, r2


# Predict future sales
def predict_sales(date, promotion=0):

    model = joblib.load("model.joblib")

    date = pd.to_datetime(date)

    input_data = pd.DataFrame([{
        "year": date.year,
        "month": date.month,
        "day": date.day,
        "day_of_week": date.dayofweek,
        "onpromotion": promotion
    }])

    prediction = model.predict(input_data)

    return max(0, prediction[0])


# Get sales data
def get_sales_data():

    data = load_data()

    return data
