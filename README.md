# Retail Sales Forecasting Using Machine Learning

## Project Overview
Retail Sales Forecasting is a machine learning project that analyzes historical retail sales data and predicts future sales using Linear Regression. The project provides an interactive dashboard built with Streamlit to visualize sales trends, analyze historical performance, and forecast future sales.

## Objectives
- Analyze historical retail sales data.
- Identify daily and monthly sales trends.
- Apply Machine Learning for sales prediction.
- Evaluate model performance.
- Display results through an interactive dashboard.

## Technologies Used
- **Programming Language:** Python
- **Machine Learning Algorithm:** Linear Regression
- **Frontend:** Streamlit
- **Libraries:** Pandas, NumPy, Matplotlib, Scikit-learn, Joblib

## Dataset
**Dataset Name:** Store Sales – Time Series Forecasting

**Source:** Kaggle – Corporación Favorita Store Sales

Dataset link: https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data

The dataset includes historical sales, dates, store information, promotions, oil prices, transactions, and holiday information.

Note: Download the dataset from Kaggle and place the required `train.csv` file inside the `dataset` folder.

## Project Features
- Interactive sales dashboard.
- Total and average sales analysis.
- Daily sales trend visualization.
- Monthly sales analysis.
- Future sales prediction.
- Model performance evaluation using MAE, MSE, RMSE, and R² Score.

## Project Structure
```text
Retail_Sales_Forecasting/
│
├── app.py
├── backend.py
├── model.joblib
├── last_date.joblib
├── requirements.txt
├── README.md
├── .gitignore
│
├── dataset/
│   └── train.csv
│
└── images/
```

## Installation and Execution

### 1. Clone the repository
```bash
git clone https://github.com/Swapna0210/Retail_Sales_Forecasting.git
```

### 2. Open the project folder
```bash
cd Retail_Sales_Forecasting
```

### 3. Install required libraries
```bash
pip install -r requirements.txt
```

### 4. Download the dataset
Download `train.csv` from Kaggle and place it inside the `dataset` folder.

### 5. Train the model
```bash
python -c "from backend import train_model; print(train_model())"
```

### 6. Run the application
```bash
streamlit run app.py
```

## Machine Learning Algorithm
**Linear Regression**

Linear Regression is a supervised machine learning algorithm that learns the relationship between input features and sales values to predict future sales.

## Model Evaluation
The model performance is evaluated using:
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R² Score

## Author
**Swapna Dharuvemula**

GitHub: https://github.com/Swapna0210
