import pandas as pd
from sklearn.linear_model import LinearRegression

def clean_data(df):
    return df.drop_duplicates().ffill().bfill()

def calculate_kpis(df):
    return {
        "total_revenue": df["Revenue"].sum(),
        "total_units": df["Units_Sold"].sum(),
        "average_units": df["Units_Sold"].mean(),
        "average_satisfaction": df["Customer_Satisfaction"].mean()
    }

def test_clean_data_removes_duplicates_and_missing_values():
    df = pd.DataFrame({
        "Revenue": [1000, 1000, None],
        "Units_Sold": [10, 10, 20],
        "Customer_Satisfaction": [4.5, 4.5, None]
    })

    cleaned = clean_data(df)

    assert cleaned.duplicated().sum() == 0
    assert cleaned.isnull().sum().sum() == 0

def test_kpi_calculation_returns_expected_values():
    df = pd.DataFrame({
        "Revenue": [1000, 2000],
        "Units_Sold": [10, 20],
        "Customer_Satisfaction": [4.0, 5.0]
    })

    kpis = calculate_kpis(df)

    assert kpis["total_revenue"] == 3000
    assert kpis["total_units"] == 30
    assert kpis["average_units"] == 15
    assert kpis["average_satisfaction"] == 4.5

def test_ai_prediction_generates_output():
    df = pd.DataFrame({
        "Marketing_Spend": [1000, 2000, 3000],
        "Revenue": [5000, 8000, 11000]
    })

    model = LinearRegression()
    model.fit(df[["Marketing_Spend"]], df["Revenue"])
    prediction = model.predict(df[["Marketing_Spend"]])

    assert len(prediction) == len(df)