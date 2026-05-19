import time
import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression

st.set_page_config(
    page_title="AI Decision Support System",
    page_icon="🤖",
    layout="wide"
)

st.markdown("""
# 🤖 AI-Enhanced Data Processing & Decision Support System
### MSIT 5910 Capstone Project – Integrated Prototype Demo
This system demonstrates CSV upload, data cleaning, KPI analysis, filtering, visualization, AI prediction, and performance evaluation.
""")

st.sidebar.title("📌 Dashboard Controls")
uploaded_file = st.sidebar.file_uploader("📂 Upload CSV File", type=["csv"])

if uploaded_file:
    # Input module
    df = pd.read_csv(uploaded_file)

    st.success("Dataset uploaded successfully.")

    st.subheader("1. Raw Dataset Preview")
    st.dataframe(df.head(), use_container_width=True)

    original_rows = df.shape[0]
    original_cols = df.shape[1]

    # Data cleaning module with timing
    clean_start_time = time.time()

    cleaned_df = df.drop_duplicates()
    cleaned_df = cleaned_df.ffill().bfill()

    clean_end_time = time.time()
    preprocessing_time = clean_end_time - clean_start_time

    cleaned_rows = cleaned_df.shape[0]
    removed_duplicates = original_rows - cleaned_rows

    st.subheader("2. Data Cleaning Summary")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Original Rows", original_rows)
    col2.metric("Cleaned Rows", cleaned_rows)
    col3.metric("Columns", original_cols)
    col4.metric("Duplicates Removed", removed_duplicates)

    # Filtering module
    st.sidebar.subheader("🔎 Filters")

    filtered_df = cleaned_df.copy()

    if "Region" in filtered_df.columns:
        regions = ["All"] + sorted(filtered_df["Region"].dropna().unique().tolist())
        selected_region = st.sidebar.selectbox("Select Region", regions)

        if selected_region != "All":
            filtered_df = filtered_df[filtered_df["Region"] == selected_region]

    if "Product" in filtered_df.columns:
        products = ["All"] + sorted(filtered_df["Product"].dropna().unique().tolist())
        selected_product = st.sidebar.selectbox("Select Product", products)

        if selected_product != "All":
            filtered_df = filtered_df[filtered_df["Product"] == selected_product]

    st.subheader("3. Filtered Dataset Preview")
    st.dataframe(filtered_df.head(), use_container_width=True)

    # KPI module
    st.subheader("4. Business KPI Dashboard")

    total_revenue = filtered_df["Revenue"].sum() if "Revenue" in filtered_df.columns else 0
    total_units = filtered_df["Units_Sold"].sum() if "Units_Sold" in filtered_df.columns else 0
    avg_satisfaction = filtered_df["Customer_Satisfaction"].mean() if "Customer_Satisfaction" in filtered_df.columns else 0
    avg_sales = filtered_df["Units_Sold"].mean() if "Units_Sold" in filtered_df.columns else 0

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Total Revenue", f"${total_revenue:,.0f}")
    kpi2.metric("Total Units Sold", f"{total_units:,.0f}")
    kpi3.metric("Average Units Sold", f"{avg_sales:.2f}")
    kpi4.metric("Avg. Customer Satisfaction", f"{avg_satisfaction:.2f}")

    numeric_cols = filtered_df.select_dtypes(include=["int64", "float64"]).columns.tolist()

    ai_prediction_time = 0

    if len(numeric_cols) >= 2:
        st.subheader("5. Interactive Data Visualization")

        x_col = st.selectbox("Select X-axis column", numeric_cols)
        y_col = st.selectbox("Select Y-axis column", numeric_cols, index=1)

        fig = px.scatter(
            filtered_df,
            x=x_col,
            y=y_col,
            color="Region" if "Region" in filtered_df.columns else None,
            hover_data=filtered_df.columns,
            trendline="ols",
            title=f"Relationship Between {x_col} and {y_col}"
        )
        st.plotly_chart(fig, use_container_width=True)

        if "Date" in filtered_df.columns and "Revenue" in filtered_df.columns:
            st.subheader("6. Revenue Trend Over Time")

            trend_df = filtered_df.copy()
            trend_df["Date"] = pd.to_datetime(trend_df["Date"], errors="coerce")
            trend_df = trend_df.dropna(subset=["Date"])
            trend_df = trend_df.sort_values("Date")

            fig2 = px.line(
                trend_df,
                x="Date",
                y="Revenue",
                title="Revenue Trend Over Time"
            )
            st.plotly_chart(fig2, use_container_width=True)

        # AI prediction module with timing
        st.subheader("7. AI Prediction Module")

        ai_start_time = time.time()

        model = LinearRegression()
        model.fit(filtered_df[[x_col]], filtered_df[y_col])
        predictions = model.predict(filtered_df[[x_col]])

        ai_end_time = time.time()
        ai_prediction_time = ai_end_time - ai_start_time

        result_df = filtered_df[[x_col, y_col]].copy()
        result_df["AI Predicted Value"] = predictions

        st.dataframe(result_df.head(10), use_container_width=True)

        st.markdown("""
        ### 🧠 AI Explanation

        The AI module uses a basic **Linear Regression** model to analyze the relationship between the selected input variable and target variable.

        In this prototype:
        - The selected X-axis column acts as the input feature.
        - The selected Y-axis column acts as the target value.
        - The model learns the relationship between both variables.
        - The predicted values demonstrate how AI can support decision-making by estimating expected outcomes.
        """)

        # Performance evaluation section
        st.subheader("8. System Evaluation Metrics")

        evaluation_data = pd.DataFrame({
            "Metric": [
                "Preprocessing Latency",
                "AI Prediction Latency",
                "Rows Processed",
                "Duplicates Removed"
            ],
            "Value": [
                round(preprocessing_time, 4),
                round(ai_prediction_time, 4),
                cleaned_rows,
                removed_duplicates
            ],
            "Unit": [
                "seconds",
                "seconds",
                "rows",
                "rows"
            ]
        })

        st.dataframe(evaluation_data, use_container_width=True)

        perf_chart_data = pd.DataFrame({
            "Process": ["Data Cleaning", "AI Prediction"],
            "Execution Time (Seconds)": [
                preprocessing_time,
                ai_prediction_time
            ]
        })

        fig3 = px.bar(
            perf_chart_data,
            x="Process",
            y="Execution Time (Seconds)",
            title="System Performance Evaluation"
        )
        st.plotly_chart(fig3, use_container_width=True)

        st.info(
            "The evaluation section provides quantitative evidence of system performance by measuring "
            "data preprocessing latency and AI prediction execution time. These metrics help assess "
            "efficiency and support future optimization."
        )

    else:
        st.warning("Please upload a dataset with at least two numeric columns for visualization and AI prediction.")

else:
    st.info("Please upload a CSV file from the sidebar to start the demo.")