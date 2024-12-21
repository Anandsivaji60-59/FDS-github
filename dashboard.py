# -- coding: utf-8 --
import streamlit as st
import pandas as pd
import plotly.express as px
import json

# Load datasets
raw_data_path = "https://raw.githubusercontent.com/Anandsivaji60-59/FDS-github/refs/heads/main/raw_data.csv"
preprocessed_data_path = "https://raw.githubusercontent.com/Anandsivaji60-59/FDS-github/refs/heads/main/preprocessed_data.csv"
# Read raw and preprocessed data
try:
    df_raw = pd.read_csv(raw_data_path)
    df_preprocessed = pd.read_csv(preprocessed_data_path)
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# Title and Sidebar
st.title("Predicting the Success of Personalized Movie Recommendations")
st.sidebar.title("Options")

# Section 1: Data Distribution Visualization
st.header("1. Data Distribution Visualization")

data_selection = st.sidebar.radio("Choose data to display:", ("Raw Data", "Preprocessed Data"))
selected_data = df_raw if data_selection == "Raw Data" else df_preprocessed

st.write(f"### {data_selection} Distribution")
for col in selected_data.select_dtypes(include=["object", "int64", "float64"]).columns:
    fig = px.histogram(selected_data, x=col, title=f"Distribution of {col}")
    st.plotly_chart(fig, use_container_width=True)

# Section 2: Model Performance Metrics
st.header("2. Model Performance Metrics")

# Classification report for raw data
raw_report_json = """
{
    "0": {"precision": 0.96, "recall": 1.00, "f1-score": 0.98, "support": 520},
    "1": {"precision": 1.00, "recall": 0.93, "f1-score": 0.96, "support": 320},
    "accuracy": 0.9714,
    "macro avg": {"precision": 0.98, "recall": 0.96, "f1-score": 0.97, "support": 840},
    "weighted avg": {"precision": 0.97, "recall": 0.97, "f1-score": 0.97, "support": 840}
}
"""

# Classification report for preprocessed data
preprocessed_report_json = """
{
    "0": {"precision": 0.76, "recall": 0.99, "f1-score": 0.86, "support": 520},
    "1": {"precision": 0.96, "recall": 0.50, "f1-score": 0.65, "support": 320},
    "accuracy": 0.80,
    "macro avg": {"precision": 0.86, "recall": 0.74, "f1-score": 0.76, "support": 840},
    "weighted avg": {"precision": 0.84, "recall": 0.80, "f1-score": 0.78, "support": 840}
}
"""

# Load JSON reports from strings
try:
    raw_report = json.loads(raw_report_json)
    preprocessed_report = json.loads(preprocessed_report_json)
except Exception as e:
    st.error(f"Error parsing classification reports: {e}")
    st.stop()

# Display metrics for raw data
st.write("### Training Metrics (Raw Data)")
raw_metrics = ["precision", "recall", "f1-score"]
raw_metric_data = {metric: [raw_report[str(i)][metric] for i in range(2)] for metric in raw_metrics}
raw_metric_data["Class"] = ["Class 0", "Class 1"]

raw_df_metrics = pd.DataFrame(raw_metric_data)
fig = px.bar(raw_df_metrics, x="Class", y=raw_metrics, barmode="group", title="Classification Metrics for Raw Data")
st.plotly_chart(fig, use_container_width=True)

# Display metrics for preprocessed data
st.write("### Training Metrics (Preprocessed Data)")
preprocessed_metrics = ["precision", "recall", "f1-score"]
preprocessed_metric_data = {metric: [preprocessed_report[str(i)][metric] for i in range(2)] for metric in preprocessed_metrics}
preprocessed_metric_data["Class"] = ["Class 0", "Class 1"]

preprocessed_df_metrics = pd.DataFrame(preprocessed_metric_data)
fig = px.bar(preprocessed_df_metrics, x="Class", y=preprocessed_metrics, barmode="group", title="Classification Metrics for Preprocessed Data")
st.plotly_chart(fig, use_container_width=True)

# Section 3: Model Comparison
st.header("3. Model Comparison")

# Model comparison table
comparison_data = {
    "Model": ["Raw Data", "Preprocessed Data"],
    "Accuracy": [0.9714, 0.80],
    "Precision": [0.96, 0.96],
    "Recall": [1.00, 0.50],
    "F1-Score": [0.98, 0.65],
    "ROC-AUC": [0.97, 0.80]  # Placeholder, you can add actual values for ROC-AUC
}
df_comparison = pd.DataFrame(comparison_data)

# Visualization of model comparison
fig = px.bar(df_comparison, x="Model", y=["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"],
             barmode="group", title="Performance Comparison Between Raw and Preprocessed Data")
st.plotly_chart(fig, use_container_width=True)

# Display comparison table
st.write("### Comparison Table")
st.dataframe(df_comparison)

# Section 4: Insights
st.header("4. Insights")
st.markdown("""
- *Raw Data*: The model achieves a high accuracy of 97.14%, with near-perfect recall for Class 0 (1.00) and very good performance for Class 1 (0.93 recall).
- *Preprocessed Data*: After preprocessing, the model's accuracy drops to 80%. The recall for Class 1 significantly drops to 0.50, indicating the preprocessing may have compromised the model's ability to identify positive cases.
- *Significance*: While preprocessing typically helps in improving models, it seems to have decreased recall for Class 1 in this case. It is important to balance the performance between both classes.
- Use the charts and tables above to analyze and compare performance metrics interactively.
""")
