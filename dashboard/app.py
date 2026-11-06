
from pathlib import Path

import pandas as pd
import streamlit as st
import plotly.express as px


# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Satellite Telemetry Anomaly Detection",
    page_icon="🛰️",
    layout="wide"
)


# --------------------------------------------------
# 2. PROJECT PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "dataset.csv"
PREDICTIONS_PATH = BASE_DIR / "results" / "prediction_results.csv"


# --------------------------------------------------
# 3. LOAD DATA
# --------------------------------------------------

@st.cache_data
def load_data():
    dataset = pd.read_csv(DATA_PATH)
    comparison = pd.read_csv(COMPARISON_PATH)
    predictions = pd.read_csv(PREDICTIONS_PATH)

    return dataset, comparison, predictions


try:
    dataset, comparison, predictions = load_data()

except FileNotFoundError as error:
    st.error(f"Required project file not found: {error}")
    st.stop()

except Exception as error:
    st.error(f"Unable to load project data: {error}")
    st.stop()


# --------------------------------------------------
# 4. DASHBOARD HEADER
# --------------------------------------------------

st.title("🛰️ Satellite Telemetry Anomaly Detection")

st.markdown(
    """
    **OPS-SAT Benchmark | Machine Learning Based Anomaly Detection**

    This dashboard analyzes satellite telemetry data and compares
    supervised and unsupervised anomaly detection models.
    """
)

st.divider()


# --------------------------------------------------
# 5. TELEMETRY OVERVIEW
# --------------------------------------------------

st.header("Telemetry Dataset Overview")

total_samples = len(dataset)

nominal_samples = int((dataset["anomaly"] == 0).sum())
anomalous_samples = int((dataset["anomaly"] == 1).sum())

train_samples = int((dataset["train"] == 1).sum())
test_samples = int((dataset["train"] == 0).sum())


col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Samples", f"{total_samples:,}")
col2.metric("Nominal Samples", f"{nominal_samples:,}")
col3.metric("Anomalous Samples", f"{anomalous_samples:,}")
col4.metric("Test Samples", f"{test_samples:,}")


st.divider()


# --------------------------------------------------
# 6. NOMINAL VS ANOMALOUS DISTRIBUTION
# --------------------------------------------------

st.header("Telemetry Anomaly Distribution")

distribution = pd.DataFrame({
    "Category": ["Nominal", "Anomalous"],
    "Count": [nominal_samples, anomalous_samples]
})

fig_distribution = px.pie(
    distribution,
    names="Category",
    values="Count",
    title="Nominal vs Anomalous Telemetry Samples",
    hole=0.4
)

st.plotly_chart(fig_distribution, width="stretch")


# --------------------------------------------------
# 7. TRAINING VS TESTING DISTRIBUTION
# --------------------------------------------------

st.header("Training and Testing Dataset")

split_data = pd.DataFrame({
    "Dataset": ["Training", "Testing"],
    "Samples": [train_samples, test_samples]
})

fig_split = px.bar(
    split_data,
    x="Dataset",
    y="Samples",
    color="Dataset",
    title="Training vs Testing Samples",
    text="Samples"
)

st.plotly_chart(fig_split, width="stretch")


# --------------------------------------------------
# 8. MODEL PERFORMANCE COMPARISON
# --------------------------------------------------

st.header("Machine Learning Model Comparison")

st.dataframe(
    comparison,
    width="stretch",
    hide_index=True
)

metric_options = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1-score",
    "MCC",
    "ROC-AUC",
    "Average Precision (AP)"
]

selected_metric = st.selectbox(
    "Select an evaluation metric",
    metric_options
)

selected_row = comparison[comparison["Metric"] == selected_metric]

metric_data = pd.DataFrame({
    "Model": ["Random Forest", "Isolation Forest", "XGBoost"],
    "Score": [
        selected_row["Random Forest"].iloc[0],
        selected_row["Isolation Forest"].iloc[0],
        selected_row["XGBoost"].iloc[0]
    ]
})

fig_comparison = px.bar(
    metric_data,
    x="Model",
    y="Score",
    color="Model",
    title=f"{selected_metric} - Model Comparison",
    text_auto=".4f"
)

fig_comparison.update_yaxes(range=[0, 1.1])

st.plotly_chart(fig_comparison, width="stretch")


# --------------------------------------------------
# 9. FOOTER
# --------------------------------------------------



    # Confusion Matrix Comparison

st.divider()

st.header("Confusion Matrix Comparison")

st.markdown(
    "A confusion matrix evaluates how accurately each model "
    "identifies nominal and anomalous telemetry samples."
)

confusion_data = {
    "Random Forest": [[414, 2], [15, 98]],
    "Isolation Forest": [[335, 81], [82, 31]],
    "XGBoost": [[406, 10], [11, 102]]
}

model_choice = st.selectbox(
    "Select a model to view its confusion matrix",
    ["Random Forest", "Isolation Forest", "XGBoost"]
)

selected_matrix = confusion_data[model_choice]

fig_cm = px.imshow(
    selected_matrix,
    text_auto=True,
    labels=dict(
        x="Predicted Class",
        y="Actual Class",
        color="Number of Samples"
    ),
    x=["Nominal", "Anomalous"],
    y=["Nominal", "Anomalous"],
    color_continuous_scale="Blues",
    title=f"{model_choice} - Confusion Matrix"
)

st.plotly_chart(fig_cm, width="stretch")

# --------------------------------------------------
# 10. ACTUAL VS PREDICTED ANOMALY ANALYSIS
# --------------------------------------------------

st.divider()

st.header("Actual vs Predicted Anomaly Analysis")

st.markdown(
    "Compare the actual anomaly labels with the predictions "
    "generated by the three machine learning models."
)

# Select model
prediction_columns = {
    "Random Forest": "Random_Forest_Predicted",
    "Isolation Forest": "Isolation_Forest_Predicted",
    "XGBoost": "XGBoost_Predicted"
}

selected_model = st.selectbox(
    "Select a model for prediction analysis",
    list(prediction_columns.keys())
)

prediction_column = prediction_columns[selected_model]

actual = predictions["Actual"]
predicted = predictions[prediction_column]

# Calculate prediction categories
true_positive = int(((actual == 1) & (predicted == 1)).sum())
true_negative = int(((actual == 0) & (predicted == 0)).sum())
false_positive = int(((actual == 0) & (predicted == 1)).sum())
false_negative = int(((actual == 1) & (predicted == 0)).sum())

# Display results
col1, col2, col3, col4 = st.columns(4)

col1.metric("True Positives", true_positive)
col2.metric("True Negatives", true_negative)
col3.metric("False Positives", false_positive)
col4.metric("False Negatives", false_negative)

# Actual vs Predicted distribution
analysis_data = pd.DataFrame({
    "Category": ["Nominal", "Anomalous"],
    "Actual Samples": [
        int((actual == 0).sum()),
        int((actual == 1).sum())
    ],
    "Predicted Samples": [
        int((predicted == 0).sum()),
        int((predicted == 1).sum())
    ]
})

analysis_long = analysis_data.melt(
    id_vars="Category",
    var_name="Type",
    value_name="Samples"
)

fig_analysis = px.bar(
    analysis_long,
    x="Category",
    y="Samples",
    color="Type",
    barmode="group",
    text="Samples",
    title=f"{selected_model} - Actual vs Predicted Distribution"
)

st.plotly_chart(fig_analysis, width="stretch")



# --------------------------------------------------
# 11. ANOMALY SCORE ANALYSIS
# --------------------------------------------------

st.divider()

st.header("Anomaly Score Analysis")

st.markdown(
    """
    Anomaly scores indicate how strongly a model considers
    a telemetry sample to be unusual.

    Higher scores indicate greater anomaly likelihood.
    Note that Isolation Forest scores are not probabilities.
    """
)

# Map models to their score columns
score_columns = {
    "Random Forest": "Random_Forest_Anomaly_Score",
    "Isolation Forest": "Isolation_Forest_Anomaly_Score",
    "XGBoost": "XGBoost_Anomaly_Score"
}

score_model = st.selectbox(
    "Select a model for anomaly score analysis",
    list(score_columns.keys()),
    key="anomaly_score_model"
)

score_column = score_columns[score_model]

# Prepare score data
score_data = predictions[["Actual", score_column]].copy()

score_data["Actual Class"] = score_data["Actual"].map({
    0: "Nominal",
    1: "Anomalous"
})

score_data = score_data.rename(
    columns={score_column: "Anomaly Score"}
)

# Display score distribution
fig_scores = px.histogram(
    score_data,
    x="Anomaly Score",
    color="Actual Class",
    marginal="box",
    barmode="overlay",
    opacity=0.65,
    nbins=40,
    title=f"{score_model} - Anomaly Score Distribution",
    labels={
        "Anomaly Score": "Anomaly Score",
        "count": "Number of Samples"
    }
)

st.plotly_chart(fig_scores, width="stretch")

# Display score statistics
st.subheader("Anomaly Score Statistics")

score_summary = score_data.groupby(
    "Actual Class",
    observed=True
)["Anomaly Score"].agg(
    ["count", "mean", "min", "max"]
).reset_index()

st.dataframe(
    score_summary,
    width="stretch",
    hide_index=True
)


# --------------------------------------------------
# 12. FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "Satellite Telemetry Anomaly Detection | "
    "B.Tech Capstone Project"
)

