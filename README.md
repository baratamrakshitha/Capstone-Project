# Satellite Telemetry Anomaly Detection Using Machine Learning

## Project Overview

This project focuses on detecting anomalies in satellite telemetry data using machine learning techniques. It uses the OPS-SAT benchmark dataset to train, evaluate, and compare different anomaly detection models.

## Objectives

- Explore and understand satellite telemetry data.
- Perform data analysis and preprocessing.
- Implement supervised and unsupervised machine learning models.
- Compare model performance using evaluation metrics.
- Generate anomaly predictions.
- Develop an interactive Streamlit dashboard.

## Machine Learning Models

The following models are implemented:

1. Random Forest – Supervised learning.
2. Isolation Forest – Unsupervised anomaly detection.
3. XGBoost – Supervised learning.

## Technologies Used

- Python
- Pandas and NumPy
- Scikit-learn
- PyOD
- XGBoost
- Matplotlib and Seaborn
- Plotly
- Streamlit

## Project Structure

```text
Capstone-Project/
├── dashboard/
├── data/
├── models/
├── notebooks/
├── reference/
├── report/
├── results/
└── src/
```

## Results

The models are evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Matthews Correlation Coefficient (MCC)
- ROC-AUC
- Average Precision / PR-AUC (as reported)

Confusion matrices, performance metrics, and model comparison graphs are available in the `results/` folder.

## Running the Dashboard

Install the required Python libraries and run:
```bash
.\.venv\Scripts\python.exe -m streamlit run dashboard/app.py
```

## Dataset Attribution

This project uses the OPS-SAT anomaly detection dataset (OPSSAT-AD).

- Dataset source: https://doi.org/10.5281/zenodo.12588359
- Original repository: https://github.com/kplabs-pl/OPS-SAT-AD
- Research paper: *The OPS-SAT benchmark for detecting anomalies in satellite telemetry.*

The dataset is reused subject to its applicable licence. Credit for the original dataset belongs to its respective authors.

## Academic Purpose

This project is developed as part of a B.Tech capstone project for academic and research purposes.