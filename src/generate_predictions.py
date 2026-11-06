
"""
Satellite Telemetry Anomaly Detection
Generate predictions from the three trained models.

Input:
    data/dataset.csv
    models/random_forest_model.joblib
    models/isolation_forest_model.joblib
    models/xgboost_model.json

Output:
    results/prediction_results.csv
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb


def main():
    # Project directories
    BASE_DIR = Path(__file__).resolve().parent.parent
    DATA_PATH = BASE_DIR / "data" / "dataset.csv"
    MODELS_DIR = BASE_DIR / "models"
    RESULTS_DIR = BASE_DIR / "results"

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load dataset
    print("Loading satellite telemetry dataset...")
    df = pd.read_csv(DATA_PATH)

    # Identify the 18 input features
    metadata_cols = [
        "segment", "channel", "sampling", "anomaly", "train"
    ]
    feature_cols = [
        col for col in df.columns if col not in metadata_cols
    ]

    if len(feature_cols) != 18:
        raise ValueError(
            f"Expected 18 features, found {len(feature_cols)}"
        )

    # Select the original test split
    test_df = df[df["train"] == 0].copy()
    X_test = test_df[feature_cols]
    y_actual = test_df["anomaly"].astype(int).to_numpy()

    print(f"Test samples: {len(X_test)}")

    # Load trained models
    print("Loading trained models...")

    rf_model = joblib.load(
        MODELS_DIR / "random_forest_model.joblib"
    )

    if_model = joblib.load(
        MODELS_DIR / "isolation_forest_model.joblib"
    )

    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(
        str(MODELS_DIR / "xgboost_model.json")
    )

    # Generate predictions
    print("Generating model predictions...")

    rf_pred = rf_model.predict(X_test).astype(int)
    rf_score = rf_model.predict_proba(X_test)[:, 1]

    if_raw_pred = if_model.predict(X_test)
    if_pred = np.where(if_raw_pred == -1, 1, 0).astype(int)
    if_score = -if_model.decision_function(X_test)

    xgb_pred = xgb_model.predict(X_test).astype(int)
    xgb_score = xgb_model.predict_proba(X_test)[:, 1]

    # Build prediction results
    results = pd.DataFrame({
        "Actual": y_actual,

        "Random_Forest_Predicted": rf_pred,
        "Random_Forest_Anomaly_Score": rf_score,

        "Isolation_Forest_Predicted": if_pred,
        "Isolation_Forest_Anomaly_Score": if_score,

        "XGBoost_Predicted": xgb_pred,
        "XGBoost_Anomaly_Score": xgb_score,
    })

    # Include useful sample identifiers
    for column in ["segment", "channel", "sampling"]:
        if column in test_df.columns:
            results.insert(
                0, column, test_df[column].to_numpy()
            )

    # Save predictions
    output_path = RESULTS_DIR / "prediction_results.csv"
    results.to_csv(output_path, index=False)

    print("\nPrediction generation completed!")
    print(f"Saved to: {output_path}")
    print(f"Total predictions: {len(results)}")

    print("\nActual class distribution:")
    print(results["Actual"].value_counts().sort_index())

    print("\nPrediction preview:")
    print(results.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
