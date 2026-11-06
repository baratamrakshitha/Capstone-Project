"""
Satellite Telemetry Anomaly Detection - Isolation Forest (Unsupervised)
======================================================================
Script: src/train_isolation_forest.py
Dataset: data/dataset.csv

Description:
Implements and evaluates an unsupervised Isolation Forest model for satellite
telemetry anomaly detection using the 18 handcrafted features from the
OPS-SAT-AD benchmark.

Unsupervised Model Integrity:
- Training phase: Fitted strictly on X_train without accessing target labels (y_train).
- Contamination factor: Set to 0.2 (reflecting the ~20% anomaly distribution in T).
- Label mapping: scikit-learn predicts 1 (normal) and -1 (anomaly).
  These are transformed to:
    0 = Normal (Nominal)
    1 = Anomalous (Outlier)
- Anomaly scoring for ROC-AUC and PR-AUC: Uses negative decision function
  (-decision_function) so that higher scores correspond to higher anomaly likelihood.
- Testing phase: Ground-truth test labels (y_test) are used solely for metric evaluation.

Evaluation Metrics:
- Accuracy, Precision, Recall, F1-score, MCC, ROC-AUC, Average Precision (PR-AUC),
  and Confusion Matrix.

Artifacts Generated:
- Trained model saved to: models/isolation_forest_model.joblib
- Metrics saved to: results/isolation_forest_metrics.json & results/isolation_forest_metrics.txt
- Confusion matrix plot saved to: results/isolation_forest_confusion_matrix.png
"""

import json
import os
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


def load_and_prepare_data(dataset_path: str):
    """
    Loads dataset, isolates the 18 handcrafted features, and extracts
    training and testing splits according to the original benchmark.
    NOTE: y_train is strictly withheld from model fitting (unsupervised).
    """
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file not found at: {dataset_path}")

    df = pd.read_csv(dataset_path)

    # Exclude metadata columns to preserve exact 18-feature input space
    metadata_cols = ["segment", "channel", "sampling", "anomaly", "train"]
    feature_cols = [col for col in df.columns if col not in metadata_cols]

    if len(feature_cols) != 18:
        raise ValueError(f"Expected 18 features, but found {len(feature_cols)}: {feature_cols}")

    # Benchmark train/test split masks
    train_mask = df["train"] == 1
    test_mask = df["train"] == 0

    # Training input (Unsupervised: labels are not used for fitting)
    X_train = df.loc[train_mask, feature_cols].copy()

    # Test input and evaluation labels
    X_test = df.loc[test_mask, feature_cols].copy()
    y_test = df.loc[test_mask, "anomaly"].copy()

    return X_train, X_test, y_test, feature_cols


def train_isolation_forest(X_train: pd.DataFrame) -> IsolationForest:
    """
    Trains the Isolation Forest in an unsupervised manner.
    - contamination=0.2: Expected proportion of outliers matching training anomaly distribution.
    - random_state=42: Ensures reproducible tree partitioning.
    - n_jobs=-1: Enables parallel computation.
    """
    model = IsolationForest(
        n_estimators=100,
        contamination=0.2,
        random_state=42,
        n_jobs=-1
    )
    # Fit strictly without target labels
    model.fit(X_train)
    return model


def evaluate_model(model: IsolationForest, X_test: pd.DataFrame, y_test: pd.Series):
    """
    Evaluates the unsupervised model on the unseen test set.
    Converts scikit-learn outputs (1: normal, -1: anomaly) into binary labels (0: normal, 1: anomaly).
    """
    raw_preds = model.predict(X_test)
    # Map: -1 (anomaly) -> 1, 1 (normal) -> 0
    y_pred = np.where(raw_preds == -1, 1, 0)

    # Continuous anomaly score: In scikit-learn, lower decision_function values mean more anomalous.
    # We negate the decision function so higher values indicate higher anomaly likelihood.
    anomaly_scores = -model.decision_function(X_test)

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    # Compute standard benchmark metrics
    metrics = {
        "Accuracy": float(accuracy_score(y_test, y_pred)),
        "Precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "F1-score": float(f1_score(y_test, y_pred, zero_division=0)),
        "MCC": float(matthews_corrcoef(y_test, y_pred)),
        "ROC-AUC": float(roc_auc_score(y_test, anomaly_scores)),
        "Average_Precision_PR_AUC": float(average_precision_score(y_test, anomaly_scores)),
        "True_Negatives": int(tn),
        "False_Positives": int(fp),
        "False_Negatives": int(fn),
        "True_Positives": int(tp),
        "Test_Samples_Total": int(len(y_test)),
    }

    return metrics, cm, y_pred, anomaly_scores


def save_artifacts(model: IsolationForest, metrics: dict, cm: np.ndarray, base_dir: str):
    """
    Persists trained model checkpoint, metric results, and confusion matrix visualization.
    """
    models_dir = os.path.join(base_dir, "models")
    results_dir = os.path.join(base_dir, "results")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # 1. Save trained model
    model_path = os.path.join(models_dir, "isolation_forest_model.joblib")
    joblib.dump(model, model_path)
    print(f"Model saved to: {model_path}")

    # 2. Save metrics as JSON
    json_path = os.path.join(results_dir, "isolation_forest_metrics.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics (JSON) saved to: {json_path}")

    # 3. Save formatted text report
    txt_path = os.path.join(results_dir, "isolation_forest_metrics.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("ISOLATION FOREST EVALUATION RESULTS (OPS-SAT-AD BENCHMARK)\n")
        f.write("=" * 60 + "\n\n")
        for k, v in metrics.items():
            if isinstance(v, float):
                f.write(f"{k:<25}: {v:.4f}\n")
            else:
                f.write(f"{k:<25}: {v}\n")
    print(f"Metrics (TXT) saved to: {txt_path}")

    # 4. Save Confusion Matrix plot
    cm_path = os.path.join(results_dir, "isolation_forest_confusion_matrix.png")
    plt.figure(figsize=(6, 5), dpi=150)
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Isolation Forest - Confusion Matrix (Test Set)", fontsize=13, pad=12, fontweight="bold")
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ["Nominal (0)", "Anomalous (1)"], fontsize=10)
    plt.yticks(tick_marks, ["Nominal (0)", "Anomalous (1)"], fontsize=10)

    # Annotate numbers inside matrix cells
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > thresh else "black"
            plt.text(
                j,
                i,
                f"{cm[i, j]}",
                horizontalalignment="center",
                verticalalignment="center",
                color=color,
                fontsize=14,
                fontweight="bold"
            )

    plt.ylabel("True Class", fontsize=11)
    plt.xlabel("Predicted Class", fontsize=11)
    plt.tight_layout()
    plt.savefig(cm_path)
    plt.close()
    print(f"Confusion matrix plot saved to: {cm_path}")


def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    dataset_path = os.path.join(base_dir, "data", "dataset.csv")

    print("=" * 70)
    print("SATELLITE TELEMETRY ANOMALY DETECTION - ISOLATION FOREST")
    print("=" * 70)

    # 1. Load data
    print("Loading data and configuring unsupervised training setup...")
    X_train, X_test, y_test, feature_names = load_and_prepare_data(dataset_path)
    print(f"  Training samples (unlabeled X_train): {len(X_train)}")
    print(f"  Testing samples  (X_test for eval)  : {len(X_test)} (Nominal: {sum(y_test == 0)}, Anomalous: {sum(y_test == 1)})")
    print(f"  Features used ({len(feature_names)}): {feature_names}\n")

    # 2. Train model (Strictly unsupervised)
    print("Training Isolation Forest (contamination=0.2, random_state=42)...")
    model = train_isolation_forest(X_train)
    print("Unsupervised training completed.\n")

    # 3. Evaluate model on test set
    print("Evaluating Isolation Forest on unseen test set...")
    metrics, cm, _, _ = evaluate_model(model, X_test, y_test)

    print("-" * 70)
    print("EVALUATION METRICS:")
    print("-" * 70)
    print(f"  Accuracy                : {metrics['Accuracy']:.4f}")
    print(f"  Precision               : {metrics['Precision']:.4f}")
    print(f"  Recall                  : {metrics['Recall']:.4f}")
    print(f"  F1-Score                : {metrics['F1-score']:.4f}")
    print(f"  MCC                     : {metrics['MCC']:.4f}")
    print(f"  ROC-AUC                 : {metrics['ROC-AUC']:.4f}")
    print(f"  Average Precision (PR)  : {metrics['Average_Precision_PR_AUC']:.4f}")
    print("\nConfusion Matrix:")
    print(f"  True Negatives (TN)     : {metrics['True_Negatives']}")
    print(f"  False Positives (FP)    : {metrics['False_Positives']}")
    print(f"  False Negatives (FN)    : {metrics['False_Negatives']}")
    print(f"  True Positives (TP)     : {metrics['True_Positives']}")
    print("-" * 70)

    # 4. Save artifacts
    print("Saving model checkpoint, metrics, and confusion matrix...")
    save_artifacts(model, metrics, cm, base_dir)
    print("=" * 70)
    print("Execution completed successfully.")


if __name__ == "__main__":
    main()

