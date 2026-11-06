"""
Satellite Telemetry Anomaly Detection - Random Forest Classifier
===============================================================
Script: src/train_random_forest.py
Dataset: data/dataset.csv

Description:
Trains and evaluates a Random Forest classifier for detecting anomalies in
satellite telemetry data using the exact 18 handcrafted features and the
stratified training/testing split established in the OPS-SAT-AD benchmark.

Features & Targets:
- Input Features (X): 18 handcrafted features (statistics, peaks, smoothing, derivatives)
- Excluded Columns: 'segment', 'channel', 'sampling', 'anomaly', 'train'
- Target (y): 'anomaly' (0 = Nominal, 1 = Anomalous)
- Train Split: train == 1 (Training set T, 1594 segments)
- Test Split: train == 0 (Testing set Psi, 529 segments)

Evaluation Metrics:
- Accuracy, Precision, Recall, F1-score, Matthews Correlation Coefficient (MCC),
  ROC-AUC, PR-AUC, and Confusion Matrix.

Artifacts Generated:
- Trained model saved to: models/random_forest_model.joblib
- Evaluation metrics saved to: results/random_forest_metrics.json & results/random_forest_metrics.txt
- Confusion matrix heatmap saved to: results/random_forest_confusion_matrix.png
"""

import json
import os
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
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
    Loads dataset, isolates the 18 handcrafted features, and separates
    into training and testing splits according to the original benchmark.
    """
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file not found at: {dataset_path}")

    df = pd.read_csv(dataset_path)

    # Exclude non-feature metadata and label columns
    metadata_cols = ["segment", "channel", "sampling", "anomaly", "train"]
    feature_cols = [col for col in df.columns if col not in metadata_cols]

    if len(feature_cols) != 18:
        raise ValueError(f"Expected 18 features, but found {len(feature_cols)}: {feature_cols}")

    # Split using original benchmark 'train' column
    train_mask = df["train"] == 1
    test_mask = df["train"] == 0

    X_train = df.loc[train_mask, feature_cols].copy()
    y_train = df.loc[train_mask, "anomaly"].copy()

    X_test = df.loc[test_mask, feature_cols].copy()
    y_test = df.loc[test_mask, "anomaly"].copy()

    return X_train, y_train, X_test, y_test, feature_cols


def train_random_forest(X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestClassifier:
    """
    Initializes and trains the Random Forest classifier.
    - class_weight='balanced': Compensates for class imbalance (20% anomalies).
    - random_state=42: Ensures reproducible training.
    """
    model = RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model: RandomForestClassifier, X_test: pd.DataFrame, y_test: pd.Series):
    """
    Evaluates model on the unseen test set across all benchmark metrics.
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    # Calculate all 7 paper metrics
    metrics = {
        "Accuracy": float(accuracy_score(y_test, y_pred)),
        "Precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "F1-score": float(f1_score(y_test, y_pred, zero_division=0)),
        "MCC": float(matthews_corrcoef(y_test, y_pred)),
        "ROC-AUC": float(roc_auc_score(y_test, y_prob)),
        "PR-AUC": float(average_precision_score(y_test, y_prob)),
        "True_Negatives": int(tn),
        "False_Positives": int(fp),
        "False_Negatives": int(fn),
        "True_Positives": int(tp),
        "Test_Samples_Total": int(len(y_test)),
    }

    return metrics, cm, y_pred, y_prob


def save_artifacts(model, metrics: dict, cm: np.ndarray, base_dir: str):
    """
    Saves trained model checkpoint, metric results, and confusion matrix plot.
    """
    models_dir = os.path.join(base_dir, "models")
    results_dir = os.path.join(base_dir, "results")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # 1. Save trained model
    model_path = os.path.join(models_dir, "random_forest_model.joblib")
    joblib.dump(model, model_path)
    print(f"Model saved to: {model_path}")

    # 2. Save metrics as JSON
    json_path = os.path.join(results_dir, "random_forest_metrics.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics (JSON) saved to: {json_path}")

    # 3. Save formatted text report
    txt_path = os.path.join(results_dir, "random_forest_metrics.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("RANDOM FOREST EVALUATION RESULTS (OPS-SAT-AD BENCHMARK)\n")
        f.write("=" * 60 + "\n\n")
        for k, v in metrics.items():
            if isinstance(v, float):
                f.write(f"{k:<20}: {v:.4f}\n")
            else:
                f.write(f"{k:<20}: {v}\n")
    print(f"Metrics (TXT) saved to: {txt_path}")

    # 4. Save Confusion Matrix plot
    cm_path = os.path.join(results_dir, "random_forest_confusion_matrix.png")
    plt.figure(figsize=(6, 5), dpi=150)
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Random Forest - Confusion Matrix (Test Set)", fontsize=13, pad=12, fontweight="bold")
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
    print("SATELLITE TELEMETRY ANOMALY DETECTION - RANDOM FOREST")
    print("=" * 70)

    # 1. Load data
    print("Loading data and splitting according to benchmark...")
    X_train, y_train, X_test, y_test, feature_names = load_and_prepare_data(dataset_path)
    print(f"  Training samples: {len(X_train)} (Nominal: {sum(y_train == 0)}, Anomalous: {sum(y_train == 1)})")
    print(f"  Testing samples : {len(X_test)} (Nominal: {sum(y_test == 0)}, Anomalous: {sum(y_test == 1)})")
    print(f"  Features used ({len(feature_names)}): {feature_names}\n")

    # 2. Train model
    print("Training Random Forest Classifier (class_weight='balanced', random_state=42)...")
    model = train_random_forest(X_train, y_train)
    print("Training complete.\n")

    # 3. Evaluate model
    print("Evaluating model on unseen test set...")
    metrics, cm, _, _ = evaluate_model(model, X_test, y_test)

    print("-" * 70)
    print("EVALUATION METRICS:")
    print("-" * 70)
    print(f"  Accuracy : {metrics['Accuracy']:.4f}")
    print(f"  Precision: {metrics['Precision']:.4f}")
    print(f"  Recall   : {metrics['Recall']:.4f}")
    print(f"  F1-Score : {metrics['F1-score']:.4f}")
    print(f"  MCC      : {metrics['MCC']:.4f}")
    print(f"  ROC-AUC  : {metrics['ROC-AUC']:.4f}")
    print(f"  PR-AUC   : {metrics['PR-AUC']:.4f}")
    print("\nConfusion Matrix:")
    print(f"  True Negatives (TN) : {metrics['True_Negatives']}")
    print(f"  False Positives (FP): {metrics['False_Positives']}")
    print(f"  False Negatives (FN): {metrics['False_Negatives']}")
    print(f"  True Positives (TP) : {metrics['True_Positives']}")
    print("-" * 70)

    # 4. Save artifacts
    print("Saving model, metrics, and visualization...")
    save_artifacts(model, metrics, cm, base_dir)
    print("=" * 70)
    print("Execution completed successfully.")


if __name__ == "__main__":
    main()

