"""
Satellite Telemetry Anomaly Detection - XGBoost Classifier
==========================================================
Script: src/train_xgboost.py
Dataset: data/dataset.csv

Description:
Trains and evaluates an Extreme Gradient Boosting (XGBoost) classifier for
detecting anomalies in satellite telemetry using the 18 handcrafted features
from the OPS-SAT-AD benchmark.

Key Technical Configurations:
- Input Features (X): 18 handcrafted features (statistics, peaks, smoothing, derivatives)
- Excluded Columns: 'segment', 'channel', 'sampling', 'anomaly', 'train'
- Target (y): 'anomaly' (0 = Nominal, 1 = Anomalous)
- Train Split: train == 1 (1594 samples: 1273 nominal, 321 anomalous)
- Test Split: train == 0 (529 samples: 416 nominal, 113 anomalous)
- Imbalance Handling: scale_pos_weight calculated dynamically from training labels
  (negative_count / positive_count = 1273 / 321 ≈ 3.9657)
- Reproducibility: random_state=42

Evaluation Metrics:
- Accuracy, Precision, Recall, F1-score, MCC, ROC-AUC, Average Precision (PR-AUC),
  and Confusion Matrix.

Artifacts Generated:
- Trained model saved to: models/xgboost_model.json
- Metrics saved to: results/xgboost_metrics.json & results/xgboost_metrics.txt
- Confusion matrix heatmap saved to: results/xgboost_confusion_matrix.png
"""

import json
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
import xgboost as xgb


def load_and_prepare_data(dataset_path: str):
    """
    Loads dataset, isolates the 18 handcrafted features, and extracts
    the benchmark training (train == 1) and testing (train == 0) splits.
    """
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file not found at: {dataset_path}")

    df = pd.read_csv(dataset_path)

    # Exclude metadata and label columns to maintain 18-feature input space
    metadata_cols = ["segment", "channel", "sampling", "anomaly", "train"]
    feature_cols = [col for col in df.columns if col not in metadata_cols]

    if len(feature_cols) != 18:
        raise ValueError(f"Expected 18 features, but found {len(feature_cols)}: {feature_cols}")

    # Benchmark train/test split masks
    train_mask = df["train"] == 1
    test_mask = df["train"] == 0

    X_train = df.loc[train_mask, feature_cols].copy()
    y_train = df.loc[train_mask, "anomaly"].copy()

    X_test = df.loc[test_mask, feature_cols].copy()
    y_test = df.loc[test_mask, "anomaly"].copy()

    return X_train, y_train, X_test, y_test, feature_cols


def train_xgboost(X_train: pd.DataFrame, y_train: pd.Series) -> xgb.XGBClassifier:
    """
    Trains an XGBoost classifier with dynamic scale_pos_weight
    derived from training class proportions to address class imbalance.
    """
    neg_count = int((y_train == 0).sum())
    pos_count = int((y_train == 1).sum())
    scale_pos_weight = neg_count / pos_count

    print(f"  Training label distribution: Nominal = {neg_count}, Anomalous = {pos_count}")
    print(f"  Calculated scale_pos_weight: {scale_pos_weight:.4f}")

    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric="logloss",
        
        n_jobs=-1
    )

    # Fit strictly on training split
    model.fit(X_train, y_train)
    return model


def evaluate_model(model: xgb.XGBClassifier, X_test: pd.DataFrame, y_test: pd.Series):
    """
    Evaluates XGBoost model performance on the unseen test set across all benchmark metrics.
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    # Calculate metrics
    metrics = {
        "Accuracy": float(accuracy_score(y_test, y_pred)),
        "Precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "F1-score": float(f1_score(y_test, y_pred, zero_division=0)),
        "MCC": float(matthews_corrcoef(y_test, y_pred)),
        "ROC-AUC": float(roc_auc_score(y_test, y_prob)),
        "Average_Precision_AP": float(average_precision_score(y_test, y_prob)),
        "True_Negatives": int(tn),
        "False_Positives": int(fp),
        "False_Negatives": int(fn),
        "True_Positives": int(tp),
        "Test_Samples_Total": int(len(y_test)),
    }

    return metrics, cm, y_pred, y_prob


def save_artifacts(model: xgb.XGBClassifier, metrics: dict, cm: np.ndarray, base_dir: str):
    """
    Saves native XGBoost model file, evaluation metrics, and confusion matrix plot.
    """
    models_dir = os.path.join(base_dir, "models")
    results_dir = os.path.join(base_dir, "results")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # 1. Save trained model in native JSON format
    model_path = os.path.join(models_dir, "xgboost_model.json")
    model.save_model(model_path)
    print(f"Model saved to: {model_path}")

    # 2. Save metrics as JSON
    json_path = os.path.join(results_dir, "xgboost_metrics.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics (JSON) saved to: {json_path}")

    # 3. Save formatted text report
    txt_path = os.path.join(results_dir, "xgboost_metrics.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("XGBOOST EVALUATION RESULTS (OPS-SAT-AD BENCHMARK)\n")
        f.write("=" * 60 + "\n\n")
        for k, v in metrics.items():
            if isinstance(v, float):
                f.write(f"{k:<25}: {v:.4f}\n")
            else:
                f.write(f"{k:<25}: {v}\n")
    print(f"Metrics (TXT) saved to: {txt_path}")

    # 4. Save Confusion Matrix plot
    cm_path = os.path.join(results_dir, "xgboost_confusion_matrix.png")
    plt.figure(figsize=(6, 5), dpi=150)
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("XGBoost - Confusion Matrix (Test Set)", fontsize=13, pad=12, fontweight="bold")
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
    print("SATELLITE TELEMETRY ANOMALY DETECTION - XGBOOST")
    print("=" * 70)

    # 1. Load data
    print("Loading data and configuring benchmark split...")
    X_train, y_train, X_test, y_test, feature_names = load_and_prepare_data(dataset_path)
    print(f"  Training samples: {len(X_train)} (Nominal: {sum(y_train == 0)}, Anomalous: {sum(y_train == 1)})")
    print(f"  Testing samples : {len(X_test)} (Nominal: {sum(y_test == 0)}, Anomalous: {sum(y_test == 1)})")
    print(f"  Features used ({len(feature_names)}): {feature_names}\n")

    # 2. Train model
    print("Training XGBoost Classifier...")
    model = train_xgboost(X_train, y_train)
    print("Training complete.\n")

    # 3. Evaluate model
    print("Evaluating XGBoost model on unseen test set...")
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
    print(f"  Average Precision (AP)   : {metrics['Average_Precision_AP']:.4f}")
    print("\nConfusion Matrix:")
    print(f"  True Negatives (TN)     : {metrics['True_Negatives']}")
    print(f"  False Positives (FP)    : {metrics['False_Positives']}")
    print(f"  False Negatives (FN)    : {metrics['False_Negatives']}")
    print(f"  True Positives (TP)     : {metrics['True_Positives']}")
    print("-" * 70)

    # 4. Save artifacts
    print("Saving model, metrics, and confusion matrix...")
    save_artifacts(model, metrics, cm, base_dir)
    print("=" * 70)
    print("Execution completed successfully.")


if __name__ == "__main__":
    main()

