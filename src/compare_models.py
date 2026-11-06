"""
Satellite Telemetry Anomaly Detection - Three-Model Performance Comparison
==========================================================================
Script: src/compare_models.py
Artifacts: results/random_forest_metrics.json
           results/isolation_forest_metrics.json
           results/xgboost_metrics.json

Description:
Loads, aligns, and compares evaluation metrics across all three benchmark models:
1. Random Forest (Supervised Ensemble)
2. Isolation Forest (Unsupervised Baseline)
3. XGBoost (Supervised Gradient Boosting)

Key Capabilities:
- Loads metric JSON artifacts directly without retraining or data modification.
- Evaluates 7 core metrics: Accuracy, Precision, Recall, F1-score, MCC, ROC-AUC, Average Precision (AP).
- Handles MCC (range [-1, 1]) properly on the visualization with baseline and dynamic scaling.
- Identifies the highest-scoring model for each individual metric.
- Displays full comparative tables including Confusion Matrix breakdown in the terminal.
- Exports structured comparison table to results/model_comparison.csv.
- Saves publication-quality 3-model grouped bar chart to results/model_comparison.png.
"""

import json
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd



def extract_ap(raw_dict):
    for key in [
        "Average Precision (AP)",
        "Average_Precision_AP",
        "Average_Precision_PR_AUC",
        "PR-AUC"
    ]:
        if key in raw_dict:
            return float(raw_dict[key])

    raise KeyError(
        "Average Precision (AP) not found in metrics JSON. "
        f"Available keys: {list(raw_dict.keys())}"
    )
    if key in raw_dict:
            return float(raw_dict[key])

    raise KeyError(
        "Average Precision (AP) not found in metrics JSON. "
        f"Available keys: {list(raw_dict.keys())}"
    )


def load_metrics(rf_path: str, if_path: str, xgb_path: str):
    """
    Loads JSON metric summaries for all three models.
    """

    for path, name in [
        (rf_path, "Random Forest"),
        (if_path, "Isolation Forest"),
        (xgb_path, "XGBoost")
    ]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{name} metrics file not found at: {path}"
            )

    with open(rf_path, "r", encoding="utf-8") as f:
        rf_raw = json.load(f)

    with open(if_path, "r", encoding="utf-8") as f:
        if_raw = json.load(f)

    with open(xgb_path, "r", encoding="utf-8") as f:
        xgb_raw = json.load(f)

    metrics_list = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1-score",
        "MCC",
        "ROC-AUC",
        "Average Precision (AP)"
    ]

    rf_data = {
        "Accuracy": float(rf_raw["Accuracy"]),
        "Precision": float(rf_raw["Precision"]),
        "Recall": float(rf_raw["Recall"]),
        "F1-score": float(rf_raw["F1-score"]),
        "MCC": float(rf_raw["MCC"]),
        "ROC-AUC": float(rf_raw["ROC-AUC"]),
        "Average Precision (AP)": extract_ap(rf_raw)
    }

    if_data = {
        "Accuracy": float(if_raw["Accuracy"]),
        "Precision": float(if_raw["Precision"]),
        "Recall": float(if_raw["Recall"]),
        "F1-score": float(if_raw["F1-score"]),
        "MCC": float(if_raw["MCC"]),
        "ROC-AUC": float(if_raw["ROC-AUC"]),
        "Average Precision (AP)": extract_ap(if_raw)
    }

    xgb_data = {
        "Accuracy": float(xgb_raw["Accuracy"]),
        "Precision": float(xgb_raw["Precision"]),
        "Recall": float(xgb_raw["Recall"]),
        "F1-score": float(xgb_raw["F1-score"]),
        "MCC": float(xgb_raw["MCC"]),
        "ROC-AUC": float(xgb_raw["ROC-AUC"]),
        "Average Precision (AP)": extract_ap(xgb_raw)
    }

    return (
        metrics_list,
        rf_data,
        if_data,
        xgb_data,
        rf_raw,
        if_raw,
        xgb_raw
    )



def create_comparison_dataframe(metrics_list, rf_data, if_data, xgb_data):
    """
    Builds comparison DataFrame with scores for all three models and highlights the best performer.
    """
    records = []
    for metric in metrics_list:
        rf_val = rf_data[metric]
        if_val = if_data[metric]
        xgb_val = xgb_data[metric]

        # Determine winner
        scores = {
            "Random Forest": rf_val,
            "Isolation Forest": if_val,
            "XGBoost": xgb_val
        }
        best_score = max(scores.values())
        best_models = [m for m, s in scores.items() if np.isclose(s, best_score, atol=1e-5)]
        best_model_str = ", ".join(best_models)

        records.append({
            "Metric": metric,
            "Random Forest": round(rf_val, 4),
            "Isolation Forest": round(if_val, 4),
            "XGBoost": round(xgb_val, 4),
            "Best Model": best_model_str
        })

    return pd.DataFrame(records)


def plot_comparison_chart(df_comp, save_path: str):
    """
    Generates a grouped bar chart comparing all 7 metrics across all three models.
    Properly accommodates MCC's potential range [-1, 1] relative to [0, 1] bounded metrics.
    """
    metrics = df_comp["Metric"].tolist()
    rf_scores = df_comp["Random Forest"].tolist()
    if_scores = df_comp["Isolation Forest"].tolist()
    xgb_scores = df_comp["XGBoost"].tolist()

    all_scores = rf_scores + if_scores + xgb_scores
    min_score = min(all_scores)

    x = np.arange(len(metrics))
    width = 0.26  # Width of each bar

    plt.figure(figsize=(14, 7), dpi=150)

    # Bars for each model
    bars_rf = plt.bar(
        x - width,
        rf_scores,
        width,
        label="Random Forest (Supervised)",
        color="#2b5c8f",
        edgecolor="black",
        linewidth=0.8,
        alpha=0.9
    )
    bars_if = plt.bar(
        x,
        if_scores,
        width,
        label="Isolation Forest (Unsupervised)",
        color="#e66101",
        edgecolor="black",
        linewidth=0.8,
        alpha=0.9
    )
    bars_xgb = plt.bar(
        x + width,
        xgb_scores,
        width,
        label="XGBoost (Supervised)",
        color="#238b45",
        edgecolor="black",
        linewidth=0.8,
        alpha=0.9
    )

    # Title & Labels
    plt.title(
        "OPS-SAT Satellite Telemetry Anomaly Detection: 3-Model Performance Benchmark",
        fontsize=14,
        pad=16,
        fontweight="bold"
    )
    plt.xlabel("Evaluation Metric", fontsize=12, labelpad=10)
    plt.ylabel("Score", fontsize=12)
    plt.xticks(x, metrics, rotation=15, ha="right", fontsize=10, fontweight="medium")

    # Correct MCC Range Handling:
    # Most metrics span [0, 1], but MCC spans [-1, 1].
    # Set y-axis floor dynamically and add zero-reference line if needed.
    max_score = max(all_scores)

    y_min = min(0.0, min_score - 0.08)
    y_max = max(1.10, max_score + 0.08)

    plt.ylim(y_min, y_max)
    plt.axhline(0, color="black", linestyle="-", linewidth=0.8, alpha=0.5)
    plt.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, 1.18),
    ncol=3,
    frameon=True,
    fontsize=9
)
    

    # Helper function to place text labels on bars
    def annotate_bars(bars, color):
        for bar in bars:
            h = bar.get_height()
            if h >= 0:
                y_pos = h + 0.02
                va = "bottom"
            else:
                y_pos = h - 0.04
                va = "top"
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                y_pos,
                f"{h:.3f}",
                ha="center",
                va=va,
                fontsize=8.5,
                fontweight="bold",
                color=color
            )

    annotate_bars(bars_rf, "#1c3d5a")
    annotate_bars(bars_if, "#8c3b00")
    annotate_bars(bars_xgb, "#0e4d25")

    # Informative note about MCC range
    plt.figtext(
        0.13,
        0.02,
        "* Note: Matthews Correlation Coefficient (MCC) naturally spans [-1.0, +1.0]; all other metrics span [0.0, 1.0].",
        fontsize=9,
        style="italic",
        color="#555555"
    )

    plt.tight_layout(rect=[0, 0.04, 1, 0.90])
    plt.savefig(save_path)
    plt.close()
    print(f"Comparison chart saved to: {save_path}")


def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    results_dir = os.path.join(base_dir, "results")

    rf_json = os.path.join(results_dir, "random_forest_metrics.json")
    if_json = os.path.join(results_dir, "isolation_forest_metrics.json")
    xgb_json = os.path.join(results_dir, "xgboost_metrics.json")
    csv_out = os.path.join(results_dir, "model_comparison.csv")
    png_out = os.path.join(results_dir, "model_comparison.png")

    print("=" * 88)
    print("SATELLITE TELEMETRY ANOMALY DETECTION - THREE-MODEL BENCHMARK COMPARISON")
    print("=" * 88)

    # 1. Load metrics
    metrics_list, rf_data, if_data, xgb_data, rf_raw, if_raw, xgb_raw = load_metrics(
        rf_json, if_json, xgb_json
    )

    # 2. Build comparison dataframe
    df_comp = create_comparison_dataframe(metrics_list, rf_data, if_data, xgb_data)

    # 3. Display comparison table in terminal
    print("\nBENCHMARK COMPARISON TABLE (Test Set Size = 529 Samples):")
    print("-" * 88)
    header_fmt = "{:<26} | {:<14} | {:<16} | {:<14} | {:<18}"
    row_fmt = "{:<26} | {:<14.4f} | {:<16.4f} | {:<14.4f} | {:<18}"
    print(header_fmt.format("Metric", "Random Forest", "Isolation Forest", "XGBoost", "Best Model"))
    print("-" * 88)
    for _, row in df_comp.iterrows():
        print(row_fmt.format(
            row["Metric"],
            row["Random Forest"],
            row["Isolation Forest"],
            row["XGBoost"],
            row["Best Model"]
        ))
    print("-" * 88)

    # 4. Display Confusion Matrix summary for all 3 models
    print("\nCONFUSION MATRIX SUMMARY:")
    print("-" * 88)
    cm_header = "{:<26} | {:<14} | {:<16} | {:<14}"
    cm_row = "{:<26} | {:<14} | {:<16} | {:<14}"
    print(cm_header.format("Metric / Count", "Random Forest", "Isolation Forest", "XGBoost"))
    print("-" * 88)
    print(cm_row.format("True Negatives (TN)", rf_raw["True_Negatives"], if_raw["True_Negatives"], xgb_raw["True_Negatives"]))
    print(cm_row.format("False Positives (FP)", rf_raw["False_Positives"], if_raw["False_Positives"], xgb_raw["False_Positives"]))
    print(cm_row.format("False Negatives (FN)", rf_raw["False_Negatives"], if_raw["False_Negatives"], xgb_raw["False_Negatives"]))
    print(cm_row.format("True Positives (TP)", rf_raw["True_Positives"], if_raw["True_Positives"], xgb_raw["True_Positives"]))
    print("-" * 88)

    # 5. Save updated comparison table as CSV
    df_comp.to_csv(csv_out, index=False)
    print(f"\nComparison table exported to: {csv_out}")

    # 6. Generate and save grouped bar chart
    plot_comparison_chart(df_comp, png_out)

    # 7. Summary of Wins & Operational Insights

    rf_wins = sum(
        df_comp["Best Model"].str.contains("Random Forest")
    )
    if_wins = sum(
        df_comp["Best Model"].str.contains("Isolation Forest")
    )
    xgb_wins = sum(
        df_comp["Best Model"].str.contains("XGBoost")
    )

    print("\nPERFORMANCE SUMMARY & KEY TAKEAWAYS:")
    print(f"- Random Forest leads in {rf_wins} of {len(metrics_list)} metrics.")
    print(f"- Isolation Forest leads in {if_wins} of {len(metrics_list)} metrics.")
    print(f"- XGBoost leads in {xgb_wins} of {len(metrics_list)} metrics.")

    print(f"- Random Forest Precision: {rf_data['Precision']:.4f}")
    print(f"- XGBoost Recall: {xgb_data['Recall']:.4f}")
    print(f"- Isolation Forest Recall: {if_data['Recall']:.4f}")

    print("=" * 88)


if __name__ == "__main__":
    main()
