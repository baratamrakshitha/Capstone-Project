"""
Satellite Telemetry Anomaly Detection - Exploratory Data Analysis (EDA)
======================================================================
Script: src/explore_data.py
Dataset: data/dataset.csv

Description:
This script performs foundational data exploration on the OPS-SAT-AD benchmark
dataset. It loads the dataset, inspects its schema, checks data cleanliness,
summarizes target and split distributions, computes descriptive statistics
for the 18 handcrafted telemetry features, and saves visualization plots.

Operational Requirements Adhered To:
- Target column: 'anomaly' (0 = Nominal, 1 = Anomalous)
- Excluded metadata from model features: 'segment', 'channel', 'sampling', 'anomaly', 'train'
- Exactly 18 handcrafted features analyzed as model input candidates
- No dataset modifications or synthetic data generation
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def main():
    # ---------------------------------------------------------
    # 1. Path Configuration
    # ---------------------------------------------------------
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    dataset_path = os.path.join(base_dir, "data", "dataset.csv")
    results_dir = os.path.join(base_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    print("=" * 70)
    print("SATELLITE TELEMETRY ANOMALY DETECTION - DATA EXPLORATION")
    print("=" * 70)
    print(f"Loading dataset from: {dataset_path}")

    # ---------------------------------------------------------
    # 2. Load the CSV Dataset
    # ---------------------------------------------------------
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at expected location: {dataset_path}")

    df = pd.read_csv(dataset_path)
    print("Dataset successfully loaded.\n")

    # ---------------------------------------------------------
    # 3. Display First 5 Records
    # ---------------------------------------------------------
    print("-" * 70)
    print("1. FIRST 5 RECORDS")
    print("-" * 70)
    print(df.head())
    print()

    # ---------------------------------------------------------
    # 4. Display Dataset Shape
    # ---------------------------------------------------------
    print("-" * 70)
    print("2. DATASET SHAPE")
    print("-" * 70)
    print(f"Total Rows (Telemetry Segments): {df.shape[0]}")
    print(f"Total Columns: {df.shape[1]}")
    print()

    # ---------------------------------------------------------
    # 5. Display All Column Names
    # ---------------------------------------------------------
    print("-" * 70)
    print("3. ALL COLUMN NAMES")
    print("-" * 70)
    for idx, col in enumerate(df.columns, start=1):
        print(f"  {idx:2d}. {col}")
    print()

    # Define metadata columns (NOT to be used as model input features)
    metadata_cols = ["segment", "channel", "sampling", "anomaly", "train"]
    # The 18 handcrafted features from the OPS-SAT research paper
    feature_cols = [col for col in df.columns if col not in metadata_cols]

    print(f"Identified Metadata/Label columns ({len(metadata_cols)}): {metadata_cols}")
    print(f"Identified Input Feature columns ({len(feature_cols)}): {feature_cols}")
    print()

    # ---------------------------------------------------------
    # 6. Display Data Types
    # ---------------------------------------------------------
    print("-" * 70)
    print("4. COLUMN DATA TYPES")
    print("-" * 70)
    print(df.dtypes)
    print()

    # ---------------------------------------------------------
    # 7. Check Missing Values
    # ---------------------------------------------------------
    print("-" * 70)
    print("5. MISSING VALUES CHECK")
    print("-" * 70)
    missing_series = df.isnull().sum()
    missing_cols = missing_series[missing_series > 0]
    if missing_cols.empty:
        print("No missing values found across any columns (0 nulls).")
    else:
        print("Missing values per column:")
        print(missing_cols)
    print()

    # ---------------------------------------------------------
    # 8. Check Duplicate Records
    # ---------------------------------------------------------
    print("-" * 70)
    print("6. DUPLICATE RECORDS CHECK")
    print("-" * 70)
    duplicates = df.duplicated().sum()
    print(f"Duplicate full-row records found: {duplicates}")
    print()

    # ---------------------------------------------------------
    # 9. Count Normal vs. Anomalous Samples (Target: anomaly)
    # ---------------------------------------------------------
    print("-" * 70)
    print("7. TARGET CLASS DISTRIBUTION (Column: 'anomaly')")
    print("-" * 70)
    # 0 = Nominal (Normal), 1 = Anomalous
    class_counts = df["anomaly"].value_counts().sort_index()
    normal_count = class_counts.get(0, 0)
    anomaly_count = class_counts.get(1, 0)
    total_count = len(df)

    print(f"  Nominal (Normal - Class 0) : {normal_count:5d} ({normal_count / total_count * 100:.2f}%)")
    print(f"  Anomalous (Faulty - Class 1): {anomaly_count:5d} ({anomaly_count / total_count * 100:.2f}%)")
    print(f"  Total Segments              : {total_count:5d}")
    print()

    # ---------------------------------------------------------
    # 10. Count Training vs. Testing Samples (Column: 'train')
    # ---------------------------------------------------------
    print("-" * 70)
    print("8. TRAINING VS. TESTING SPLIT (Column: 'train')")
    print("-" * 70)
    # 1 = Training set (T), 0 = Test set (Psi)
    split_counts = df["train"].value_counts().sort_index(ascending=False)
    train_count = split_counts.get(1, 0)
    test_count = split_counts.get(0, 0)

    print(f"  Training Set (train = 1): {train_count:5d} ({train_count / total_count * 100:.2f}%)")
    print(f"  Testing Set  (train = 0): {test_count:5d} ({test_count / total_count * 100:.2f}%)")
    print(f"  Total Segments          : {total_count:5d}")
    print("\nDetailed cross-tabulation (Split vs Class):")
    cross_tab = pd.crosstab(
        df["train"].map({1: "Training (T)", 0: "Testing (Psi)"}),
        df["anomaly"].map({0: "Nominal", 1: "Anomalous"}),
        margins=True
    )
    print(cross_tab)
    print()

    # ---------------------------------------------------------
    # 11. Display Descriptive Statistics (18 Features)
    # ---------------------------------------------------------
    print("-" * 70)
    print("9. DESCRIPTIVE STATISTICS (18 Handcrafted Features)")
    print("-" * 70)
    pd.set_option("display.max_columns", 10)
    pd.set_option("display.width", 1000)
    desc_stats = df[feature_cols].describe().T
    print(desc_stats[["count", "mean", "std", "min", "25%", "50%", "75%", "max"]])
    print()

    # ---------------------------------------------------------
    # 12. Plot Normal vs. Anomalous Distribution
    # ---------------------------------------------------------
    print("-" * 70)
    print("10. GENERATING PLOTS")
    print("-" * 70)

    class_labels = ["Nominal (0)", "Anomalous (1)"]
    counts = [normal_count, anomaly_count]
    colors = ["#2b5c8f", "#d95f02"]

    plt.figure(figsize=(8, 6), dpi=150)
    bars = plt.bar(class_labels, counts, color=colors, width=0.5, edgecolor="black", linewidth=1.2)
    plt.title("OPS-SAT-AD Target Class Distribution", fontsize=14, pad=15, fontweight="bold")
    plt.ylabel("Number of Telemetry Segments", fontsize=12)
    plt.xlabel("Class", fontsize=12)
    plt.ylim(0, max(counts) * 1.15)
    plt.grid(axis="y", linestyle="--", alpha=0.6)

    # Annotate bar values and percentages
    for bar in bars:
        height = bar.get_height()
        pct = (height / total_count) * 100
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 25,
            f"{int(height)}\n({pct:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold"
        )

    plt.tight_layout()
    class_plot_path = os.path.join(results_dir, "normal_vs_anomalous_distribution.png")
    plt.savefig(class_plot_path)
    plt.close()
    print(f"Saved: {class_plot_path}")

    # ---------------------------------------------------------
    # 13. Plot Training vs. Testing Distribution
    # ---------------------------------------------------------
    # Cross counts for grouped bar chart
    train_nom = len(df[(df["train"] == 1) & (df["anomaly"] == 0)])
    train_anom = len(df[(df["train"] == 1) & (df["anomaly"] == 1)])
    test_nom = len(df[(df["train"] == 0) & (df["anomaly"] == 0)])
    test_anom = len(df[(df["train"] == 0) & (df["anomaly"] == 1)])

    x = np.arange(2)
    width = 0.35

    plt.figure(figsize=(9, 6), dpi=150)
    nom_bars = plt.bar(x - width / 2, [train_nom, test_nom], width, label="Nominal (Normal)", color="#2b5c8f", edgecolor="black")
    anom_bars = plt.bar(x + width / 2, [train_anom, test_anom], width, label="Anomalous", color="#d95f02", edgecolor="black")

    plt.title("OPS-SAT-AD Dataset Split: Training (T) vs. Testing (Psi)", fontsize=14, pad=15, fontweight="bold")
    plt.ylabel("Number of Telemetry Segments", fontsize=12)
    plt.xticks(x, [f"Training Set (T)\nTotal: {train_count}", f"Testing Set (Psi)\nTotal: {test_count}"], fontsize=11)
    plt.legend(frameon=True, fontsize=11)
    plt.ylim(0, max(train_nom, test_nom) * 1.15)
    plt.grid(axis="y", linestyle="--", alpha=0.6)

    # Annotate bars
    for bar in list(nom_bars) + list(anom_bars):
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 20,
            f"{int(height)}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold"
        )

    plt.tight_layout()
    split_plot_path = os.path.join(results_dir, "training_vs_testing_distribution.png")
    plt.savefig(split_plot_path)
    plt.close()
    print(f"Saved: {split_plot_path}")

    print("-" * 70)
    print("Exploratory Data Analysis completed successfully.")
    print("=" * 70)


if __name__ == "__main__":
    main()

