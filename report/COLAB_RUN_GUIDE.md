# Google Colab Execution & Demonstration Guide
## Satellite Telemetry Anomaly Detection Using Machine Learning
**OPS-SAT-AD Benchmark Demonstration Notebook**

---

### Executive Overview

This guide provides end-to-end instructions for running and presenting the Capstone Project demonstration notebook:
```
report/Satellite_Telemetry_Anomaly_Detection_Demo.ipynb
```
The notebook is self-contained, interactive, and pre-configured to run seamlessly on both **Google Colab** (cloud environment) and **local JupyterLab / VS Code** environments. It replicates the complete experimental pipeline:
1. Telemetry exploration and class imbalance characterization.
2. Handcrafted statistical feature extraction verification (18 features).
3. Rigorous benchmark evaluation preserving the canonical Ruszczak et al. (2024) split ($T=1,594$, $\Psi=529$).
4. Supervised vs. Unsupervised anomaly detection comparison (Random Forest, Isolation Forest, XGBoost).
5. Comprehensive diagnostic plots (ROC Curves, PR Curves, Confusion Matrices, Anomaly Score Distributions).
6. Ground-truth vs. Predicted anomaly alignment analysis.
7. Interactive Streamlit Dashboard architectural review.

---

## 1. Prerequisites and Runtime Requirements

- **Platform:** Google Colab (Free tier or Pro tier). A standard CPU runtime is sufficient; GPU is optional (execution takes under 45 seconds on CPU).
- **Python Version:** Python 3.9, 3.10, or 3.11.
- **Required Files:**
  - `Satellite_Telemetry_Anomaly_Detection_Demo.ipynb` (Demonstration notebook)
  - `dataset.csv` (Main extracted features dataset, 2,123 rows × 23 columns)
  - `segments.csv` (Raw continuous time-series telemetry data, 303,493 rows × 8 columns — optional for main ML pipeline, verified in Section 3)
- **Core Dependencies:**
  - `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`, `xgboost`, `joblib`
  *(All pre-installed in Google Colab; the notebook includes an automated verification/install cell in Section 2).*

---

## 2. Step-by-Step Setup in Google Colab

You can run the notebook in Google Colab using either of two file setup methods:

### Method A: Direct Upload to Colab Session Storage (Recommended - Quickest)

This method requires no Google Drive connection and is ideal for quick presentations and examiner demonstrations.

1. **Open Google Colab:**
   - Navigate to [colab.research.google.com](https://colab.research.google.com).
2. **Upload the Notebook:**
   - In the Colab startup dialog, click the **Upload** tab.
   - Click **Browse** and select `report/Satellite_Telemetry_Anomaly_Detection_Demo.ipynb` from your local project folder.
   - The notebook will open in the Colab editor.
3. **Upload Dataset Files:**
   - In the left sidebar of Colab, click the **Folder icon** (📁) to open the Files panel.
   - Click the **Upload to session storage** button (folder with an upward arrow) or drag-and-drop the files.
   - You can upload `dataset.csv` and `segments.csv` in either of the following locations:
     - **Option 1 (Flat upload):** Upload `dataset.csv` and `segments.csv` directly into `/content/`.
     - **Option 2 (Folder structure):** Create a folder named `data` in `/content/` and upload `dataset.csv` and `segments.csv` into `/content/data/`.
   - *Note:* The notebook's Section 3 includes an **intelligent automatic path detector** that checks `data/dataset.csv`, `dataset.csv`, `../data/dataset.csv`, and `/content/data/dataset.csv`. It automatically finds the dataset regardless of where it was placed!

---

### Method B: Google Drive Mounting (Persistent Storage)

This method is ideal if you want your models, metrics, and generated figures automatically saved to your personal Google Drive.

1. **Upload Project to Google Drive:**
   - Create a folder in your Google Drive, for example: `My Drive/Capstone_Project/`.
   - Upload `Satellite_Telemetry_Anomaly_Detection_Demo.ipynb`, `dataset.csv`, and `segments.csv` into this folder.
2. **Open Notebook in Colab:**
   - Right-click `Satellite_Telemetry_Anomaly_Detection_Demo.ipynb` in Google Drive and select **Open with > Google Colaboratory**.
3. **Mount Google Drive in Colab:**
   - Run the optional Drive mount cell in Section 3:
     ```python
     from google.colab import drive
     drive.mount('/content/drive')
     import os
     os.chdir('/content/drive/MyDrive/Capstone_Project')
     ```
   - All relative paths (`data/dataset.csv`, `models/`, `results/`) will now resolve directly against your Drive folder.

---

## 3. Running the Notebook Step-by-Step

### Execution Sequence
1. **Initialize Runtime:**
   - In the Colab menu, click **Runtime > Change runtime type**.
   - Ensure the Hardware accelerator is set to **CPU** (or **T4 GPU** if preferred).
2. **Run All Cells:**
   - To run the entire notebook sequentially, click **Runtime > Run all** (or press `Ctrl + F9`).
   - The total execution time from cell 1 to cell 30 is approximately **30 to 45 seconds**.
3. **Interactive Step-by-Step Execution:**
   - If presenting to an examiner, run each section cell-by-cell using `Shift + Enter`.
   - Each section contains markdown narrative explaining the engineering significance, methodology, mathematical equations, and operational takeaways.

---

## 4. Notebook Section Walkthrough & Presentation Script

Use this structured script to present the notebook to your external guide or project evaluation committee:

| Section | Content & Key Highlights | Talking Points for Examiner |
| :--- | :--- | :--- |
| **Section 1: Project Overview** | Problem statement, ESA OPS-SAT satellite mission context, research objectives. | *"OPS-SAT is ESA's flying laboratory. Detecting telemetry anomalies in real-time prevents catastrophic mission loss. We benchmark against Ruszczak et al. (2024)."* |
| **Section 2: Library Setup** | `sklearn`, `xgboost`, `pandas`, `seaborn`, `matplotlib`. | *"We utilize reproducible ML packages with fixed random seeds (`random_state=42`) ensuring identical metrics across machines."* |
| **Section 3: Dataset Loading** | Automated multi-path resolver for `dataset.csv` and `segments.csv`. | *"The dataset contains 2,123 segmented events across 23 features, while the raw telemetry has 303,493 time-series samples across 7 channels."* |
| **Section 4: EDA** | Class imbalance visualization (20.44% anomalies), channel distributions, sampling rates. | *"Extreme class imbalance is characteristic of space missions where anomalies are rare. The benchmark establishes an official 75/25 split ($T=1,594$, $\Psi=529$)."* |
| **Section 5: Feature Prep** | Selection of 18 handcrafted features (12 raw statistics, 2 smoothed, 4 derivative). Exclusion of metadata. | *"We strictly isolate the 18 statistical features and exclude metadata (`segment`, `channel`, `sampling`, `train`) to prevent data leakage."* |
| **Section 6: Random Forest** | Supervised ensemble with `class_weight='balanced'`. 100 decision trees. | *"Random Forest achieves 98.00% precision with an MCC of 0.9028, minimizing false alarms which cause operational fatigue in satellite mission control."* |
| **Section 7: Isolation Forest** | Unsupervised anomaly isolation with `contamination=0.20`. Trained purely without labels. | *"Isolation Forest models true operational scenarios where novel anomaly classes have no training labels. It achieves 69.19% accuracy and 0.6325 ROC-AUC."* |
| **Section 8: XGBoost** | Gradient-boosted trees with `scale_pos_weight=3.9657`. | *"XGBoost delivers the highest recall (90.27%), detecting 102 out of 113 anomalous events on the unseen test partition."* |
| **Section 9: Model Comparison** | Multi-metric comparative table, grouped bar charts, ROC and PR curves. | *"Random Forest leads in Precision, F1-Score, MCC, and ROC-AUC. XGBoost leads in Recall. Both significantly outperform unsupervised Isolation Forest."* |
| **Section 10: Anomaly Scores** | Continuous anomaly score distributions, confidence histograms, and thresholding. | *"Continuous scores allow flight operators to tune alert thresholds dynamically based on orbital risk posture."* |
| **Section 11: Confusion Matrix** | Normalized and raw confusion matrices, error breakdown ($FP$ vs $FN$). | *"Random Forest produced only 2 false positives out of 416 nominal test cases, making it exceptionally reliable for automated onboard alerting."* |
| **Section 12: Streamlit Dashboard** | Architectural review of `dashboard/app.py` and instructions for local execution. | *"For operations, we developed an interactive Streamlit UI allowing telemetry filtering, probability inspection, and automated diagnostic reports."* |

---

## 5. Summary of Benchmark Results

The notebook reproduces these exact metrics on the unseen test set ($\Psi = 529$ records):

```
====================================================================================================
                        SATELLITE TELEMETRY ANOMALY DETECTION BENCHMARK RESULTS
====================================================================================================
Metric                  Random Forest (Supervised)   XGBoost (Supervised)   Isolation Forest (Unsupervised)   Best Model
----------------------------------------------------------------------------------------------------
Accuracy                0.9679                       0.9603                 0.6919                            Random Forest
Precision               0.9800                       0.9107                 0.2768                            Random Forest
Recall                  0.8673                       0.9027                 0.2743                            XGBoost
F1-Score                0.9202                       0.9067                 0.2756                            Random Forest
MCC                     0.9028                       0.8815                 0.0799                            Random Forest
ROC-AUC                 0.9903                       0.9897                 0.6325                            Random Forest
Average Precision (AP)  0.9728                       0.9705                 0.3737                            Random Forest
====================================================================================================
```

### Confusion Matrix Breakdown ($\Psi = 529$ Test Samples, 416 Nominal, 113 Anomalous)
- **Random Forest:** $TN = 414, \ FP = 2, \ FN = 15, \ TP = 98$
- **XGBoost:** $TN = 406, \ FP = 10, \ FN = 11, \ TP = 102$
- **Isolation Forest:** $TN = 335, \ FP = 81, \ FN = 82, \ TP = 31$

---

## 6. How to Run the Streamlit Dashboard

As clarified in Section 12 of the notebook, the interactive **Streamlit Dashboard** (`dashboard/app.py`) is designed as a standalone operational application.

### Running Locally (on your laptop / workstation)
1. Open PowerShell or Command Prompt in the project root directory.
2. Ensure dependencies are installed:
   ```bash
   pip install streamlit plotly pandas numpy scikit-learn xgboost joblib
   ```
3. Launch the dashboard:
   ```bash
   streamlit run dashboard/app.py
   ```
4. The dashboard will automatically open in your default browser at `http://localhost:8501`.

### Running Inside Google Colab (Optional Tunneling)
If your examiner asks to view the live dashboard directly from Colab:
1. Create a new code cell at the bottom of the Colab notebook:
   ```python
   !pip install -q streamlit localtunnel
   !streamlit run dashboard/app.py &>/content/logs.txt &
   !npx localtunnel --port 8501
   ```
2. Click the public `localtunnel.me` URL generated in the cell output to view the live interactive web dashboard.

---

## 7. Troubleshooting & Frequently Asked Questions

### Q1: Colab displays `FileNotFoundError: dataset.csv not found`
- **Cause:** `dataset.csv` has not been uploaded to the Colab session storage.
- **Fix:** Click the folder icon on the left sidebar, click the upload icon, and select `dataset.csv` from your computer. Re-run Section 3. The built-in path resolver will automatically detect it.

### Q2: Why are the metrics slightly different if I change `random_state`?
- **Answer:** Decision tree partitioning and subsampling involve pseudo-random processes. Setting `random_state=42` guarantees exact reproducibility across all machines, matching the research paper's findings.

### Q3: Why is Isolation Forest's precision lower than Random Forest?
- **Answer:** Isolation Forest is completely **unsupervised**; it never sees the ground truth anomaly labels during training. It isolates points based solely on tree partition depth. In contrast, Random Forest and XGBoost are **supervised**, learning complex non-linear decision boundaries directly from ground-truth examples.

---
*Prepared for B.Tech Capstone Project Examination & Defense.*

