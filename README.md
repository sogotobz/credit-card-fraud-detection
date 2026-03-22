# credit-card-fraud-detection
A machine learning project detecting fraudulent credit card transactions.
🔍 Overview
This project analyzes an anonymized credit card transaction dataset and builds machine learning models to detect fraudulent activity. Because fraud cases are extremely rare, the project focuses on handling class imbalance, understanding transaction patterns, and evaluating models using metrics that matter in real‑world fraud detection, such as recall and ROC–AUC.

The goal is to create a clear, interpretable, and effective fraud‑detection pipeline that demonstrates both technical skill and analytical reasoning.

Project Structure
credit-card-fraud-detection/
│
├── Credit_card_fraud_detection.ipynb   # Main notebook with full analysis
├── Credit_card_fraud_detection.py      # Python script version
├── requirements.txt                    # Project dependencies
├── README.md                           # Project documentation
├── roc curve.png                       # ROC curve visualization
├── fraud distribution.png              # Class distribution plot
├── transaction amount by class.png     # Amount by class visualization
├── Scatterplot of time vs amount.png   # Time vs amount scatterplot
├── Distribution of transaction amount.png
├── kernel density stimiton of amount.png
└── ...

Dataset
The dataset contains anonymized credit card transactions with the following fields:

Time — Seconds elapsed since the first transaction

Amount — Transaction amount

V1–V28 — PCA‑transformed features (anonymized)

Class — Target variable

0 = Non‑fraudulent

1 = Fraudulent

Fraud cases represent less than 1% of all transactions, making this a highly imbalanced classification problem.

🧪 Methods and Workflow
1. Exploratory Data Analysis (EDA)
Class distribution visualization

Transaction amount analysis

Time‑based patterns

Scatter plots of fraud vs non‑fraud

Correlation heatmap

KDE plots for transaction amounts

2. Modeling Approach
Two models were trained and compared:

Logistic Regression (with class weighting)
Serves as a simple, interpretable baseline

Uses class_weight='balanced' to address imbalance

XGBoost (with scale_pos_weight)
Handles non‑linear patterns

Strong performance on tabular data

Tuned to emphasize minority fraud cases

3. Evaluation Metrics
Because accuracy is misleading in imbalanced datasets, the project focuses on:

Recall (Sensitivity) — ability to catch fraud

Precision — correctness of fraud predictions

F1‑Score — balance of precision and recall

Confusion Matrix

ROC Curve & AUC

Conclusion
This project demonstrates how machine learning can be applied to detect rare fraudulent transactions in a highly imbalanced dataset. Logistic Regression offers interpretability and a solid baseline, while XGBoost delivers superior recall and ROC–AUC performance.

The results highlight a key reality of fraud detection:
Catching more fraud (high recall) often means accepting more false positives.  
In real‑world systems, this trade‑off is tuned based on financial risk tolerance.

This project provides a strong foundation for further enhancements such as:

Threshold tuning

SMOTE oversampling

SHAP explainability

Deployment as an API or scoring pipeline
