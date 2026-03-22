# Credit Card Fraud Detection EDA and Modeling

# 1. Import Libraries and Setup

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 
import plotly.express as px
from sklearn.metrics import roc_curve, auc
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, classification_report
 




# 2. Load the dataset and data exploration
df = pd.read_csv(r"C:\Users\sogot\OneDrive\Desktop\creditcard fraud\creditcard.csv")
print(df.head())

print(df.columns)
print(df.describe())
print(df.info())
print(df['Class'].value_counts())


# 3. Data Visualization: Fraud Distribution and Amount
sns.countplot(x='Class', data=df)
plt.title('Distribution of Fraudulent and Non-Fraudulent Transactions')
plt.xlabel('Class (0: Non-Fraudulent, 1: Fraudulent)')
plt.ylabel('Count')
plt.show()

sns.boxplot(x='Class', y='Amount', data=df)
plt.title('Transaction Amount by Class')
plt.xlabel('Class (0: Non-Fraudulent, 1: Fraudulent)')
plt.ylabel('Amount')
plt.show()

sns.histplot(data=df, x='Time', hue='Class', bins=50, kde=True)
plt.title('Distribution of Transactions by Time and Class')
plt.xlabel('Time')
plt.ylabel('Count')
plt.show()

# 4. Scatter and Sample Plots: Fraud vs Non-Fraud
fraud=df[df['Class'] == 1]
non_fraud=df[df['Class'] == 0]
plt.figure(figsize=(10,6))
sns.scatterplot(x='Time', y='Amount', data=fraud, color='red', label='Fraudulent')
sns.scatterplot(x='Time', y='Amount', data=non_fraud, color='blue', label='Non-Fraudulent')
plt.xlabel('Time')
plt.ylabel('Amount')
plt.title('Scatter Plot of Time vs Amount for Fraudulent and Non-Fraudulent Transactions')
plt.show()

non_fraud_sample = non_fraud.sample(n=len(fraud), random_state=42)
plt.figure(figsize=(10,6))
sns.scatterplot(x='Time', y='Amount', data=fraud, color='red', label='Fraudulent')
sns.scatterplot(x='Time', y='Amount', data=non_fraud_sample, color='blue', label='Non-Fraudulent')
plt.xlabel('Time')
plt.ylabel('Amount')
plt.title('Scatter Plot of Time vs Amount for Fraudulent and Sampled Non-Fraudulent Transactions')
plt.legend()
plt.show()

# 5. Correlation and KDE Plots
plt.figure(figsize=(10,6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()  

plt.figure(figsize=(11,7))
sns.kdeplot(data=df[df['Class'] == 1], x='Amount', fill=True, label='Fraudulent', color='red', alpha=0.5)
sns.kdeplot(data=df[df['Class'] == 0], x='Amount', fill=True, label='Non-Fraudulent', color='blue', alpha=0.5)
plt.xlabel('Amount')
plt.ylabel('Density')
plt.title('Kernel Density Estimation of Amount for Fraudulent and Non-Fraudulent Transactions')
plt.legend()
plt.show()

# 6. Train-Test Split and Logistic Regression Model
x=df.drop('Class', axis=1)
y=df['Class']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42,stratify=y)
model = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
model.fit(x_train, y_train)
lr_prob = model.predict_proba(x_test)[:, 1]
lr_pred = (lr_prob >= 0.5).astype(int)

lr_fpr, lr_tpr, lr_thresholds = roc_curve(y_test, lr_prob)
lr_roc_auc = auc(lr_fpr, lr_tpr)




print("Logistic Regression balanced Performance:")
print(classification_report(y_test, lr_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, lr_pred))
print("Accuracy:", accuracy_score(y_test, lr_pred))
print("Precision:", precision_score(y_test, lr_pred))
print("Recall:", recall_score(y_test, lr_pred))
print("X_train shape:", x_train.shape)
print("y_train shape:", y_train.shape)
print("X_test shape:", x_test.shape)
print("y_test shape:", y_test.shape)   

# 7. XGBoost Model and ROC Curve Comparison
scale= len(y_train[y_train==1])/len(y_train)

xgb = XGBClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric='logloss',
    scale_pos_weight=1/scale
)

xgb.fit(x_train, y_train)
xgb_prob = xgb.predict_proba(x_test)[:, 1]
xgb_pred = (xgb_prob >= 0.5).astype(int)
xgb_fpr, xgb_tpr, xgb_thresholds = roc_curve(y_test, xgb_prob)
xgb_roc_auc = auc(xgb_fpr, xgb_tpr)

plt.figure(figsize=(8, 6))
plt.plot(lr_fpr, lr_tpr, color='green', label=f'Logistic Regression (AUC = {lr_roc_auc:.2f})')
plt.plot(xgb_fpr, xgb_tpr, color='blue', label=f'XGBoost (AUC = {xgb_roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='red', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

print("XGBoost Results:")
print(classification_report(y_test, xgb_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, xgb_pred))
print("ROC AUC:", xgb_roc_auc)

