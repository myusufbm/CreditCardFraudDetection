# main.py
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

# ---------------------------
# Streamlit App Configuration
# ---------------------------
st.set_page_config(
    page_title="Credit Card Fraud Detection App",
    layout="centered"
)

# ---------------------------
# App Title and Description
# ---------------------------
st.title("💳 Credit Card Fraud Detection App")
st.markdown("""
Upload your **credit card transactions CSV** (same format as the training dataset) to train and evaluate a fraud detection model **on the fly**.

Example dataset: [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
""")

# ---------------------------
# File Uploader
# ---------------------------
uploaded_file = st.file_uploader("📤 Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    # ---------------------------
    # Data Loading
    # ---------------------------
    st.subheader("📌 Data Preview")
    data = pd.read_csv(uploaded_file)
    st.dataframe(data.head())

    # ---------------------------
    # Check for 'Class' Column
    # ---------------------------
    if 'Class' not in data.columns:
        st.error("❗ ERROR: The uploaded CSV must include a 'Class' column for supervised training.")
        st.stop()

    # ---------------------------
    # Exploratory Data Analysis
    # ---------------------------
    st.subheader("🔎 Class Distribution (Before Balancing)")
    class_counts = data['Class'].value_counts()
    st.write(class_counts)

    fig, ax = plt.subplots()
    sns.countplot(x='Class', data=data, ax=ax)
    ax.set_title('Class Distribution')
    st.pyplot(fig)

    # ---------------------------
    # Feature / Target Split
    # ---------------------------
    X = data.drop('Class', axis=1)
    y = data['Class']

    # ---------------------------
    # Handle Imbalance with SMOTE
    # ---------------------------
    st.subheader("⚖️ Balancing Data with SMOTE")
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X, y)

    balanced_counts = pd.Series(y_res).value_counts()
    st.success(f"✅ Data balanced! Class counts after SMOTE:\n{balanced_counts.to_dict()}")

    # ---------------------------
    # Split Data
    # ---------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X_res, y_res, test_size=0.2, random_state=42
    )

    # ---------------------------
    # Train Model
    # ---------------------------
    st.subheader("🧠 Training Random Forest Classifier")
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    st.success("✅ Model training complete!")

    # ---------------------------
    # Make Predictions
    # ---------------------------
    y_pred = clf.predict(X_test)

    # ---------------------------
    # Evaluation Metrics
    # ---------------------------
    st.subheader("📈 Model Evaluation")

    st.markdown("**Classification Report:**")
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df)

    st.markdown("**Confusion Matrix:**")
    cm = confusion_matrix(y_test, y_pred)
    fig2, ax2 = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2)
    ax2.set_xlabel('Predicted')
    ax2.set_ylabel('Actual')
    st.pyplot(fig2)

    st.success("✅ Evaluation complete!")
else:
    st.info("👈 Please upload a CSV file to get started.")
