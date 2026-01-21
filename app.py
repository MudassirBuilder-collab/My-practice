# app.py
# ---------------------------------------------
# COVID Prediction Web App (Final Working Version)
# KNN + SMOTE + Best K + Advanced Visualizations
# Written in human-style clean code
# ---------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from imblearn.over_sampling import SMOTE

# Make plots look nicer
sns.set_style("whitegrid")

# ---------------------------------------------
# Page Configuration
# ---------------------------------------------
st.set_page_config(page_title="COVID Predictor App", layout="wide")

# ---------------------------------------------
# Load & Process Data
# ---------------------------------------------
@st.cache_data
def load_data():
    raw = pd.read_csv("patient.csv")
    
    # Only required columns
    df = raw[["sex", "age", "pneumonia", "diabetes", "asthma", "outcome"]].copy()
    
    # Encode features: 1 = Yes, 0 = No
    df["sex"] = df["sex"].map({1: 1, 2: 0})
    df["pneumonia"] = df["pneumonia"].map({1: 1, 2: 0})
    df["diabetes"] = df["diabetes"].map({1: 1, 2: 0})
    df["asthma"] = df["asthma"].map({1: 1, 2: 0})
    
    # Outcome: 1 = Positive, 0 = Negative
    df["outcome"] = df["outcome"].map({1: 1, 2: 0})
    
    return raw, df.dropna()

raw_df, df = load_data()

# ---------------------------------------------
# Feature / Target
# ---------------------------------------------
X = df.drop("outcome", axis=1)
y = df["outcome"]

scaler = StandardScaler()
X["age"] = scaler.fit_transform(X[["age"]])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# Apply SMOTE to training set
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# ---------------------------------------------
# Find Best K using Cross Validation
# ---------------------------------------------
k_scores = {}
for k in range(3, 21, 2):
    model = KNeighborsClassifier(n_neighbors=k, weights="distance")
    scores = cross_val_score(model, X_train_res, y_train_res, cv=5)
    k_scores[k] = scores.mean()

best_k = max(k_scores, key=k_scores.get)

# ---------------------------------------------
# Train Final Model
# ---------------------------------------------
knn = KNeighborsClassifier(n_neighbors=best_k, weights="distance")
knn.fit(X_train_res, y_train_res)

y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# ---------------------------------------------
# UI Tabs
# ---------------------------------------------
tab1, tab2 = st.tabs(["📊 Model Analysis", "🧪 Check Yourself"])

# =================================================
# TAB 1 — MODEL ANALYSIS
# =================================================
with tab1:
    st.title("📊 COVID Prediction Model Analysis")
    
    st.subheader("Raw Data (First 10 Rows)")
    st.dataframe(raw_df.head(10))
    
    st.subheader("Processed Data (After Encoding)")
    st.dataframe(df.head(10))
    
    st.metric("Model Accuracy", f"{accuracy*100:.2f}%")
    st.write(f"Best K Selected Automatically: **{best_k}**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Accuracy Pie Chart
        fig, ax = plt.subplots()
        ax.pie(
            [accuracy, 1 - accuracy],
            labels=["Correct", "Wrong"],
            autopct="%1.1f%%",
            startangle=90,
            explode=[0.05, 0],
            colors=["#4CAF50", "#FF5252"]
        )
        ax.set_title("Prediction Accuracy Breakdown")
        st.pyplot(fig)
    
    with col2:
        # Confusion Matrix Heatmap
        fig, ax = plt.subplots()
        sns.heatmap(
            conf_matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Negative", "Positive"],
            yticklabels=["Negative", "Positive"],
            ax=ax
        )
        ax.set_title("Confusion Matrix")
        st.pyplot(fig)
    
    # K Optimization Curve
    st.subheader("Cross-Validation Accuracy vs K")
    fig, ax = plt.subplots()
    ax.plot(list(k_scores.keys()), list(k_scores.values()), marker="o", linewidth=2)
    ax.set_xlabel("K Value")
    ax.set_ylabel("CV Accuracy")
    ax.set_title("Finding Best K for KNN")
    st.pyplot(fig)
    
    # Age vs COVID Probability (Logistic Regression Trend)
    st.subheader("Age vs COVID Outcome Trend")
    try:
        import statsmodels.api as sm
        fig, ax = plt.subplots()
        sns.regplot(
            x=df["age"],
            y=df["outcome"],
            logistic=True,
            scatter_kws={"alpha":0.3},
            line_kws={"color": "red"},
            ax=ax
        )
        ax.set_xlabel("Age")
        ax.set_ylabel("Probability of COVID Positive")
        st.pyplot(fig)
    except:
        st.warning("Install `statsmodels` to show logistic regression trend: pip install statsmodels")
    
    # Outcome Distribution
    st.subheader("Outcome Distribution")
    fig, ax = plt.subplots()
    df["outcome"].value_counts().plot(kind="bar", color=["#2196F3","#FF5722"], ax=ax)
    ax.set_xticklabels(["Negative", "Positive"], rotation=0)
    ax.set_ylabel("Number of Patients")
    ax.set_title("COVID Outcome Distribution")
    st.pyplot(fig)

# =================================================
# TAB 2 — LIVE PREDICTION
# =================================================
with tab2:
    st.title("🧪 Check Yourself / COVID Prediction")
    
    sex = st.selectbox("Sex", ["Male", "Female"])
    age = st.slider("Age", 1, 100, 30)
    pneumonia = st.selectbox("Pneumonia", ["Yes", "No"])
    diabetes = st.selectbox("Diabetes", ["Yes", "No"])
    asthma = st.selectbox("Asthma", ["Yes", "No"])
    
    input_df = pd.DataFrame({
        "sex": [1 if sex=="Male" else 0],
        "age": [age],
        "pneumonia": [1 if pneumonia=="Yes" else 0],
        "diabetes": [1 if diabetes=="Yes" else 0],
        "asthma": [1 if asthma=="Yes" else 0]
    })
    
    input_df["age"] = scaler.transform(input_df[["age"]])
    
    if st.button("Predict"):
        pred = knn.predict(input_df)[0]
        probs = knn.predict_proba(input_df)[0]
        
        # Probability Bar Graph
        fig, ax = plt.subplots()
        ax.bar(["Negative","Positive"], probs, color=["#2196F3","#FF5722"])
        ax.set_ylim(0,1)
        ax.set_ylabel("Probability")
        ax.set_title("Prediction Probability")
        st.pyplot(fig)
        
        if pred == 1:
            st.error(f"⚠️ COVID Positive ({probs[1]*100:.2f}% probability)")
        else:
            st.success(f"✅ COVID Negative ({probs[0]*100:.2f}% confidence)")
