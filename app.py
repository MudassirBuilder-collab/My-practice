import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Preprocessing vs Postprocessing (KNN)", layout="wide")

st.title("📊 Preprocessing vs Postprocessing for KNN Model")
st.write("This app demonstrates how data preprocessing impacts KNN performance.")

# ==========================
# LOAD DATA
# ==========================
df_raw = pd.read_csv(r"C:\Users\L4-PC23\Downloads\customer_lifestyle_300_rows - customer_lifestyle_300_rows.csv")
#df_raw = pd.read_csv(r"C:\Users\dell\Downloads\customer_lifestyle_300_rows.csv")
df_raw = pd.read_csv(r"C:\Users\L4-PC23\Downloads\customer_lifestyle_300_rows - customer_lifestyle_300_rows.csv")


# ==========================
# RAW DATA SECTION
# ==========================
st.header("🔴 RAW DATA (Before Preprocessing)")

st.subheader("Raw Dataset Preview")
st.dataframe(df_raw.head(10))

st.subheader("Missing Values (Raw Data)")
st.write(df_raw.isnull().sum())

fig1, ax1 = plt.subplots(figsize=(8,4))
sns.heatmap(df_raw.isnull(), cbar=False, ax=ax1)
ax1.set_title("Missing Values Heatmap (RAW DATA)")
st.pyplot(fig1)

st.subheader("Raw Feature Distributions (Unscaled)")
fig2 = df_raw[['age','monthly_income','steps_per_day','bmi']].hist(
    figsize=(8,6), bins=20
)
plt.suptitle("RAW Numerical Features")
st.pyplot(plt)

# ==========================
# PREPROCESSING
# ==========================
df_clean = df_raw.copy()

text_cols = ['gender','exercise_level','location','purchase_decision']
for col in text_cols:
    df_clean[col] = df_clean[col].str.lower().str.strip()

num_cols = ['age','monthly_income','steps_per_day','bmi']
cat_cols = ['gender','exercise_level','location']

for col in num_cols:
    df_clean[col].fillna(df_clean[col].median(), inplace=True)

for col in cat_cols:
    df_clean[col].fillna(df_clean[col].mode()[0], inplace=True)

encoder = LabelEncoder()
for col in cat_cols + ['purchase_decision']:
    df_clean[col] = encoder.fit_transform(df_clean[col])

X = df_clean.drop('purchase_decision', axis=1)
y = df_clean['purchase_decision']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ==========================
# POSTPROCESSED DATA
# ==========================
st.header("🟢 POSTPROCESSED DATA (After Preprocessing)")

df_processed = pd.DataFrame(X_scaled, columns=X.columns)

st.subheader("Processed Data Preview")
st.dataframe(df_processed.head(10))

st.subheader("Missing Values (After Cleaning)")
st.write(df_processed.isnull().sum())

fig3, ax3 = plt.subplots(figsize=(8,4))
sns.heatmap(df_processed.isnull(), cbar=False, ax=ax3)
ax3.set_title("No Missing Values (POSTPROCESSED DATA)")
st.pyplot(fig3)

st.subheader("Scaled Feature Distributions")
fig4 = df_processed.hist(figsize=(8,6), bins=20)
plt.suptitle("POSTPROCESSED (Scaled) Features")
st.pyplot(plt)

# ==========================
# MODEL COMPARISON
# ==========================
st.header("⚖️ KNN Performance Comparison")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

knn_raw = KNeighborsClassifier(n_neighbors=5)
knn_raw.fit(X_train, y_train)
pred_raw = knn_raw.predict(X_test)
acc_raw = accuracy_score(y_test, pred_raw)

X_train_s, X_test_s, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.25, random_state=42
)

knn_scaled = KNeighborsClassifier(n_neighbors=5)
knn_scaled.fit(X_train_s, y_train)
pred_scaled = knn_scaled.predict(X_test_s)
acc_scaled = accuracy_score(y_test, pred_scaled)

st.metric("Accuracy WITHOUT Preprocessing", round(acc_raw, 3))
st.metric("Accuracy WITH Preprocessing", round(acc_scaled, 3))

# ==========================
# SUMMARY SECTION
# ==========================
st.header("📝 Summary: Preprocessing vs Postprocessing")

st.markdown("""
### 📌 Key Explanation

**Preprocessing** is the stage where raw data is cleaned and transformed before applying a machine learning algorithm.  
In this project, preprocessing includes:
- Handling missing values  
- Fixing inconsistent categorical text  
- Encoding categorical variables  
- Scaling numerical features  

**Postprocessing** refers to analyzing the cleaned and transformed data and evaluating the model results.  
This includes:
- Verifying that no missing values remain  
- Visualizing scaled features  
- Comparing model accuracy  

### 🎯 Why Preprocessing Matters for KNN
KNN relies on distance calculations. Without feature scaling, variables with large values (like income) dominate the distance calculation, leading to poor predictions. After preprocessing, all features contribute fairly, resulting in improved model performance.

**Conclusion:**  
Preprocessing is a critical step that directly improves the accuracy, reliability, and fairness of KNN predictions.
""")

st.success("✅ End-to-End demonstration completed successfully!")
