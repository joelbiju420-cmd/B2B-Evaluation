import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import os

# -----------------------------
# Page Setup
# -----------------------------
st.set_page_config(page_title="B2B Client Risk Dashboard", layout="wide")

st.title("B2B Client Risk & Churn Prediction System")

# -----------------------------
# Load Dataset Safely
# -----------------------------
@st.cache_data
def load_data():

    file_path = "B2B_Client_Churn_5000.csv"

    if not os.path.exists(file_path):
        st.error("CSV file not found. Upload 'B2B_Client_Churn_5000.csv' to the project folder.")
        st.stop()

    df = pd.read_csv(file_path)
    return df


df = load_data()

# -----------------------------
# Basic Data Preview
# -----------------------------
st.subheader("Dataset Preview")
st.dataframe(df.head())

# -----------------------------
# Create Churn Variable
# -----------------------------
df["Churned"] = df["Renewal_Status"].map({"Yes": 0, "No": 1})

# -----------------------------
# Risk Scoring Logic
# -----------------------------
def calculate_risk(row):

    score = 0

    if row["Payment_Delay_Days"] > 30:
        score += 3

    if row["Monthly_Usage_Score"] < 40:
        score += 3

    if row["Contract_Length_Months"] < 6:
        score += 2

    if row["Support_Tickets_Last30Days"] > 5:
        score += 2

    return score


df["Risk_Score"] = df.apply(calculate_risk, axis=1)

def risk_category(score):

    if score >= 6:
        return "High Risk"
    elif score >= 3:
        return "Medium Risk"
    else:
        return "Low Risk"


df["Risk_Category"] = df["Risk_Score"].apply(risk_category)

# -----------------------------
# Risk Distribution Chart
# -----------------------------
st.subheader("Client Risk Distribution")

risk_counts = df["Risk_Category"].value_counts()

fig = plt.figure()
plt.bar(risk_counts.index, risk_counts.values)
plt.xlabel("Risk Category")
plt.ylabel("Number of Clients")

st.pyplot(fig)

# -----------------------------
# Revenue vs Risk Chart
# -----------------------------
st.subheader("Revenue vs Risk Score")

fig2 = plt.figure()
plt.scatter(df["Risk_Score"], df["Monthly_Revenue_USD"])
plt.xlabel("Risk Score")
plt.ylabel("Monthly Revenue")

st.pyplot(fig2)

# -----------------------------
# Machine Learning Model
# -----------------------------
st.subheader("Churn Prediction Model")

features = [
    "Monthly_Usage_Score",
    "Payment_Delay_Days",
    "Contract_Length_Months",
    "Support_Tickets_Last30Days"
]

X = df[features]
y = df["Churned"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

model = DecisionTreeClassifier()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

st.write("Model Accuracy:", round(accuracy * 100, 2), "%")

# -----------------------------
# Confusion Matrix
# -----------------------------
st.subheader("Confusion Matrix")

cm = confusion_matrix(y_test, predictions)

st.write(cm)

# -----------------------------
# Feature Importance
# -----------------------------
st.subheader("Feature Importance")

importance = pd.Series(model.feature_importances_, index=features)

st.bar_chart(importance)

# -----------------------------
# Retention Strategy Generator
# -----------------------------
st.subheader("AI Retention Strategy")

if st.button("Generate Retention Strategy"):

    st.write("1. Offer flexible payment plans for delayed clients.")
    st.write("2. Provide engagement training for low-usage clients.")
    st.write("3. Assign account managers to high complaint clients.")
    st.write("4. Offer long-term contract discounts.")
    st.write("5. Conduct quarterly relationship review meetings.")

# -----------------------------
# High Risk Client Table
# -----------------------------
st.subheader("Top High Risk Clients")

high_risk = df[df["Risk_Category"] == "High Risk"].sort_values(
    by="Monthly_Revenue_USD", ascending=False
)

st.dataframe(high_risk.head(10))

# -----------------------------
# Download Data
# -----------------------------
st.subheader("Download Filtered Data")

csv = df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download CSV",
    data=csv,
    file_name="client_data.csv",
    mime="text/csv"
)
