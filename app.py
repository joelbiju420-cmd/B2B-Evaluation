import streamlit as st
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

st.set_page_config(page_title="B2B Client Risk Dashboard", layout="wide")

st.title("B2B Client Risk & Churn Prediction Dashboard")

# -------------------------------
# Load Dataset
# -------------------------------

data = pd.read_csv("B2B_Client_Churn_5000.csv")

st.subheader("Dataset Preview")
st.dataframe(data.head())

# -------------------------------
# Risk Score Logic
# -------------------------------

def calculate_risk(row):

    score = 0

    if row["Payment_Delay_Days"] > 30:
        score += 3

    if row["Monthly_Usage_Score"] < 50:
        score += 2

    if row["Contract_Length_Months"] < 12:
        score += 2

    if row["Support_Tickets_Last30Days"] > 5:
        score += 2

    return score


data["Risk_Score"] = data.apply(calculate_risk, axis=1)

# -------------------------------
# Risk Category
# -------------------------------

def risk_category(score):

    if score >= 6:
        return "High"

    elif score >= 3:
        return "Medium"

    else:
        return "Low"


data["Risk_Category"] = data["Risk_Score"].apply(risk_category)

# -------------------------------
# Sidebar Filters
# -------------------------------

st.sidebar.header("Filters")

region = st.sidebar.selectbox("Region", data["Region"].unique())

industry = st.sidebar.selectbox("Industry", data["Industry"].unique())

risk = st.sidebar.selectbox("Risk Category", data["Risk_Category"].unique())

filtered_data = data[
    (data["Region"] == region)
    & (data["Industry"] == industry)
    & (data["Risk_Category"] == risk)
]

# -------------------------------
# KPI Metrics
# -------------------------------

total_clients = len(data)

high_risk = len(data[data["Risk_Category"] == "High"])

churn_rate = (data["Renewal_Status"] == "No").mean() * 100

avg_revenue = data["Monthly_Revenue_USD"].mean()

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Clients", total_clients)

col2.metric("High Risk Clients", high_risk)

col3.metric("Churn Rate %", round(churn_rate, 2))

col4.metric("Average Revenue", round(avg_revenue, 2))

# -------------------------------
# Risk Distribution Chart
# -------------------------------

st.subheader("Risk Category Distribution")

fig, ax = plt.subplots()

data["Risk_Category"].value_counts().plot(kind="bar", ax=ax)

st.pyplot(fig)

# -------------------------------
# Industry Risk Analysis
# -------------------------------

st.subheader("Industry-wise Risk Analysis")

fig, ax = plt.subplots()

sns.countplot(data=data, x="Industry", hue="Risk_Category")

plt.xticks(rotation=45)

st.pyplot(fig)

# -------------------------------
# Revenue vs Usage Scatter
# -------------------------------

st.subheader("Revenue vs Usage")

fig, ax = plt.subplots()

sns.scatterplot(
    data=data,
    x="Monthly_Revenue_USD",
    y="Monthly_Usage_Score",
    hue="Risk_Category"
)

st.pyplot(fig)

# -------------------------------
# Machine Learning Model
# -------------------------------

st.subheader("Machine Learning: Churn Prediction")

ml_data = data.copy()

ml_data["Renewal_Status"] = ml_data["Renewal_Status"].map({"Yes": 1, "No": 0})

ml_data = pd.get_dummies(ml_data, columns=["Industry", "Region", "Plan"], drop_first=True)

X = ml_data.drop(
    ["Client_ID", "Company_Name", "Renewal_Status", "Last_Renewal_Date", "Risk_Category"],
    axis=1
)

y = ml_data["Renewal_Status"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = DecisionTreeClassifier()

model.fit(X_train, y_train)

pred = model.predict(X_test)

accuracy = accuracy_score(y_test, pred)

st.write("Model Accuracy:", round(accuracy, 3))

# -------------------------------
# Confusion Matrix
# -------------------------------

cm = confusion_matrix(y_test, pred)

fig, ax = plt.subplots()

sns.heatmap(cm, annot=True, fmt="d")

st.pyplot(fig)

# -------------------------------
# Feature Importance
# -------------------------------

st.subheader("Feature Importance")

importance = pd.Series(model.feature_importances_, index=X.columns)

st.bar_chart(importance)

# -------------------------------
# Top High Risk Clients
# -------------------------------

st.subheader("Top 20 High Risk Clients")

high_clients = data[data["Risk_Category"] == "High"]

st.dataframe(high_clients.head(20))

# -------------------------------
# Retention Strategy
# -------------------------------

if st.button("Generate Retention Strategy"):

    st.write("### Suggested Strategies")

    st.write("• Offer discount for delayed payments")

    st.write("• Assign dedicated account managers")

    st.write("• Provide training to increase product usage")

    st.write("• Offer long-term contract incentives")

    st.write("• Improve customer support response time")

# -------------------------------
# Responsible AI
# -------------------------------

st.subheader("Ethical Implications of Predicting Client Churn")

st.write("""
• Machine learning models may contain bias depending on the training data.

• Labeling customers as 'High Risk' can influence business decisions unfairly.

• Client data must be protected to ensure privacy.

• AI predictions should support human decision-making, not replace it.
""")
