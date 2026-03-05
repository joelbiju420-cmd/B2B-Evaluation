import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="B2B Client Risk Dashboard", layout="wide")

st.title("B2B Client Risk & Churn Prediction Dashboard")

# -----------------------------
# Load Data
# -----------------------------

data = pd.read_csv("B2B_Client_Churn_5000.csv")

# -----------------------------
# Risk Score Logic
# -----------------------------

def calculate_risk(row):

    score = 0

    if row["Payment_Delay_Days"] > 30:
        score += 3

    if row["Monthly_Usage"] < 50:
        score += 2

    if row["Contract_Length"] < 6:
        score += 2

    if row["Support_Tickets"] > 5:
        score += 2

    return score

data["Risk_Score"] = data.apply(calculate_risk, axis=1)

def risk_category(score):
    if score <= 2:
        return "Low Risk"
    elif score <=5:
        return "Medium Risk"
    else:
        return "High Risk"

data["Risk_Category"] = data["Risk_Score"].apply(risk_category)

# -----------------------------
# Sidebar Filters
# -----------------------------

st.sidebar.header("Filters")

region = st.sidebar.multiselect(
    "Select Region",
    data["Region"].unique(),
    default=data["Region"].unique()
)

industry = st.sidebar.multiselect(
    "Select Industry",
    data["Industry"].unique(),
    default=data["Industry"].unique()
)

risk = st.sidebar.multiselect(
    "Select Risk Category",
    data["Risk_Category"].unique(),
    default=data["Risk_Category"].unique()
)

filtered = data[
    (data["Region"].isin(region)) &
    (data["Industry"].isin(industry)) &
    (data["Risk_Category"].isin(risk))
]

# -----------------------------
# KPI Cards
# -----------------------------

total_clients = len(filtered)
high_risk = len(filtered[filtered["Risk_Category"]=="High Risk"])
avg_revenue = filtered["Revenue"].mean()

# Machine Learning preparation

le = LabelEncoder()

ml_data = data.copy()

ml_data["Industry"] = le.fit_transform(ml_data["Industry"])
ml_data["Region"] = le.fit_transform(ml_data["Region"])
ml_data["Renewal_Status"] = le.fit_transform(ml_data["Renewal_Status"])

X = ml_data.drop(["Client_ID","Renewal_Status","Risk_Category"], axis=1)
y = ml_data["Renewal_Status"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = DecisionTreeClassifier(max_depth=5)
model.fit(X_train, y_train)

pred = model.predict(X_test)

accuracy = accuracy_score(y_test, pred)

churn_rate = (1 - y.mean()) * 100

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Clients", total_clients)
col2.metric("High Risk Clients", high_risk)
col3.metric("Predicted Churn Rate %", round(churn_rate,2))
col4.metric("Average Revenue", round(avg_revenue,2))

# -----------------------------
# Risk Distribution Chart
# -----------------------------

st.subheader("Risk Category Distribution")

fig, ax = plt.subplots()

sns.countplot(data=filtered, x="Risk_Category", ax=ax)

st.pyplot(fig)

# -----------------------------
# Industry Risk Analysis
# -----------------------------

st.subheader("Industry-wise Risk")

fig2, ax2 = plt.subplots()

sns.countplot(data=filtered, x="Industry", hue="Risk_Category", ax=ax2)

plt.xticks(rotation=45)

st.pyplot(fig2)

# -----------------------------
# Revenue vs Risk
# -----------------------------

st.subheader("Revenue vs Risk Scatter")

fig3, ax3 = plt.subplots()

sns.scatterplot(
    data=filtered,
    x="Revenue",
    y="Risk_Score",
    hue="Risk_Category",
    ax=ax3
)

st.pyplot(fig3)

# -----------------------------
# Contract Length vs Churn
# -----------------------------

st.subheader("Contract Length vs Churn")

fig4, ax4 = plt.subplots()

sns.boxplot(
    data=data,
    x="Renewal_Status",
    y="Contract_Length",
    ax=ax4
)

st.pyplot(fig4)

# -----------------------------
# Confusion Matrix
# -----------------------------

st.subheader("Model Performance")

st.write("Accuracy:", round(accuracy,2))

cm = confusion_matrix(y_test, pred)

fig5, ax5 = plt.subplots()

sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")

st.pyplot(fig5)

# -----------------------------
# Feature Importance
# -----------------------------

st.subheader("Feature Importance")

importance = pd.Series(
    model.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

st.bar_chart(importance)

# -----------------------------
# Top High Risk Clients
# -----------------------------

st.subheader("Top 20 High Risk Clients")

highrisk_table = data[data["Risk_Category"]=="High Risk"].head(20)

st.dataframe(highrisk_table)

# -----------------------------
# Retention Strategy
# -----------------------------

st.subheader("AI Retention Strategy")

if st.button("Generate Retention Strategy"):

    st.success("Recommended Actions")

    st.write("• Offer discount for clients with payment delays over 30 days")

    st.write("• Assign dedicated account manager for high revenue clients")

    st.write("• Provide long-term contract incentives")

    st.write("• Reduce support response time for clients with many tickets")

    st.write("• Offer product training for low-usage customers")

# -----------------------------
# Responsible AI Section
# -----------------------------

st.subheader("Responsible AI Considerations")

st.write("""
• Predictive models may contain bias based on industry or region.

• Labeling clients as high-risk may affect business relationships.

• Client financial and usage data must be protected.

• AI predictions should assist decision-making, not replace human judgement.
""")
