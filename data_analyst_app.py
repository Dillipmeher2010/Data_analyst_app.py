import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from scipy.stats import zscore

st.title("Data Analyst App with Advanced Features")

# Step 1: File Upload
uploaded_file = st.file_uploader("Upload your CSV or Excel file", type=["csv", "xlsx"])

# Function: Data Cleaning
def data_cleaning(df):
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col].fillna(df[col].mode()[0], inplace=True)
        else:
            df[col].fillna(df[col].median(), inplace=True)
    
    # Outlier Treatment
    num_cols = df.select_dtypes(include=[np.number]).columns
    df = df[(np.abs(zscore(df[num_cols])) < 3).all(axis=1)]
    
    return df

# Function: Exploratory Data Analysis
def exploratory_data_analysis(df):
    st.write("## Summary Statistics")
    st.write(df.describe())
    
    st.write("## Univariate Analysis")
    for col in df.select_dtypes(include=[np.number]).columns:
        st.write(f"Distribution of {col}")
        fig, ax = plt.subplots()
        sns.histplot(df[col], kde=True, ax=ax)
        st.pyplot(fig)

    st.write("## Bivariate Analysis")
    num_cols = df.select_dtypes(include=[np.number]).columns
    if len(num_cols) >= 2:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(df[num_cols].corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)
    else:
        st.write("Not enough numerical columns for Bivariate Analysis.")

# Function: Advanced Analysis
def advanced_analysis(df):
    # Correlation Heatmap
    st.write("### Correlation Analysis")
    if st.checkbox("Show correlation heatmap"):
        corr = df.corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

    # Feature Engineering for Date Columns
    date_columns = df.select_dtypes(include=['datetime', 'object']).columns.tolist()
    for col in date_columns:
        try:
            df[col] = pd.to_datetime(df[col])
            st.write(f"Date feature engineering on: {col}")
            df[f"{col}_year"] = df[col].dt.year
            df[f"{col}_month"] = df[col].dt.month
            df[f"{col}_day"] = df[col].dt.day
            df[f"{col}_weekday"] = df[col].dt.weekday
        except Exception as e:
            st.write(f"Skipping column {col} for date parsing. Error: {e}")

    # Predictive Modeling (Simple Linear Regression)
    st.write("### Predictive Modeling")
    target_col = st.selectbox("Select Target Column for Prediction", df.select_dtypes(include=np.number).columns)
    if target_col:
        feature_cols = st.multiselect("Select Feature Columns for Prediction", 
                                      df.select_dtypes(include=np.number).drop(columns=[target_col]).columns)
        if feature_cols:
            X = df[feature_cols]
            y = df[target_col]
            model = LinearRegression()
            model.fit(X, y)
            predictions = model.predict(X)
            st.write("Prediction Results")
            st.write(pd.DataFrame({"Actual": y, "Predicted": predictions}))

    # Clustering Analysis
    st.write("### Clustering Analysis")
    n_clusters = st.slider("Select number of clusters", 2, 10, 3)
    if st.button("Perform Clustering"):
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(df.select_dtypes(include=np.number))
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(data_scaled)
        df['Cluster'] = kmeans.labels_
        st.write("Clustered Data")
        st.write(df.head())
        fig = px.scatter(df, x=feature_cols[0], y=feature_cols[1], color="Cluster", title="Clustering Analysis")
        st.plotly_chart(fig)

    # Time-Series Decomposition
    st.write("### Time-Series Decomposition")
    date_col = st.selectbox("Select Date Column for Time Series Analysis", date_columns)
    time_series_col = st.selectbox("Select Column for Trend Analysis", df.select_dtypes(include=np.number).columns)
    if date_col and time_series_col:
        df[date_col] = pd.to_datetime(df[date_col])
        df.set_index(date_col, inplace=True)
        st.line_chart(df[time_series_col])

# Function: Chat Assistant
def chat_assistant(query):
    response = ""
    if "outliers" in query.lower():
        response = "For outlier treatment, you can use IQR or Z-score methods. Outliers can affect statistical accuracy."
    elif "correlation" in query.lower():
        response = "Use correlation analysis to find relationships between numerical variables. High correlation indicates potential predictive relationships."
    elif "model" in query.lower():
        response = "Try building predictive models with linear regression or decision trees for numerical data."
    else:
        response = "I'm here to help with questions on data analysis steps, outliers, correlations, or model recommendations!"
    return response

# Main Pipeline Execution
if uploaded_file is not None:
    # Load the data
    try:
        if uploaded_file.name.endswith('.csv'):
            data = pd.read_csv(uploaded_file)
        else:
            data = pd.read_excel(uploaded_file)
        st.write("Data Loaded Successfully!")
    except Exception as e:
        st.write("Error loading data:", e)
    
    # Data Cleaning
    st.subheader("Data Cleaning Pipeline")
    data = data_cleaning(data)
    st.write("Data after Cleaning:")
    st.write(data.head())

    # Exploratory Data Analysis
    st.subheader("Exploratory Data Analysis")
    exploratory_data_analysis(data)

    # Advanced Analysis
    st.subheader("Advanced Data Analysis")
    advanced_analysis(data)

    # Chat Assistant
    st.sidebar.subheader("Chatbot Assistant")
    query = st.sidebar.text_input("Ask the assistant:")
    if st.sidebar.button("Get Recommendation"):
        st.sidebar.write(chat_assistant(query))

else:
    st.write("Please upload a CSV or Excel file to proceed.")
