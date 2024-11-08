# Install the required packages first by running:
# !pip install streamlit pandas matplotlib seaborn plotly openpyxl

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

st.title("Data Analyst App")

# Step 1: Upload the file
uploaded_file = st.file_uploader("Upload your CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file is not None:
    # Step 2: Load the data
    try:
        if uploaded_file.name.endswith('.csv'):
            data = pd.read_csv(uploaded_file)
        else:
            data = pd.read_excel(uploaded_file)
        st.write("Data Loaded Successfully!")
    except Exception as e:
        st.write("Error loading data:", e)

    # Display the first few rows of the data
    st.subheader("Raw Data Preview")
    st.write(data.head())

    # Step 3: Data Cleaning
    st.subheader("Data Cleaning")

    # Missing value treatment
    st.write("### Missing Value Treatment")
    if st.checkbox("Show missing values by column"):
        st.write(data.isnull().sum())

    missing_value_option = st.selectbox("Select how to handle missing values:", 
                                        ("Drop rows", "Fill with mean", "Fill with median", "Fill with mode"))
    if missing_value_option == "Drop rows":
        data = data.dropna()
    elif missing_value_option == "Fill with mean":
        data = data.fillna(data.mean())
    elif missing_value_option == "Fill with median":
        data = data.fillna(data.median())
    elif missing_value_option == "Fill with mode":
        data = data.fillna(data.mode().iloc[0])
    
    st.write("Data after Missing Value Treatment")
    st.write(data.head())

    # Outlier treatment
    st.write("### Outlier Treatment")
    numeric_cols = data.select_dtypes(include=np.number).columns.tolist()
    if st.checkbox("Show outliers using IQR method"):
        for col in numeric_cols:
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = data[((data[col] < (Q1 - 1.5 * IQR)) | (data[col] > (Q3 + 1.5 * IQR)))]
            st.write(f"{col}: {len(outliers)} outliers")

    outlier_option = st.selectbox("Select outlier handling method:", ("None", "Remove", "Cap"))
    if outlier_option == "Remove":
        for col in numeric_cols:
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            data = data[~((data[col] < (Q1 - 1.5 * IQR)) | (data[col] > (Q3 + 1.5 * IQR)))]
    elif outlier_option == "Cap":
        for col in numeric_cols:
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_cap = Q1 - 1.5 * IQR
            upper_cap = Q3 + 1.5 * IQR
            data[col] = np.where(data[col] < lower_cap, lower_cap, data[col])
            data[col] = np.where(data[col] > upper_cap, upper_cap, data[col])

    st.write("Data after Outlier Treatment")
    st.write(data.head())

    # Step 4: Summary Statistics
    st.subheader("Summary Statistics")
    st.write(data.describe())

    # Step 5: Univariate Analysis
    st.subheader("Univariate Analysis")
    column = st.selectbox("Select a column for univariate analysis", data.columns)
    if column:
        st.write("Histogram")
        fig, ax = plt.subplots()
        sns.histplot(data[column], kde=True, ax=ax)
        st.pyplot(fig)
        
        st.write("Boxplot")
        fig, ax = plt.subplots()
        sns.boxplot(x=data[column], ax=ax)
        st.pyplot(fig)

    # Step 6: Bivariate Analysis
    st.subheader("Bivariate Analysis")
    col1 = st.selectbox("Select first column for bivariate analysis", data.columns)
    col2 = st.selectbox("Select second column for bivariate analysis", data.columns)
    if col1 and col2:
        st.write("Scatter plot")
        fig = px.scatter(data, x=col1, y=col2)
        st.plotly_chart(fig)

    # Step 7: Multivariate Analysis
    st.subheader("Multivariate Analysis")
    multivariate_cols = st.multiselect("Select columns for pairplot", data.columns)
    if multivariate_cols:
        st.write("Pairplot")
        fig = sns.pairplot(data[multivariate_cols])
        st.pyplot(fig)

    # Step 8: Trend and Pattern Analysis (if date column is present)
    date_cols = data.select_dtypes(include=[np.datetime64]).columns.tolist()
    if date_cols:
        date_col = st.selectbox("Select a date column for trend analysis", date_cols)
        if date_col:
            data[date_col] = pd.to_datetime(data[date_col])
            st.write("Line plot of data over time")
            trend_col = st.selectbox("Select a column to observe trend", numeric_cols)
            fig = px.line(data, x=date_col, y=trend_col)
            st.plotly_chart(fig)
else:
    st.write("Please upload a CSV or Excel file to proceed.")
