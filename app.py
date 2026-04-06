import streamlit as st
import pandas as pd
import numpy as np
from sklearn import preprocessing

st.title("Gradient Descent vs Stochastic Gradient Descent")

# Upload dataset
uploaded_file = st.file_uploader("insurance_data (3).csv", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(r"D:\Stochastic_Gradient_Descent\insurance_data (3).csv")

    st.write("Dataset Preview:")
    st.dataframe(df.head())

    if 'charges' not in df.columns:
        st.error("Dataset must contain 'charges' column")
    else:
        # Scaling
        sx = preprocessing.MinMaxScaler()
        sy = preprocessing.MinMaxScaler()

        X = df.drop('charges', axis='columns')
        y = df['charges']

        scaled_X = sx.fit_transform(X)
        scaled_y = sy.fit_transform(y.values.reshape(-1, 1))

        # Dataset Info
        st.write("Full Dataset:")
        st.dataframe(df)

        st.write("Dataset Shape:", df.shape)

        st.write("Column Names:")
        st.write(df.columns)

        st.write("Statistical Summary:")
        st.write(df.describe())

        st.write("Data Types:")
        st.write(df.dtypes)

        # Visualizations (FIXED INDENTATION)
        st.write("Correlation Heatmap:")
        st.write(df.corr(numeric_only=True))

        st.write("Sample Visualization:")
        st.bar_chart(df.select_dtypes(include=np.number).iloc[:20])

        # Gradient Descent Function
        def batch_gradient_descent(X, y, epochs, lr):
            n, m = X.shape
            weights = np.ones(m)
            bias = 0
            losses = []

            for i in range(epochs):
                y_pred = np.dot(X, weights) + bias
                loss = np.mean((y - y_pred) ** 2)

                dW = -(2/n) * np.dot(X.T, (y - y_pred))
                dB = -(2/n) * np.sum(y - y_pred)

                weights -= lr * dW
                bias -= lr * dB

                losses.append(loss)

            return weights, bias, losses

        # User Inputs
        epochs = st.slider("Epochs", 10, 500, 100)
        lr = st.slider("Learning Rate", 0.001, 0.1, 0.01)

        if st.button("Train Model"):
            weights, bias, losses = batch_gradient_descent(
                scaled_X, scaled_y.flatten(), epochs, lr
            )

            st.success("Training Completed!")

            st.write("Weights:", weights)
            st.write("Bias:", bias)

            st.line_chart(losses)

# OPTIONAL: fallback dataset if no upload
else:
    st.warning("Please upload a dataset. Using sample dataset instead.")

    df = pd.DataFrame({
        "age": np.random.randint(18, 65, 1000),
        "bmi": np.random.uniform(15, 40, 1000),
        "children": np.random.randint(0, 6, 1000),
        "charges": np.random.randint(2000, 50000, 1000)
    })

    st.write("Generated Large Dataset:")
    st.dataframe(df)

    st.write("Dataset Shape:", df.shape)
    st.write("Columns:", df.columns)

    st.dataframe(df.head())

    st.write("Columns in dataset:", df.columns)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.dataframe(df.head())

    # Normalize column names
    df.columns = df.columns.str.strip().str.lower()

    # If dataset too small → generate large dataset
    if df.shape[0] < 50:
        st.warning("Dataset too small. Generating large dataset...")

        df = pd.DataFrame({
            "age": np.random.randint(18, 65, 1000),
            "bmi": np.random.uniform(15, 40, 1000),
            "children": np.random.randint(0, 6, 1000),
            "charges": np.random.randint(2000, 50000, 1000)
        })
