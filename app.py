import streamlit as st
import pandas as pd
import numpy as np
from sklearn import preprocessing

st.title("Gradient Descent vs Stochastic Gradient Descent")

# Upload dataset
uploaded_file = st.file_uploader(r"c:\Users\athis\Downloads\insurance_data (3).csv", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(r"D:\Stochastic_Gradient_Descent\insurance_data (3).csv")
    st.write("Dataset Preview:")
    st.dataframe(df.head())

    if 'price' not in df.columns:
        st.error("Dataset must contain 'price' column")
    else:
        # Scaling
        sx = preprocessing.MinMaxScaler()
        sy = preprocessing.MinMaxScaler()

        X = df.drop('price', axis='columns')
        y = df['price']

        scaled_X = sx.fit_transform(X)
        scaled_y = sy.fit_transform(y.values.reshape(-1, 1))

        st.write("Scaled Features:")
        st.write(scaled_X[:5])

        st.write("Scaled Target:")
        st.write(scaled_y[:5])

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