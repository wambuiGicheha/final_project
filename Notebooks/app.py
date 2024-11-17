import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Import your KNN class from the implementation you created
from knn_implementation import KNN  # Make sure the KNN implementation is in a file named 'knn_implementation.py'
# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Instantiate and train your KNN model
knn = KNN(k=3)  # Set k as desired
knn.fit(X_train, y_train)
# Streamlit app title and description
st.title("K-Nearest Neighbors (KNN) Classifier")
st.write("This app allows you to interact with a KNN classifier trained on the Iris dataset.")

# User input for feature values
st.sidebar.header("Input Features")
sepal_length = st.sidebar.slider("Sepal Length", float(X[:, 0].min()), float(X[:, 0].max()), float(X[:, 0].mean()))
sepal_width = st.sidebar.slider("Sepal Width", float(X[:, 1].min()), float(X[:, 1].max()), float(X[:, 1].mean()))
petal_length = st.sidebar.slider("Petal Length", float(X[:, 2].min()), float(X[:, 2].max()), float(X[:, 2].mean()))
petal_width = st.sidebar.slider("Petal Width", float(X[:, 3].min()), float(X[:, 3].max()), float(X[:, 3].mean()))

# Collect the input values into a numpy array
input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

# Make a prediction
prediction = knn.predict(input_data)
predicted_class = iris.target_names[prediction[0]]

# Display the prediction
st.write("### Prediction")
st.write(f"The model predicts that the input data belongs to the class: **{predicted_class}**")

# Optionally, show accuracy on the test set
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
st.write("### Model Accuracy")
st.write(f"The accuracy of the KNN model on the test set is: **{accuracy:.2f}**")
