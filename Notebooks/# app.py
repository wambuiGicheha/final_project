# app.py

import streamlit as st
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Title and description
st.title("MoodLens: Social Media-Based Depression Analysis")
st.write("Data\reduced_reddit.csv")

# Load pre-trained model and tokenizer 
@st.cache_resource
def load_model():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased') 
    model = BertForSequenceClassification.from_pretrained('path_to_your_model')  # 
    return tokenizer, model

tokenizer, model = load_model()

# Function for prediction
def predict(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    return predictions[0].tolist()  # Returns list of probabilities for each class

# User input for single text analysis
user_input = st.text_area("Enter text here for analysis:")

if user_input:
    st.write("Analyzing...")
    prediction = predict(user_input)
    st.write("Prediction probabilities:", prediction)
    # Customize this based on your model's output labels
    st.write(f"The model predicts this text as {prediction.index(max(prediction))} with a probability of {max(prediction):.2f}")

# File uploader for batch processing
uploaded_file = st.file_uploader("Data\reduced_reddit.csv", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("Data Preview:", data.head())
'
    if 'text_column' in data.columns:
        st.write("Analyzing uploaded data...")
        data['Predictions'] = data['text_column'].apply(lambda x: predict(x)[0])  # Modify based on your prediction output format
        st.write(data)
    else:
        st.write("Error: Please ensure your CSV file has a 'text_column' column for analysis.")
