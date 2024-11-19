import nltk
from preprocessing import TextPreprocessor 
import streamlit as st
import joblib
from lime.lime_text import LimeTextExplainer
import streamlit.components.v1 as components

# Download stopwords
nltk.download('punkt')  # Downloads Punkt tokenizer
nltk.download('punkt_tab')  # Ensures punkt_tab is available
nltk.download('stopwords')  # Ensure stopwords are available
nltk.download('wordnet')  # Ensure WordNet is available for lemmatization

# Load the trained machine learning pipeline
pipeline = joblib.load('models/sentiments_pipeline.pkl')

# Streamlit app configuration
st.set_page_config(page_title="MoodLens: Your Lens to Mental Well-Being ", page_icon="🧠", layout="wide")

# CSS for app styling
st.markdown(
    """
    <style>
    .stApp {
        background-color: #e0f7fa; /* Light blue shade */
    }
    .stSidebar {
        background-color: #C8A2C8; /* Light purple for the sidebar */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# App title and description
st.title("MoodLens: Your Lens to Mental Well-Being 🧠")
st.write("""
This application analyzes social media posts from Reddit to predict whether a user is likely experiencing depression or not. 
It leverages a pre-trained machine learning model for advanced sentiment analysis.
""")

# Input section
st.header("Enter Reddit Text for Analysis")
user_input = st.text_area(
    "Paste your text here (e.g., Reddit title or body entry):",
    placeholder="Type or paste text here...",
    height=100
)

# Button to make predictions
if st.button("Analyze Sentiment"):
    if user_input.strip() == "":
        st.error("Please enter some text for analysis.")
    else:
        # Predict sentiment
        prediction = pipeline.predict([user_input])[0]
        probability = pipeline.predict_proba([user_input])[0]
        
        # Interpret prediction
        result = "Depressed" if prediction == 1 else "Not Depressed"
        confidence = probability[int(prediction)] * 100
        
        # Display results
        st.subheader("Prediction Result")
        st.write(f"**Prediction:** {result}")
        st.write(f"**Confidence:** {confidence:.2f}%")

# Button for explanation using LIME
if st.button("Explain Prediction"):
    if user_input.strip() == "":
        st.error("Please enter some text for analysis.")
    else:
        # Generate LIME explanation
        explainer = LimeTextExplainer(class_names=["Not Depressed", "Depressed"])
        explanation = explainer.explain_instance(user_input, pipeline.predict_proba, num_features=5)
        
        # Display explanation
        st.subheader("LIME Explanation")
        html_content = explanation.as_html()
        components.html(html_content, height=800, scrolling=True)

# Sidebar content
st.sidebar.header("About the App")
st.sidebar.write("""
This tool is part of a machine learning project that predicts user mental wellness based on text data from Reddit posts. 
It uses **TF-IDF** for text vectorization and a logistic regression model for classification.
""")
st.sidebar.write("The dataset used for training includes labeled social media posts.")

# Footer with a disclaimer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; font-size: 0.9em; color: gray;'>
        <strong>Disclaimer:</strong> This tool is designed to flag posts that may indicate potential signs of depression. It is not a diagnostic tool. 
        For professional help, please consult a licensed mental health professional or a trusted support service.
    </div>
    """,
    unsafe_allow_html=True
)

# Footer credits
st.write("---")
st.write("Developed by Group_11. Powered by Streamlit and Scikit-learn.")

       

