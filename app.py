import time
import streamlit as st
import joblib

# Load the trained model pipeline
pipeline = joblib.load('models/sentiments_pipeline.pkl')

# Input section
user_input = st.text_area("Paste your text here for analysis:", height=100)

# Button for prediction
if st.button("Analyze Sentiment"):
    if user_input.strip() == "":
        st.error("Please enter some text for analysis.")
    else:
        # Display progress bar
        progress_bar = st.progress(0)
        for percent_complete in range(100):
            time.sleep(0.01)  # Simulate some work being done
            progress_bar.progress(percent_complete + 1)

        # Predict sentiment
        prediction = pipeline.predict([user_input])[0]
        probability = pipeline.predict_proba([user_input])[0]
        result = "Depressed" if prediction == 1 else "Not Depressed"
        confidence = probability[int(prediction)] * 100

        # Display the results
        st.subheader("Prediction Result")
        st.write(f"**Prediction:** {result}")
        st.write(f"**Confidence:** {confidence:.2f}%")

