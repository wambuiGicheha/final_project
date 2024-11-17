import streamlit as st
import pickle
import matplotlib.pyplot as plt
import pandas as pd

# Load the trained Random Forest model (assuming you have already pickled it)
with open('random_forest_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Define the visualization function for feature importance
def plot_feature_importance(model):
    if hasattr(model.named_steps['classifier'], 'feature_importances_'):
        feature_importances = model.named_steps['classifier'].feature_importances_
        feature_names = model.named_steps['preprocessor'].transformers_[0][1].get_feature_names_out()
        numerical_feature_names = ['upvotes', 'num_comments', 'year', 'month', 'day', 'hour', 'upvotes_per_comment', 'has_body']
        all_feature_names = list(feature_names) + numerical_feature_names
        importance_df = pd.DataFrame({
            'Feature': all_feature_names,
            'Importance': feature_importances
        }).sort_values(by='Importance', ascending=False)

        plt.figure(figsize=(10, 6))
        plt.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.title('Feature Importance for Random Forest Model')
        plt.gca().invert_yaxis()
        st.pyplot(plt)
    else:
        st.write("Model does not support feature importances.")

# Streamlit interface
st.title("Random Forest Model Feature Importance")
if st.button("Show Feature Importance"):
    plot_feature_importance(model)


