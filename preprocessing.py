import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string

from sklearn.base import TransformerMixin, BaseEstimator
from nltk.tokenize import word_tokenize
import numpy as np  # Optional: For handling array transformations
import pandas as pd

# Ensure necessary NLTK resources are downloaded
nltk.download('punkt')  # Downloads Punkt tokenizer
nltk.download('punkt_tab')  # Ensures punkt_tab is available
nltk.download('stopwords')  # Ensure stopwords are available
nltk.download('wordnet')  # Ensure WordNet is available for lemmatization

# Initialize stopwords, punctuation, and lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()



class TextPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self  # No fitting required for this transformer

    def transform(self, X, y=None):
        # Ensure X is a pandas Series and apply preprocessing
        if isinstance(X, pd.Series):
            return X.apply(self._preprocess_text)
        elif isinstance(X, (list, np.ndarray)):
            return [self._preprocess_text(text) for text in X]
        else:
            raise ValueError("Input should be a pandas Series, list, or numpy array")

    def _preprocess_text(self, text):
        if pd.isnull(text):
            return ''  # Return empty string for missing values
        text = text.lower()  # Lowercase
        text = ''.join([char for char in text if char not in string.punctuation])  # Remove punctuation
        tokens = word_tokenize(text)  # Tokenize
        tokens = [word for word in tokens if word not in stop_words]  # Remove stopwords
        tokens = [lemmatizer.lemmatize(word) for word in tokens]  # Lemmatize
        return ' '.join(tokens)  # Join tokens back into string







