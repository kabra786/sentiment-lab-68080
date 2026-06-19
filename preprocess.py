"""
Text preprocessing module for legal notice classification.
"""

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


# Download required NLTK resources
nltk.download("stopwords")
nltk.download("punkt")
nltk.download("punkt_tab")


stemmer = PorterStemmer()
stop_words = set(stopwords.words("english"))


def preprocess_text(text):
    """
    Clean and preprocess legal notice text.

    Args:
        text (str):
            Original legal notice text.

    Returns:
        str:
            Cleaned and processed text.
    """

    # Remove HTML tags
    text = re.sub(r"<.*?>", " ", str(text))

    # Convert text to lowercase
    text = text.lower()

    # Remove punctuation, numbers and special characters
    text = re.sub(r"[^a-z\s]", " ", text)

    # Tokenization
    words = text.split()

    # Stopword removal
    words = [
        word for word in words
        if word not in stop_words
    ]

    # Stemming
    words = [
        stemmer.stem(word)
        for word in words
    ]

    return " ".join(words)