"""
Feature extraction module for legal notice classification.
"""

import numpy as np
from sklearn.feature_extraction.text import (
    CountVectorizer,
    TfidfVectorizer
)


def create_bow(max_features=5000):
    """
    Create Bag of Words vectorizer.

    Args:
        max_features (int):
            Maximum number of features.

    Returns:
        CountVectorizer:
            Configured BoW vectorizer.
    """

    return CountVectorizer(
        max_features=max_features
    )


def create_tfidf(max_features=5000):
    """
    Create TF-IDF vectorizer.

    Args:
        max_features (int):
            Maximum number of features.

    Returns:
        TfidfVectorizer:
            Configured TF-IDF vectorizer.
    """

    return TfidfVectorizer(
        max_features=max_features,
        sublinear_tf=True
    )


def get_top_terms(vectorizer, matrix, top_n=20):
    """
    Extract top terms based on frequency or TF-IDF score.

    Args:
        vectorizer:
            Fitted CountVectorizer or TfidfVectorizer.

        matrix:
            Transformed document-term matrix.

        top_n (int):
            Number of top terms to return.

    Returns:
        list:
            List of top terms with scores.
    """

    terms = vectorizer.get_feature_names_out()

    scores = np.array(
        matrix.sum(axis=0)
    ).flatten()

    top_indices = scores.argsort()[::-1][:top_n]

    results = []

    for index in top_indices:
        results.append(
            (
                terms[index],
                float(scores[index])
            )
        )

    return results


def print_top_terms(terms):
    """
    Display top terms in a readable format.

    Args:
        terms (list):
            List of terms and scores.
    """

    print("\nTop Terms")
    print("-" * 40)

    for word, score in terms:
        print(
            f"{word}: {score:.2f}"
        )