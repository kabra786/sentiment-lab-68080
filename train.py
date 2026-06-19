"""
Training script for Legal Notice Classification.
Includes MLflow tracking.
"""

import os
import json
import time
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

from src.preprocess import preprocess_text
from src.features import (
    create_bow,
    create_tfidf
)

from src.evaluate import evaluate_model


# Load configuration
with open("config.json", "r") as file:
    config = json.load(file)


# Load dataset
df = pd.read_csv(
    "data/raw/legal_notices.csv"
)


print("Dataset Loaded Successfully")
print(df.head())


# Text preprocessing
df["clean_text"] = df["notice"].apply(
    preprocess_text
)


# Features and labels
X = df["clean_text"]

# Use category column (A, B, C)
y = df["category"]


# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=config["test_size"],
    random_state=config["random_seed"],
    stratify=y
)


print("Data Split Completed")

# ===============================
# Create Feature Representations
# ===============================

# Bag of Words
bow_vectorizer = create_bow(
    config["max_features"]
)

X_train_bow = bow_vectorizer.fit_transform(
    X_train
)

X_test_bow = bow_vectorizer.transform(
    X_test
)


# TF-IDF
tfidf_vectorizer = create_tfidf(
    config["max_features"]
)

X_train_tfidf = tfidf_vectorizer.fit_transform(
    X_train
)

X_test_tfidf = tfidf_vectorizer.transform(
    X_test
)


print("Feature extraction completed")


# ===============================
# MLflow Experiment Setup
# ===============================

mlflow.set_experiment(
    "Legal_Notice_Classification"
)


# ===============================
# Function to train and log models
# ===============================

def run_experiment(
    model,
    model_name,
    vectorizer_name,
    X_train_data,
    X_test_data
):
    """
    Train model, evaluate it and
    log all information to MLflow.
    """

    with mlflow.start_run(
        run_name=f"{model_name}_{vectorizer_name}"
    ):

        # Model training and evaluation
        results = evaluate_model(
            model=model,
            X_train=X_train_data,
            X_test=X_test_data,
            y_train=y_train,
            y_test=y_test,
            model_name=f"{model_name}_{vectorizer_name}"
        )


        # Log general parameters
        mlflow.log_param(
            "Model",
            model_name
        )

        mlflow.log_param(
            "Vectorizer",
            vectorizer_name
        )

        mlflow.log_param(
            "Random Seed",
            config["random_seed"]
        )

        mlflow.log_param(
            "Max Features",
            config["max_features"]
        )


        # Log model parameters
        if model_name == "LogisticRegression":

            mlflow.log_param(
                "C",
                model.C
            )

            mlflow.log_param(
                "Max Iterations",
                model.max_iter
            )


        elif model_name == "MultinomialNB":

            mlflow.log_param(
                "Alpha",
                model.alpha
            )


        # Log performance metrics
        mlflow.log_metric(
            "Accuracy",
            results["Accuracy"]
        )

        mlflow.log_metric(
            "F1 Weighted",
            results["F1 Weighted"]
        )

        mlflow.log_metric(
            "F1 Macro",
            results["F1 Macro"]
        )

        mlflow.log_metric(
            "Training Time",
            results["Training Time"]
        )


        # Log confusion matrix image
        mlflow.log_artifact(
            results["Confusion Matrix"]
        )


        # Save sklearn model
        mlflow.sklearn.log_model(
            model,
            "model"
        )


        print(
            f"Completed: {model_name} + {vectorizer_name}"
        )

        return results
    

# ==================================
# Main Model Experiments (4 Runs)
# ==================================

all_results = []


# 1. Logistic Regression + Bag of Words
lr_bow = LogisticRegression(
    C=config["model_1"]["C"],
    max_iter=config["model_1"]["max_iter"]
)

all_results.append(
    run_experiment(
        model=lr_bow,
        model_name="LogisticRegression",
        vectorizer_name="BagOfWords",
        X_train_data=X_train_bow,
        X_test_data=X_test_bow
    )
)


# 2. Logistic Regression + TF-IDF
lr_tfidf = LogisticRegression(
    C=config["model_1"]["C"],
    max_iter=config["model_1"]["max_iter"]
)

all_results.append(
    run_experiment(
        model=lr_tfidf,
        model_name="LogisticRegression",
        vectorizer_name="TFIDF",
        X_train_data=X_train_tfidf,
        X_test_data=X_test_tfidf
    )
)


# 3. Naive Bayes + Bag of Words
nb_bow = MultinomialNB(
    alpha=config["model_2"]["alpha"]
)

all_results.append(
    run_experiment(
        model=nb_bow,
        model_name="MultinomialNB",
        vectorizer_name="BagOfWords",
        X_train_data=X_train_bow,
        X_test_data=X_test_bow
    )
)


# 4. Naive Bayes + TF-IDF
nb_tfidf = MultinomialNB(
    alpha=config["model_2"]["alpha"]
)

all_results.append(
    run_experiment(
        model=nb_tfidf,
        model_name="MultinomialNB",
        vectorizer_name="TFIDF",
        X_train_data=X_train_tfidf,
        X_test_data=X_test_tfidf
    )
)


# ==================================
# Hyperparameter Experiments (3 Runs)
# Logistic Regression C values
# ==================================

for c_value in [0.1, 1, 10]:

    lr_experiment = LogisticRegression(
        C=c_value,
        max_iter=config["model_1"]["max_iter"]
    )

    all_results.append(
        run_experiment(
            model=lr_experiment,
            model_name=f"LogisticRegression_C_{c_value}",
            vectorizer_name="TFIDF",
            X_train_data=X_train_tfidf,
            X_test_data=X_test_tfidf
        )
    )


# ==================================
# Final Comparison Table
# ==================================

results_df = pd.DataFrame(all_results)


print("\n" + "=" * 50)
print("FINAL MODEL COMPARISON")
print("=" * 50)

print(
    results_df[
        [
            "Model",
            "Accuracy",
            "F1 Macro",
            "F1 Weighted",
            "Training Time",
            "Inference Time"
        ]
    ]
)


# ==================================
# Save results CSV
# ==================================

os.makedirs(
    "results",
    exist_ok=True
)

results_df.to_csv(
    "results/model_comparison.csv",
    index=False
)


print("\nAll experiments completed successfully!")
print("MLflow runs created successfully.")
print("Results saved in results folder.")