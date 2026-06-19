"""
Model evaluation module for legal notice classification.
"""

import os
import time
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


def evaluate_model(
    model,
    X_train,
    X_test,
    y_train,
    y_test,
    model_name,
    save_path="results"
):
    """
    Train and evaluate a machine learning model.

    Args:
        model:
            Machine learning model object.

        X_train:
            Training feature matrix.

        X_test:
            Testing feature matrix.

        y_train:
            Training labels.

        y_test:
            Testing labels.

        model_name (str):
            Name of the model.

        save_path (str):
            Folder path to save confusion matrix.

    Returns:
        dict:
            Dictionary containing performance metrics.
    """

    # Create results folder if not available
    os.makedirs(save_path, exist_ok=True)


    # Training time
    start_train = time.time()

    model.fit(X_train, y_train)

    training_time = time.time() - start_train


    # Prediction time
    start_predict = time.time()

    predictions = model.predict(X_test)

    inference_time = time.time() - start_predict


    # Calculate metrics
    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision_macro = precision_score(
        y_test,
        predictions,
        average="macro"
    )

    recall_macro = recall_score(
        y_test,
        predictions,
        average="macro"
    )

    f1_macro = f1_score(
        y_test,
        predictions,
        average="macro"
    )


    precision_weighted = precision_score(
        y_test,
        predictions,
        average="weighted"
    )

    recall_weighted = recall_score(
        y_test,
        predictions,
        average="weighted"
    )

    f1_weighted = f1_score(
        y_test,
        predictions,
        average="weighted"
    )


    # Create confusion matrix
    cm = confusion_matrix(
        y_test,
        predictions
    )


    plt.figure(figsize=(7, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["A", "B", "C"],
        yticklabels=["A", "B", "C"]
    )

    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")
    plt.title(f"{model_name} Confusion Matrix")


    # Save confusion matrix image
    file_name = (
        model_name.replace(" ", "_")
        + "_confusion_matrix.png"
    )

    image_path = os.path.join(
        save_path,
        file_name
    )

    plt.savefig(
        image_path,
        bbox_inches="tight"
    )

    plt.close()


    # Return all results
    return {

        "Model": model_name,

        "Accuracy": accuracy,

        "Precision Macro":
            precision_macro,

        "Recall Macro":
            recall_macro,

        "F1 Macro":
            f1_macro,

        "Precision Weighted":
            precision_weighted,

        "Recall Weighted":
            recall_weighted,

        "F1 Weighted":
            f1_weighted,

        "Training Time":
            training_time,

        "Inference Time":
            inference_time,

        "Confusion Matrix":
            image_path
    }