# AI-Based Legal Notice Classification System

## Project Description

This project implements a multi-class machine learning classifier for legal notices. The system classifies legal documents into three categories:

- A: Contract Dispute
- B: Intellectual Property Claim
- C: Regulatory Compliance

Two machine learning approaches are compared:

- Logistic Regression
- Multinomial Naive Bayes

Two feature extraction methods are used:

- Bag of Words (CountVectorizer)
- TF-IDF (TfidfVectorizer)

The best-performing model is selected based on evaluation metrics.

---

## Project Structure

```
sentiment-lab-68080/
│
├── data/
├── notebooks/
├── src/
├── results/
├── mlruns/
├── config.json
├── requirements.txt
└── README.md
```

---

## Installation

Create virtual environment:

```bash
python -m venv venv
```

Activate environment:

Windows:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running Notebook

Open the notebook:

```
notebooks/sentiment_analysis.ipynb
```

Run all cells sequentially.

---

## Configuration File

The `config.json` file stores:

- Random seed
- Test size
- Maximum features
- Logistic Regression parameters
- Naive Bayes parameters

---

## Results Summary

Four experiments are performed:

1. Logistic Regression + Bag of Words
2. Logistic Regression + TF-IDF
3. Naive Bayes + Bag of Words
4. Naive Bayes + TF-IDF

Evaluation metrics include:

- Accuracy
- Precision
- Recall
- F1-score
- Training time
- Inference time
- Confusion matrices

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- NLTK
- Matplotlib
- Seaborn
- MLflow
- Jupyter Notebook