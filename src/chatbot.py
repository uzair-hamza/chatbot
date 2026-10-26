"""Train a TF-IDF + Logistic Regression intent classifier for the UET FAQ bot.

Data: intents.json (my own file, written for this project).
Model: TfidfVectorizer (word n-grams 1-2) + LogisticRegression.
Evaluation: stratified 80/20 split, random_state=42.
Outputs: model/chatbot_model.joblib, results/eval.log, results/confusion_matrix.png
"""
import json
import os
import random
import sys

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_data(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    texts, labels, responses = [], [], {}
    for intent in data["intents"]:
        responses[intent["tag"]] = intent["responses"]
        for pattern in intent["patterns"]:
            texts.append(pattern)
            labels.append(intent["tag"])
    return texts, labels, responses


def main():
    texts, labels, responses = load_data(os.path.join(BASE, "intents.json"))
    print(f"Total patterns: {len(texts)}")
    print(f"Total intents: {len(set(labels))}")

    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.20, random_state=42, stratify=labels
    )
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

    model = Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=True, ngram_range=(1, 2))),
        ("clf", LogisticRegression(max_iter=1000)),
    ])
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0)

    print(f"\nAccuracy: {acc:.4f}")
    print(report)

    os.makedirs(os.path.join(BASE, "model"), exist_ok=True)
    os.makedirs(os.path.join(BASE, "results"), exist_ok=True)
    joblib.dump({"model": model, "responses": responses},
                os.path.join(BASE, "model", "chatbot_model.joblib"))

    with open(os.path.join(BASE, "results", "eval.log"), "w", encoding="utf-8") as f:
        f.write(f"Total patterns: {len(texts)}\n")
        f.write(f"Total intents: {len(set(labels))}\n")
        f.write(f"Train size: {len(X_train)}, Test size: {len(X_test)}\n")
        f.write(f"Split: stratified 80/20, random_state=42\n")
        f.write(f"\nAccuracy: {acc:.4f}\n\n")
        f.write(report)

    # Confusion matrix plot
    labels_sorted = sorted(set(labels))
    cm = confusion_matrix(y_test, y_pred, labels=labels_sorted)
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels_sorted)))
    ax.set_xticklabels(labels_sorted, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(labels_sorted)))
    ax.set_yticklabels(labels_sorted, fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix (accuracy {acc:.2%})")
    for i in range(len(labels_sorted)):
        for j in range(len(labels_sorted)):
            if cm[i, j]:
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(BASE, "results", "confusion_matrix.png"), dpi=150)
    print("Saved: model/chatbot_model.joblib, results/eval.log, results/confusion_matrix.png")


if __name__ == "__main__":
    main()
