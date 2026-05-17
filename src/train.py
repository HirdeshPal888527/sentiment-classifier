"""
Sentiment Analysis - IMDB Movie Reviews
Train and compare multiple ML models, save the best one.
"""

import os
import re
import time
import pickle
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH  = os.path.join(BASE_DIR, "data", "IMDB_Dataset.csv")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
MODEL_DIR  = os.path.join(BASE_DIR, "model")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODEL_DIR,   exist_ok=True)

# ── Stopwords (built-in, no NLTK download needed) ────────────────────────────
STOPWORDS = {
    "i","me","my","myself","we","our","ours","ourselves","you","your","yours",
    "yourself","he","him","his","himself","she","her","hers","herself","it",
    "its","itself","they","them","their","theirs","themselves","what","which",
    "who","whom","this","that","these","those","am","is","are","was","were",
    "be","been","being","have","has","had","having","do","does","did","doing",
    "a","an","the","and","but","if","or","because","as","until","while","of",
    "at","by","for","with","about","against","between","into","through",
    "during","before","after","above","below","to","from","up","down","in",
    "out","on","off","over","under","again","further","then","once","here",
    "there","when","where","why","how","all","both","each","few","more","most",
    "other","some","such","no","nor","not","only","own","same","so","than",
    "too","very","s","t","can","will","just","don","should","now","d","ll",
    "m","o","re","ve","y","ain","aren","couldn","didn","doesn","hadn","hasn",
    "haven","isn","ma","mightn","mustn","needn","shan","shouldn","wasn",
    "weren","won","wouldn"
}

# ── Text Preprocessing ────────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text)          # remove HTML tags
    text = re.sub(r"[^a-zA-Z\s]", " ", text)       # keep only letters
    text = text.lower()
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 2]
    return " ".join(tokens)

# ── Load & Preprocess ─────────────────────────────────────────────────────────
def load_data():
    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)
    df["label"] = (df["sentiment"] == "positive").astype(int)
    print(f"  Total reviews : {len(df):,}")
    print(f"  Positive      : {df['label'].sum():,}")
    print(f"  Negative      : {(df['label']==0).sum():,}")

    print("Cleaning text (this takes ~30 seconds)...")
    t0 = time.time()
    df["clean_review"] = df["review"].apply(clean_text)
    print(f"  Done in {time.time()-t0:.1f}s")
    return df

# ── EDA Plots ─────────────────────────────────────────────────────────────────
def save_eda_plots(df):
    print("Saving EDA plots...")

    # 1. Class distribution
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = df["sentiment"].value_counts()
    bars = ax.bar(counts.index, counts.values, color=["#2196F3", "#F44336"], edgecolor="white", width=0.5)
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200,
                f"{val:,}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_title("Class Distribution", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Sentiment")
    ax.set_ylabel("Count")
    ax.set_ylim(0, counts.max() * 1.15)
    sns.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "class_distribution.png"), dpi=150)
    plt.close()

    # 2. Review length distribution
    df["review_len"] = df["review"].apply(lambda x: len(x.split()))
    fig, ax = plt.subplots(figsize=(8, 4))
    for sentiment, color in [("positive", "#2196F3"), ("negative", "#F44336")]:
        subset = df[df["sentiment"] == sentiment]["review_len"]
        ax.hist(subset, bins=60, alpha=0.6, color=color, label=sentiment, edgecolor="none")
    ax.set_title("Review Length Distribution by Sentiment", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Number of Words")
    ax.set_ylabel("Frequency")
    ax.legend()
    sns.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "review_length_distribution.png"), dpi=150)
    plt.close()

    print("  EDA plots saved to results/")

# ── Train & Evaluate ──────────────────────────────────────────────────────────
def train_models(X_train, X_test, y_train, y_test):
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, C=1.0, random_state=42),
        "Naive Bayes"        : MultinomialNB(alpha=0.1),
        "LinearSVC"          : LinearSVC(max_iter=2000, C=1.0, random_state=42),
    }

    results = {}
    print("\n── Model Training & Evaluation ──────────────────────────────────")
    for name, model in models.items():
        t0 = time.time()
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc   = accuracy_score(y_test, preds)
        elapsed = time.time() - t0
        results[name] = {"model": model, "preds": preds, "accuracy": acc, "time": elapsed}
        print(f"\n{name}")
        print(f"  Accuracy : {acc*100:.2f}%  ({elapsed:.1f}s)")
        print(classification_report(y_test, preds, target_names=["Negative", "Positive"]))

    return results

# ── Confusion Matrix Plot ─────────────────────────────────────────────────────
def save_confusion_matrices(results, y_test):
    print("Saving confusion matrices...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (name, res) in zip(axes, results.items()):
        cm = confusion_matrix(y_test, res["preds"])
        disp = ConfusionMatrixDisplay(cm, display_labels=["Negative", "Positive"])
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(f"{name}\nAcc: {res['accuracy']*100:.2f}%", fontsize=11, fontweight="bold")
    plt.suptitle("Confusion Matrices – Model Comparison", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrices.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved to results/confusion_matrices.png")

# ── Accuracy Comparison Bar Chart ────────────────────────────────────────────
def save_accuracy_chart(results):
    names = list(results.keys())
    accs  = [r["accuracy"] * 100 for r in results.values()]

    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ["#2196F3", "#FF9800", "#4CAF50"]
    bars = ax.bar(names, accs, color=colors, edgecolor="white", width=0.5)
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f"{acc:.2f}%", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_ylim(85, 95)
    ax.set_title("Model Accuracy Comparison", fontsize=14, fontweight="bold", pad=12)
    ax.set_ylabel("Accuracy (%)")
    sns.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "model_comparison.png"), dpi=150)
    plt.close()
    print("  Saved to results/model_comparison.png")

# ── Save Best Model ───────────────────────────────────────────────────────────
def save_best_model(results, vectorizer):
    best_name = max(results, key=lambda k: results[k]["accuracy"])
    best      = results[best_name]
    print(f"\n✅ Best model: {best_name} ({best['accuracy']*100:.2f}%)")

    with open(os.path.join(MODEL_DIR, "best_model.pkl"), "wb") as f:
        pickle.dump(best["model"], f)
    with open(os.path.join(MODEL_DIR, "vectorizer.pkl"), "wb") as f:
        pickle.dump(vectorizer, f)
    print("  Model and vectorizer saved to model/")
    return best_name

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    df = load_data()
    save_eda_plots(df)

    print("\nVectorizing with TF-IDF...")
    vectorizer = TfidfVectorizer(max_features=50000, ngram_range=(1, 2), sublinear_tf=True)
    X = vectorizer.fit_transform(df["clean_review"])
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  Train: {X_train.shape[0]:,}  |  Test: {X_test.shape[0]:,}")

    results = train_models(X_train, X_test, y_train, y_test)
    save_confusion_matrices(results, y_test)
    save_accuracy_chart(results)
    best_name = save_best_model(results, vectorizer)

    print("\n── Summary ──────────────────────────────────────────────────────")
    print(f"{'Model':<25} {'Accuracy':>10}  {'Time':>8}")
    print("-" * 48)
    for name, res in results.items():
        marker = " ← best" if name == best_name else ""
        print(f"{name:<25} {res['accuracy']*100:>9.2f}%  {res['time']:>6.1f}s{marker}")
    print("\nAll results saved to results/")

if __name__ == "__main__":
    main()
