"""
Sentiment Prediction – load saved model and predict on new text.

Usage:
    python src/predict.py "This movie was absolutely brilliant!"
    python src/predict.py  (interactive mode)
"""

import os, sys, re, pickle

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR  = os.path.join(BASE_DIR, "model")

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

def clean_text(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = text.lower()
    tokens = [t for t in text.split() if t not in STOPWORDS and len(t) > 2]
    return " ".join(tokens)

def load_model():
    model_path = os.path.join(MODEL_DIR, "best_model.pkl")
    vec_path   = os.path.join(MODEL_DIR, "vectorizer.pkl")
    if not os.path.exists(model_path):
        print("❌ Model not found. Please run `python src/train.py` first.")
        sys.exit(1)
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(vec_path, "rb") as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

def predict(text: str, model, vectorizer) -> dict:
    cleaned = clean_text(text)
    X = vectorizer.transform([cleaned])
    label = model.predict(X)[0]
    # Confidence via decision function or predict_proba
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0]
        confidence = proba[label]
    elif hasattr(model, "decision_function"):
        score = model.decision_function(X)[0]
        # sigmoid approximation
        import math
        confidence = 1 / (1 + math.exp(-abs(score)))
    else:
        confidence = None

    sentiment = "POSITIVE 😊" if label == 1 else "NEGATIVE 😞"
    return {"sentiment": sentiment, "label": label, "confidence": confidence}

def main():
    model, vectorizer = load_model()

    if len(sys.argv) > 1:
        text = " ".join(sys.argv[1:])
        result = predict(text, model, vectorizer)
        print(f"\nReview    : {text}")
        print(f"Sentiment : {result['sentiment']}")
        if result["confidence"]:
            print(f"Confidence: {result['confidence']*100:.1f}%")
    else:
        print("=" * 55)
        print("  IMDB Sentiment Classifier – Interactive Mode")
        print("  Type 'quit' to exit")
        print("=" * 55)
        while True:
            text = input("\nEnter a movie review:\n> ").strip()
            if text.lower() in ("quit", "exit", "q"):
                print("Goodbye!")
                break
            if not text:
                continue
            result = predict(text, model, vectorizer)
            print(f"\n  → Sentiment : {result['sentiment']}")
            if result["confidence"]:
                print(f"  → Confidence: {result['confidence']*100:.1f}%")

if __name__ == "__main__":
    main()
