"""Terminal chat loop for the UET Peshawar FAQ bot.

Usage: python src/predict.py
Type 'quit' or 'exit' to stop. Low-confidence inputs get a fallback reply.
"""
import os
import random
import sys

import joblib

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE, "model", "chatbot_model.joblib")
FALLBACK = ("Sorry, I am not sure about that. I can help with admissions, courses, "
            "exams, fees, hostel, library, scholarships, NCBC, and NAVTTC courses. "
            "Try asking in simple words.")


def load():
    if not os.path.exists(MODEL_PATH):
        print("Model not found. Run: python src/chatbot.py  (to train it first)")
        sys.exit(1)
    return joblib.load(MODEL_PATH)


def predict(bundle, text):
    model = bundle["model"]
    probs = model.predict_proba([text])[0]
    idx = probs.argmax()
    intent = model.classes_[idx]
    confidence = float(probs[idx])
    if confidence < 0.10:
        # 0.10 is deliberately low: with 18 intents, random chance is ~0.06,
        # and this small dataset spreads probability across similar intents,
        # so real questions often score 0.15-0.40 even when correct.
        return "unknown", confidence, FALLBACK
    reply = random.choice(bundle["responses"][intent])
    return intent, confidence, reply


def main():
    bundle = load()
    print("UET Peshawar FAQ Bot — type 'quit' to exit.")
    while True:
        try:
            user = input("You: ").strip()
        except EOFError:
            break
        if not user:
            continue
        if user.lower() in ("quit", "exit", "bye"):
            print("Bot: Goodbye! Good luck with your studies.")
            break
        intent, conf, reply = predict(bundle, user)
        print(f"Bot [{intent} | confidence {conf:.2f}]: {reply}")


if __name__ == "__main__":
    main()
