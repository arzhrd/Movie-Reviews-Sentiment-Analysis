import pickle

# -----------------------------
# 1. Load Model + Vectorizer
# -----------------------------
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# -----------------------------
# 2. Predict Function
# -----------------------------
def predict_sentiment(text):
    text_vec = vectorizer.transform([text])
    prediction = model.predict(text_vec)[0]
    return "Positive 😊" if prediction == 1 else "Negative 😡"

# -----------------------------
# 3. Test Prediction
# -----------------------------
sample_review = input("Enter a movie review: ")
print("Sentiment:", predict_sentiment(sample_review))
