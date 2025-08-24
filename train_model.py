import pandas as pd
import nltk
import re
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from nltk.corpus import stopwords

# Download stopwords (only first time needed)
nltk.download('stopwords')

# -----------------------------
# 1. Load Dataset
# -----------------------------
# Example: dataset must have 2 columns: "review" and "sentiment"
# sentiment = 1 (positive), 0 (negative)
data = pd.read_csv("IMDB Dataset.csv")  

# -----------------------------
# 2. Preprocess Text
# -----------------------------
def clean_text(text):
    text = text.lower()                                   # lowercase
    text = re.sub(r"[^a-zA-Z\s]", "", text)               # remove punctuation/numbers
    words = text.split()
    words = [w for w in words if w not in stopwords.words("english")]  
    return " ".join(words)

data["review"] = data["review"].apply(clean_text)

# -----------------------------
# 3. Split Data
# -----------------------------
X = data["review"]
y = data["sentiment"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -----------------------------
# 4. Convert text → vectors
# -----------------------------
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# -----------------------------
# 5. Train Model
# -----------------------------
model = LogisticRegression()
model.fit(X_train_vec, y_train)

print("✅ Training complete. Accuracy:", model.score(X_test_vec, y_test))

# -----------------------------
# 6. Save Model + Vectorizer
# -----------------------------
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("✅ Model and Vectorizer saved successfully.")
