import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords

# -----------------------------
# Setup
# -----------------------------
# Download NLTK stopwords (runs once on the cloud environment)
nltk.download('stopwords')

# -----------------------------
# Text Preprocessing Function
# (MUST be identical to the one used in training)
# -----------------------------
def clean_text(text):
    """
    Cleans the input text by lowercasing, removing punctuation/numbers,
    and removing stopwords.
    """
    text = text.lower()                                 # lowercase
    text = re.sub(r"[^a-zA-Z\s]", "", text)              # remove punctuation/numbers
    words = text.split()
    words = [w for w in words if w not in stopwords.words("english")]
    return " ".join(words)

# -----------------------------
# Load Model + Vectorizer
# -----------------------------
# Use st.cache_resource to load the models only once
@st.cache_resource
def load_models():
    """
    Loads the saved model and vectorizer from disk.
    Uses Streamlit's cache to avoid reloading on every interaction.
    """
    try:
        with open("model.pkl", "rb") as f:
            model = pickle.load(f)
        
        with open("vectorizer.pkl", "rb") as f:
            vectorizer = pickle.load(f)
            
        return model, vectorizer
    except FileNotFoundError:
        st.error("Error: model.pkl or vectorizer.pkl not found.")
        st.error("Please make sure these files are in the same directory as app.py.")
        return None, None
    except Exception as e:
        st.error(f"An error occurred while loading the models: {e}")
        return None, None

model, vectorizer = load_models()

# -----------------------------
# Streamlit App Interface
# -----------------------------
st.set_page_config(layout="wide", page_title="Movie Review Sentiment Analyzer")
st.title("Movie Review Sentiment Analyzer 🎬")
st.markdown("Enter a movie review below to predict whether it's **Positive** or **Negative**.")

# Only proceed if models were loaded successfully
if model and vectorizer:
    # User input text area
    review_text = st.text_area(
        "Enter your review:", 
        height=150, 
        placeholder="This movie was absolutely fantastic! The acting was superb and the plot was gripping."
    )

    # Analyze button
    if st.button("Analyze Sentiment", type="primary"):
        if review_text.strip():  # Check if input is not just whitespace
            try:
                # 1. Clean the input text
                cleaned_review = clean_text(review_text)
                
                # 2. Vectorize the cleaned text
                review_vec = vectorizer.transform([cleaned_review])
                
                # 3. Predict sentiment
                prediction = model.predict(review_vec)[0]
                
                # 4. Get prediction probability (confidence)
                probability = model.predict_proba(review_vec)[0]
                
                # 5. Display the result
                if prediction == 1:
                    sentiment = "Positive"
                    emoji = "😊"
                    confidence = probability[1]
                    st.success(f"**Sentiment: {sentiment} {emoji}**")
                    st.metric(label="Confidence", value=f"{confidence:.2%}")
                else:
                    sentiment = "Negative"
                    emoji = "😡"
                    confidence = probability[0]
                    st.error(f"**Sentiment: {sentiment} {emoji}**")
                    st.metric(label="Confidence", value=f"{confidence:.2%}")

            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")
        else:
            # Show a warning if the text area is empty
            st.warning("Please enter a review to analyze.")
else:
    st.info("The application could not load the necessary model files to start.")
