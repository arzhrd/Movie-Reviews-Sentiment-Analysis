

---

# 🎬 Movie Reviews Sentiment Analysis

A machine learning project to classify **movie reviews** as **Positive 😀** or **Negative 😡** using **Natural Language Processing (NLP)** and **Logistic Regression**.

---

## 📌 Features

* Preprocesses movie reviews (tokenization, stopword removal, lowercasing).
* Uses **TF-IDF Vectorizer** for feature extraction.
* Trains a **Logistic Regression** classifier.
* Saves trained **model** and **vectorizer** (`model.pkl`, `vectorizer.pkl`).
* Allows interactive prediction from user input.

---

## 📂 Project Structure

```
Movie Reviews Sentiment Analysis/
│── train_model.py      # Train the model
│── predict.py          # Predict sentiment of new reviews
│── model.pkl           # Saved trained model
│── vectorizer.pkl      # Saved TF-IDF vectorizer
│── requirements.txt    # Dependencies
│── README.md           # Project documentation
│── venv/               # Virtual environment (not uploaded to GitHub)
```

---

## ⚙️ Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/movie-reviews-sentiment-analysis.git
   cd movie-reviews-sentiment-analysis
   ```

2. Create a virtual environment and activate it:

   ```bash
   python3 -m venv venv
   source venv/bin/activate   # Linux/Mac
   venv\Scripts\activate      # Windows
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 Usage

### 1. Train the Model

```bash
python train_model.py
```

* This will train the classifier and save:

  * `model.pkl` (trained model)
  * `vectorizer.pkl` (TF-IDF vectorizer)

### 2. Predict Sentiment

```bash
python predict.py
```

* Enter any movie review (e.g., *"This movie was really amazing!"*).
* The model will return:

  * **Positive 😀**
  * **Negative 😡**

---

## 📊 Example Output

```bash
$ python predict.py
Enter a movie review: This movie was really amazing!
Sentiment: Positive 😀
```

```bash
$ python predict.py
Enter a movie review: The plot was boring and predictable.
Sentiment: Negative 😡
```

---

## 📦 Requirements

* Python 3.8+
* scikit-learn
* nltk

Install all dependencies:

```bash
pip install -r requirements.txt
```

---

## 🔮 Future Improvements

* Add **neutral sentiment** classification.
* Deploy as a **Flask/Django web app**.
* Create a **Streamlit dashboard** for easy use.

---

## 👨‍💻 Author

**Arshad** ✨
📌 *Passionate about Machine Learning & AI*

---
