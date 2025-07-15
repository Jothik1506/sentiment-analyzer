# 🎯 Sentiment Analyzer (IMDb & Twitter)

This project is a machine learning-based web app that performs **sentiment analysis** on movie reviews or tweets using trained models from IMDb and Twitter (Sentiment140) datasets.

---

## 🧠 What It Does

- Accepts a text input (review or tweet)
- Predicts sentiment: ✅ Positive / ❌ Negative (binary classification)
- Allows switching between **IMDb** and **Twitter** models
- Displays real-time output via a **Streamlit** web app

---

## 🧱 Tech Stack

- Python 3.x
- Streamlit
- Scikit-learn
- Pandas
- NLTK (for text preprocessing)
- Trained models: `LinearSVC` + `TF-IDF`

---

## 🚀 How to Run

### 1. Clone this repository

```bash
git clone https://github.com/Jothik1506/sentiment-analyzer.git
cd sentiment-analyzer
```

### 2. Create a virtual environment (optional but recommended)

```bash
python -m venv venv
venv\Scripts\activate  # On Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the app

```bash
streamlit run app.py
```

> App opens at `http://localhost:8501`

---

## 📁 Project Structure

```
sentiment-analyzer/
├── app.py                   # Streamlit frontend
├── model.py                 # Preprocessing + prediction functions
├── train_model.py           # For training the models
├── imdb_model.pkl           # IMDb trained model
├── twitter_model.pkl        # Twitter trained model
├── imdb_vectorizer.pkl
├── twitter_vectorizer.pkl
├── requirements.txt
```

---

## 📊 Model Accuracy

| Dataset | Accuracy |
|---------|----------|
| IMDb    | ~90%     |
| Twitter | ~80%     |

---

## 🔄 Future Improvements

- Emoji-aware predictions
- Neutral sentiment support
- Add deep learning models (e.g., BERT)

---

## 👤 Author

**Vanam Jothik Krishna Siva Naga Sai Kanth**  
GitHub: [Jothik1506](https://github.com/Jothik1506)
