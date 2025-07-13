import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    cleaned = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(cleaned)

import pickle

def load_model_and_vectorizer(dataset):
    if dataset == 'imdb':
        model_file = 'imdb_model.pkl'
        vect_file = 'imdb_vectorizer.pkl'
    elif dataset == 'twitter':
        model_file = 'twitter_model.pkl'
        vect_file = 'twitter_vectorizer.pkl'
    else:
        raise ValueError("Unsupported dataset")

    with open(model_file, "rb") as f:
        model = pickle.load(f)

    with open(vect_file, "rb") as f:
        vectorizer = pickle.load(f)

    return model, vectorizer

def predict_sentiment(text, dataset):
    model, vectorizer = load_model_and_vectorizer(dataset)

    cleaned = clean_text(text)
    vect = vectorizer.transform([cleaned])
    prediction = model.predict(vect)[0]

    if dataset == 'twitter':
        if prediction == 0:
            return "Negative"
        elif prediction == 1:
            return "Neutral"
        else:
            return "Positive"
    else:  # imdb
        return "Positive" if prediction == 1 else "Negative"
    if not cleaned.strip():
        return "Unable to analyze: input has no valid words"

