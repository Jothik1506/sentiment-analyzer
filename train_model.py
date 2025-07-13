import pandas as pd
from model import clean_text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
import pickle

# SELECT dataset: 'imdb' or 'twitter'
dataset = 'twitter'  # change to 'imdb' if needed

if dataset == 'imdb':
    df = pd.read_csv(r"C:\Users\vanam\Downloads\pPython\IMDB Dataset.csv")
    df.columns = ['text', 'sentiment']
    df['label'] = df['sentiment'].map({'positive': 1, 'negative': 0})
    model_file = "imdb_model.pkl"
    vect_file = "imdb_vectorizer.pkl"

elif dataset == 'twitter':
    df = pd.read_csv(r"C:\Users\vanam\Downloads\pPython\X Data set.csv", encoding='latin-1', header=None)
    df = df[[0, 5]]
    df.columns = ['label', 'text']
    df = df[df['label'].isin([0, 4])]  # drop neutral
    df['label'] = df['label'].map({0: 0, 4: 1})  # 0=Negative, 1=Positive
    model_file = "twitter_model.pkl"
    vect_file = "twitter_vectorizer.pkl"

else:
    raise ValueError("Invalid dataset selection")

# Clean and prepare data
df = df.dropna()
df['cleaned_text'] = df['text'].apply(clean_text)

X = df['cleaned_text']
y = df['label']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vect = vectorizer.fit_transform(X_train)
X_test_vect = vectorizer.transform(X_test)

# Train model
model = LinearSVC()
model.fit(X_train_vect, y_train)

# Evaluate model
y_pred = model.predict(X_test_vect)
print("\n📊 Model Evaluation Report:")
print(classification_report(y_test, y_pred))

# Save model and vectorizer
with open(model_file, "wb") as f:
    pickle.dump(model, f)

with open(vect_file, "wb") as f:
    pickle.dump(vectorizer, f)

print(f"\n✅ Model trained and saved for {dataset.upper()}")