import nltk
import string
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("punkt_tab", quiet=True)

# ── dataset ───────────────────────────────────────────────
data = {
    "text": [
        "Win free iPhone now click here",
        "Congratulations you won 1 million dollars",
        "Free money claim your prize today",
        "Click here to get free gift card",
        "You have been selected for cash prize",
        "Buy cheap medicine online now",
        "Hot singles in your area click now",
        "Urgent your account will be closed",
        "Meeting at 3pm tomorrow",
        "Can you send me the report please",
        "Lunch at 1pm today?",
        "Project deadline is next Friday",
        "Please review the attached document",
        "See you at the office tomorrow",
        "Call me when you are free",
        "The team meeting is postponed",
    ],
    "label": [
        "spam", "spam", "spam", "spam", "spam", "spam", "spam", "spam",
        "ham",  "ham",  "ham",  "ham",  "ham",  "ham",  "ham",  "ham",
    ]
}

df = pd.DataFrame(data)
print("Dataset:")
print(df["label"].value_counts())
print()

# ── preprocessing ─────────────────────────────────────────
stop_words  = set(stopwords.words("english"))
lemmatizer  = WordNetLemmatizer()

def preprocess(text):
    text   = text.lower()
    text   = text.translate(str.maketrans("", "", string.punctuation))
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in stop_words]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return " ".join(tokens)

df["clean"] = df["text"].apply(preprocess)
print("Cleaned text sample:")
print(df[["text", "clean"]].head(3).to_string())
print()

# ── TF-IDF — convert text to numbers ──────────────────────
tfidf = TfidfVectorizer()
X = tfidf.fit_transform(df["clean"])
y = df["label"]

print(f"TF-IDF shape: {X.shape}")  # (samples, unique_words)
print(f"Vocabulary size: {len(tfidf.vocabulary_)}")
print()

# ── train / test split ────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ── Naive Bayes classifier ────────────────────────────────
model = MultinomialNB()
model.fit(X_train, y_train)

# ── evaluate ──────────────────────────────────────────────
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print()
print(classification_report(y_test, y_pred))

# ── predict new messages ──────────────────────────────────
def predict(text):
    clean  = preprocess(text)
    vector = tfidf.transform([clean])
    result = model.predict(vector)[0]
    proba  = model.predict_proba(vector)[0]
    return result, max(proba)

print("── Predictions ──────────────────────")
tests = [
    "Win free cash now click here",
    "Can we reschedule the meeting?",
    "Claim your free prize today",
    "See you tomorrow at office",
]
for t in tests:
    label, confidence = predict(t)
    print(f"[{label.upper()}] ({confidence:.0%}) → {t}")
