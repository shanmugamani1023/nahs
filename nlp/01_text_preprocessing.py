import nltk
import string

nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("punkt_tab", quiet=True)

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

# ── sample text ───────────────────────────────────────────
text = "Hello! I am Shanmugamani. I am Learning NLP and Building AI Systems in 2024."

print("Original:")
print(text)
print()

# ── step 1: lowercase ─────────────────────────────────────
text_lower = text.lower()
print("Step 1 - Lowercase:")
print(text_lower)
print()

# ── step 2: remove punctuation ────────────────────────────
text_clean = text_lower.translate(str.maketrans("", "", string.punctuation))
print("Step 2 - Remove Punctuation:")
print(text_clean)
print()

# ── step 3: tokenize (split into words) ───────────────────
tokens = word_tokenize(text_clean)
print("Step 3 - Tokenize:")
print(tokens)
print()

# ── step 4: remove stopwords ──────────────────────────────
stop_words = set(stopwords.words("english"))
filtered = [word for word in tokens if word not in stop_words]
print("Step 4 - Remove Stopwords:")
print(filtered)
print()

# ── step 5a: stemming ─────────────────────────────────────
stemmer = PorterStemmer()
stemmed = [stemmer.stem(word) for word in filtered]
print("Step 5a - Stemming:")
print(stemmed)
print()

# ── step 5b: lemmatization ────────────────────────────────
lemmatizer = WordNetLemmatizer()
lemmatized = [lemmatizer.lemmatize(word) for word in filtered]
print("Step 5b - Lemmatization:")
print(lemmatized)
print()

# ── full pipeline as function ─────────────────────────────
def preprocess(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in stop_words]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return tokens

print("Full Pipeline:")
print(preprocess("I am Running and Playing football in the Gardens!"))
