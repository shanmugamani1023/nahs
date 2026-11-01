# NLP to LLM — Questions & Answers

---

## 01 — Text Preprocessing

**Q: What is the difference between Stemming and Lemmatization?**

Stemming cuts word endings without a dictionary — fast but rough.
Lemmatization looks up a real dictionary (WordNet) — slower but accurate.

| | Stemming | Lemmatization |
|---|---|---|
| Speed | Fast | Slower |
| Accuracy | Rough (`studies → studi`) | Accurate (`studies → study`) |
| Uses dictionary | No | Yes |
| Use when | Speed matters | Accuracy matters |

---

**Q: What is the WordNet dictionary?**

WordNet is a large English language database built by Princeton University.
It stores word relationships — base forms, synonyms, verb groups.

```
run  → base form of: running, runs, ran
good → base form of: better, best
system → base form of: systems
```

Downloaded via: `nltk.download("wordnet")`
Stored at: `~/.nltk_data/corpora/wordnet/`

Stemming = scissors (just cuts word endings)
Lemmatization = dictionary (looks up real root word)

---

**Q: What are stopwords?**

Common words that carry no meaning — removed before processing.

Examples: `I, am, the, is, at, and, in, of, to, a, an`

```python
from nltk.corpus import stopwords
stop_words = set(stopwords.words("english"))
filtered = [word for word in tokens if word not in stop_words]
```

---

## 02 — Spam Classifier (TF-IDF + Naive Bayes)

**Q: What is TF-IDF and how does it work?**

TF-IDF converts text into numbers representing word importance.

**TF (Term Frequency)** — how often word appears in THIS message:
```
Message: "free iPhone free gift free" (5 words)
TF("free") = 3/5 = 0.6
```

**IDF (Inverse Document Frequency)** — how rare word is across ALL messages:
```
"free"  in 6/16 messages → IDF = log(16/6) = 0.98  (common in spam)
"iPhone" in 1/16 messages → IDF = log(16/1) = 2.77  (rare = more important)
"the"   in 15/16 messages → IDF = log(16/15) = 0.06 (ignore)
```

**TF-IDF = TF × IDF:**
```
"free"   → 0.6 × 0.98 = 0.59
"iPhone" → 0.2 × 2.77 = 0.55
"the"    → 0.9 × 0.06 = 0.05 ← ignored
```

One line: TF-IDF gives high score to words frequent in THIS document but rare across ALL documents.

---

**Q: How does Naive Bayes learn the spam pattern?**

It learns by counting word occurrences per class during training.

**Training — counts words:**
```
Word     | In SPAM | In HAM
---------|---------|-------
free     |    5    |   0
win      |    3    |   0
meeting  |    0    |   4
report   |    0    |   3
```

**Calculates probability:**
```
P("free" | SPAM) = high
P("free" | HAM)  = very low
```

**Prediction — multiplies probabilities:**
```
New message: "Win free money now"

P(SPAM) × P("win"|SPAM) × P("free"|SPAM) = 0.252
P(HAM)  × P("win"|HAM)  × P("free"|HAM)  = 0.000001

0.252 >> 0.000001 → SPAM ✓
```

**Why "Naive"?**
It assumes every word is independent — ignores word relationships.
"free prize" → treats "free" and "prize" as separate words.
Naive assumption, but works surprisingly well in practice.

**Simple analogy:**
You've seen 100 emails. Every spam had "free" and "win". Every normal email had "meeting". New email says "free win" → you immediately think SPAM. Naive Bayes does exactly this.

---
