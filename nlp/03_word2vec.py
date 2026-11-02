from gensim.models import Word2Vec
import nltk
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

# ── dataset — retail sentences ────────────────────────────
sentences = [
    ["rice", "basmati", "grain", "food", "store"],
    ["wheat", "flour", "grain", "food", "bread"],
    ["milk", "dairy", "drink", "fresh", "store"],
    ["apple", "fruit", "fresh", "healthy", "food"],
    ["banana", "fruit", "fresh", "healthy", "food"],
    ["mango", "fruit", "fresh", "sweet", "food"],
    ["soap", "clean", "wash", "hygiene", "store"],
    ["shampoo", "clean", "wash", "hygiene", "hair"],
    ["phone", "mobile", "device", "electronics", "store"],
    ["laptop", "computer", "device", "electronics", "store"],
    ["shirt", "cloth", "wear", "fashion", "store"],
    ["pants", "cloth", "wear", "fashion", "store"],
    ["customer", "buy", "store", "shop", "product"],
    ["customer", "complaint", "return", "refund", "store"],
    ["product", "price", "discount", "offer", "store"],
]

# ── train Word2Vec ─────────────────────────────────────────
model = Word2Vec(
    sentences=sentences,
    vector_size=50,   # each word = 50 numbers
    window=3,         # look 3 words left and right for context
    min_count=1,      # include words that appear at least once
    workers=4,        # parallel training
    epochs=100        # training iterations
)

print(f"Vocabulary size: {len(model.wv)}")
print(f"Vector size: {model.wv.vector_size}")
print()

# ── word vector ────────────────────────────────────────────
print("Vector for 'rice' (first 10 values):")
print(model.wv["rice"][:10])
print()

# ── similar words ──────────────────────────────────────────
print("Words similar to 'rice':")
for word, score in model.wv.most_similar("rice", topn=3):
    print(f"  {word:15} → similarity: {score:.3f}")
print()

print("Words similar to 'apple':")
for word, score in model.wv.most_similar("apple", topn=3):
    print(f"  {word:15} → similarity: {score:.3f}")
print()

print("Words similar to 'soap':")
for word, score in model.wv.most_similar("soap", topn=3):
    print(f"  {word:15} → similarity: {score:.3f}")
print()

# ── word similarity score ──────────────────────────────────
print("Similarity scores:")
print(f"  rice  vs wheat  : {model.wv.similarity('rice', 'wheat'):.3f}")   # both grains
print(f"  rice  vs soap   : {model.wv.similarity('rice', 'soap'):.3f}")    # unrelated
print(f"  apple vs banana : {model.wv.similarity('apple', 'banana'):.3f}") # both fruits
print(f"  phone vs laptop : {model.wv.similarity('phone', 'laptop'):.3f}") # both electronics
print()

# ── word arithmetic ────────────────────────────────────────
print("Word arithmetic:")
# fruit - apple + rice = ? (what is to rice what apple is to fruit?)
result = model.wv.most_similar(
    positive=["fruit", "rice"],
    negative=["apple"],
    topn=1
)
print(f"  fruit - apple + rice = {result[0][0]}")

# ── save model ─────────────────────────────────────────────
model.save("word2vec_retail.model")
print("\nModel saved.")
