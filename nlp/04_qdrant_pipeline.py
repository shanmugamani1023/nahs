import pandas as pd
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

# ── STEP 1: LOAD CSV ──────────────────────────────────────
print("Step 1: Loading CSV...")
df = pd.read_csv("products.csv")
print(f"Loaded {len(df)} products")
print(df.head(3))
print()

# ── STEP 2: PREPROCESS ────────────────────────────────────
print("Step 2: Preprocessing...")

def preprocess(row):
    fields = [
        str(row.get("Product Name", "")),
        str(row.get("Features", "")),
        str(row.get("Description", "")),
        str(row.get("Tags", ""))
    ]
    fields = [f for f in fields if f != "nan"]
    text = " ".join(fields)
    text = text.lower()
    text = " ".join(text.split())  # remove extra spaces
    return text

df["combined_text"] = df.apply(preprocess, axis=1)
print("Sample combined text:")
print(df["combined_text"][0])
print()

# ── STEP 3: GENERATE EMBEDDINGS ───────────────────────────
print("Step 3: Generating embeddings...")
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(df["combined_text"].tolist(), show_progress_bar=True)
print(f"Embeddings shape: {embeddings.shape}")  # (50, 384)
print()

# ── STEP 4: STORE IN QDRANT ───────────────────────────────
print("Step 4: Storing in Qdrant...")
client = QdrantClient(":memory:")  # in-memory for testing

# create collection
client.create_collection(
    collection_name="products",
    vectors_config=VectorParams(
        size=384,
        distance=Distance.COSINE
    )
)

# create points
points = []
for idx, row in df.iterrows():
    points.append(PointStruct(
        id=idx,
        vector=embeddings[idx].tolist(),
        payload={
            "name":     row["Product Name"],
            "barcode":  str(row["Barcode"]),
            "features": str(row["Features"]),
            "tags":     str(row["Tags"])
        }
    ))

# upsert all points
client.upsert(collection_name="products", points=points)
print(f"Stored {len(points)} products in Qdrant")
print()

# ── STEP 5: RETRIEVE ──────────────────────────────────────
print("Step 5: Retrieving...")

def search(query, top_k=3, threshold=0.3):
    query_vector = model.encode(query).tolist()
    results = client.query_points(
        collection_name="products",
        query=query_vector,
        limit=top_k,
        score_threshold=threshold
    ).points
    return results

# test queries
queries = [
    "long grain rice",
    "something healthy for breakfast",
    "cleaning product for bathroom",
    "snack for evening",
    "dairy product protein",
    "writing stationery",
]

for query in queries:
    print(f"Query: '{query}'")
    results = search(query)
    if results:
        for r in results:
            print(f"  [{r.score:.3f}] {r.payload['name']} ({r.payload['barcode']})")
    else:
        print("  No results found")
    print()

# ── STEP 6: EVALUATE RETRIEVAL QUALITY ───────────────────
print("Step 6: Evaluating retrieval quality...")
print("─" * 50)

# ground truth — what SHOULD be returned for each query
ground_truth = {
    "long grain rice":              ["BR001"],           # Basmati Rice
    "something healthy for breakfast": ["GT001", "MA001"],  # Green Tea, Milk
    "cleaning product for bathroom":   ["HC001"],           # Harpic
    "snack for evening":            ["LC001", "KK001"],  # Lays, Kurkure
    "dairy product protein":        ["EF001", "PF001"],  # Eggs, Paneer
    "writing stationery":           ["NB001", "PR001"],  # Notebook, Pen
}

hit_count  = 0
total      = len(ground_truth)

for query, expected_barcodes in ground_truth.items():
    results   = search(query, top_k=3)
    retrieved = [r.payload["barcode"] for r in results]

    # hit rate — did we get at least 1 correct?
    hit = any(b in retrieved for b in expected_barcodes)
    if hit:
        hit_count += 1

    status = "✓ HIT" if hit else "✗ MISS"
    print(f"{status} | Query: '{query}'")
    print(f"       Expected: {expected_barcodes}")
    print(f"       Retrieved: {retrieved}")
    print()

hit_rate = hit_count / total
print(f"─" * 50)
print(f"Hit Rate: {hit_count}/{total} = {hit_rate:.0%}")
