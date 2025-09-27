import pandas as pd
from sentence_transformers import SentenceTransformer
import chromadb
import pickle

# 1. Load data and create combined texts
df = pd.read_csv("TNG.csv")
combined_texts = (
    df["who"].fillna('Unknown').astype(str) + ": " +
    df["text"].fillna('').astype(str) +
    " [Episode: " + df["Episode"].astype(str) + ", Scene: " + df["scenenumber"].astype(str) + "]"
).tolist()
with open("stng_chunks.pkl", "wb") as f:
    pickle.dump(combined_texts, f)

# 2. Embed texts
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(combined_texts, show_progress_bar=True)
with open("stng_embeddings.pkl", "wb") as f:
    pickle.dump(embeddings, f)

# 3. Store in persistent ChromaDB (disk-based for instant load in Streamlit)
client = chromadb.PersistentClient(path="stng_chroma_db")
collection = client.create_collection("tng_docs")

for i, (chunk, embedding) in enumerate(zip(combined_texts, embeddings)):
    collection.add(documents=[chunk], embeddings=[embedding.tolist()], ids=[str(i)])

print("Star Trek TNG database precomputed and saved!")
