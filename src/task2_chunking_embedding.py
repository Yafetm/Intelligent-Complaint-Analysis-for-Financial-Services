import pandas as pd
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle

# Set up paths
BASE_DIR = r"C:\Users\hp\Desktop\Kifiya AIM\week 6\Technical content\workspace"
DATA_DIR = os.path.join(BASE_DIR, "data")
VECTOR_DIR = os.path.join(BASE_DIR, "vector_store")
INPUT_FILE = os.path.join(DATA_DIR, "filtered_complaints.csv")
INDEX_FILE = os.path.join(VECTOR_DIR, "faiss_index.bin")
METADATA_FILE = os.path.join(VECTOR_DIR, "metadata.pkl")

# Create vector_store directory
os.makedirs(VECTOR_DIR, exist_ok=True)

# Step 1: Load filtered dataset
print("=== Loading Filtered Dataset ===")
try:
    df = pd.read_csv(INPUT_FILE)
    print(f"Dataset loaded successfully. Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit()

# Step 2: Text chunking
print("\n=== Text Chunking ===")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    length_function=len
)

# Chunk narratives and store metadata
chunks = []
metadata = []
for idx, row in df.iterrows():
    complaint_id = row['Complaint ID']
    product = row['Product']
    narrative = row['cleaned_narrative']
    split_texts = text_splitter.split_text(narrative)
    chunks.extend(split_texts)
    metadata.extend([{'complaint_id': complaint_id, 'product': product, 'chunk': text} for text in split_texts])

print(f"Total chunks created: {len(chunks)}")
print("\nSample chunks:")
for i, chunk in enumerate(chunks[:3], 1):
    print(f"Chunk {i}: {chunk[:100]}... (Length: {len(chunk)})")
    print(f"Metadata {i}: {metadata[i-1]}")

# Step 3: Generate embeddings
print("\n=== Generating Embeddings ===")
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(chunks, show_progress_bar=True)
print(f"Embeddings shape: {embeddings.shape}")
print(f"Embedding dimension: {embeddings.shape[1]}")

# Step 4: Create and save FAISS index
print("\n=== Creating FAISS Index ===")
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)
print(f"FAISS index size: {index.ntotal}")

# Save FAISS index
faiss.write_index(index, INDEX_FILE)
print(f"FAISS index saved to {INDEX_FILE}")

# Save metadata
with open(METADATA_FILE, 'wb') as f:
    pickle.dump(metadata, f)
print(f"Metadata saved to {METADATA_FILE}")

# Step 5: Verify a sample embedding
print("\n=== Sample Embedding Verification ===")
sample_chunk = chunks[0]
sample_embedding = model.encode([sample_chunk])[0]
print(f"Sample chunk: {sample_chunk[:100]}...")
print(f"Sample embedding (first 5 values): {sample_embedding[:5]}")