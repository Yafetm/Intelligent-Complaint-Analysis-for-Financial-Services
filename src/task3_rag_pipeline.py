import pandas as pd
import os
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# Set up paths
BASE_DIR = r"C:\Users\hp\Desktop\Kifiya AIM\week 6\Technical content\workspace"
VECTOR_DIR = os.path.join(BASE_DIR, "vector_store")
INDEX_FILE = os.path.join(VECTOR_DIR, "faiss_index.bin")
METADATA_FILE = os.path.join(VECTOR_DIR, "metadata.pkl")

# Step 1: Load FAISS index and metadata
print("=== Loading FAISS Index and Metadata ===")
try:
    index = faiss.read_index(INDEX_FILE)
    with open(METADATA_FILE, 'rb') as f:
        metadata = pickle.load(f)
    print(f"Loaded FAISS index with {index.ntotal} vectors")
    print(f"Loaded metadata with {len(metadata)} entries")
except Exception as e:
    print(f"Error loading FAISS index or metadata: {e}")
    exit()

# Step 2: Initialize retriever
print("\n=== Initializing Retriever ===")
retriever_model = SentenceTransformer('all-MiniLM-L6-v2')

# Step 3: Initialize language model
print("\n=== Initializing Language Model ===")
try:
    generator = pipeline('text-generation', model='distilgpt2')
except Exception as e:
    print(f"Error initializing distilgpt2: {e}")
    exit()

# Step 4: RAG pipeline
def rag_pipeline(query, top_k=5):
    print(f"\n=== Processing Query: {query} ===")
    try:
        # Encode query
        query_embedding = retriever_model.encode([query])[0]
        # Search FAISS index
        distances, indices = index.search(np.array([query_embedding]), top_k)
        # Retrieve chunks and metadata
        retrieved_chunks = [(metadata[idx]['complaint_id'], metadata[idx]['product'], metadata[idx]['chunk']) 
                           for idx in indices[0]]
        print("\nRetrieved Chunks:")
        for i, (complaint_id, product, chunk) in enumerate(retrieved_chunks, 1):
            print(f"Chunk {i} (Complaint ID: {complaint_id}, Product: {product}): {chunk[:100]}...")
        
        # Combine chunks for context
        context = "\n".join([chunk for _, _, chunk in retrieved_chunks])
        # Simplified prompt
        prompt = f"Summarize the issues from these complaints about '{query}':\n{context}\n\nSummary (50-100 words):"
        response = generator(prompt, max_new_tokens=150, num_return_sequences=1, truncation=True)[0]['generated_text']
        # Extract summary after "Summary:"
        summary = response.split("Summary (50-100 words):")[-1].strip() if "Summary (50-100 words):" in response else response
        print("\nGenerated Summary:")
        print(summary)
        return summary
    except Exception as e:
        print(f"Error processing query: {e}")
        return None

# Step 5: Test queries
test_queries = [
    "What are common issues with credit card fraud?",
    "Problems with savings account disputes",
    "Issues with buy now pay later services"
]
print("\n=== Testing RAG Pipeline ===")
for query in test_queries:
    rag_pipeline(query)

# Save test outputs
try:
    with open(os.path.join(BASE_DIR, "rag_test_outputs.txt"), "w") as f:
        for query in test_queries:
            response = rag_pipeline(query)
            f.write(f"Query: {query}\nResponse: {response}\n\n")
    print("\nTest outputs saved to rag_test_outputs.txt")
except Exception as e:
    print(f"Error saving test outputs: {e}")