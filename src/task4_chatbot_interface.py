import streamlit as st
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import torch
import os

# Set up paths
BASE_DIR = r"C:\Users\hp\Desktop\Kifiya AIM\week 6\Technical content\workspace"
VECTOR_DIR = os.path.join(BASE_DIR, "vector_store")
INDEX_FILE = os.path.join(VECTOR_DIR, "faiss_index.bin")
METADATA_FILE = os.path.join(VECTOR_DIR, "metadata.pkl")

# Load FAISS index and metadata
try:
    index = faiss.read_index(INDEX_FILE)
    with open(METADATA_FILE, 'rb') as f:
        metadata = pickle.load(f)
    st.success(f"Loaded FAISS index with {index.ntotal} vectors and metadata with {len(metadata)} entries")
except Exception as e:
    st.error(f"Error loading FAISS index or metadata: {e}")
    st.stop()

# Initialize models
try:
    retriever_model = SentenceTransformer('all-MiniLM-L6-v2')
    # Explicitly set device to CPU
    generator = pipeline('text-generation', model='distilgpt2', device=-1)
    generator.model.to('cpu')  # Ensure model is on CPU
    st.success("Models initialized successfully")
except Exception as e:
    st.error(f"Error initializing models: {e}")
    st.stop()

# RAG pipeline
def rag_pipeline(query, top_k=3):
    try:
        with torch.no_grad():  # Reduce memory usage
            query_embedding = retriever_model.encode([query])[0]
            distances, indices = index.search(np.array([query_embedding]), top_k)
            retrieved_chunks = [(metadata[idx]['complaint_id'], metadata[idx]['product'], metadata[idx]['chunk']) 
                               for idx in indices[0]]
            context = "\n".join([chunk for _, _, chunk in retrieved_chunks])
            prompt = f"List key issues from these complaints about '{query}' in 50 words:\n{context}\n\nSummary:"
            response = generator(prompt, max_new_tokens=100, num_return_sequences=1, truncation=True, temperature=0.7)[0]['generated_text']
            summary = response.split("Summary:")[-1].strip() if "Summary:" in response else response
            return retrieved_chunks, summary
    except Exception as e:
        st.error(f"Error processing query: {e}")
        return None, None

# Streamlit interface
st.title("CrediTrust Complaint Analysis Chatbot")
st.write("Enter a query about customer complaints (e.g., 'What are common issues with credit card fraud?')")

query = st.text_input("Your Query:")
if st.button("Submit"):
    if query:
        retrieved_chunks, response = rag_pipeline(query)
        if retrieved_chunks and response:
            st.subheader("Retrieved Complaints:")
            for i, (complaint_id, product, chunk) in enumerate(retrieved_chunks, 1):
                st.write(f"**Chunk {i}** (Complaint ID: {complaint_id}, Product: {product}): {chunk[:200]}...")
            st.subheader("Summary:")
            st.write(response)
    else:
        st.write("Please enter a query.")