import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Load the transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize FAISS index
dimension = 384  # Model embedding dimension
index = faiss.IndexFlatL2(dimension)

# Load the job data from the JSON file
def load_jobs(file_name="../jobs.json"):
    file_path = os.path.join(os.path.dirname(__file__), file_name)
    with open(file_path, "r") as f:
        jobs = json.load(f)
    return jobs

# Convert job descriptions to embeddings
def embed_job_descriptions(jobs):
    job_embeddings = []
    for job in jobs:
        combined_description = f"{job['title']} at {job['company']}, located in {job['location']}. " \
                               f"Description: {job['description']}"
        embedding = model.encode(combined_description)  # Convert description to embeddings
        job_embeddings.append(embedding)
        print(f"Job: {job['title']} Embedding: {embedding[:5]}...")  # Print first 5 values of the embedding
    return job_embeddings

# Store embeddings in FAISS and verify the number of stored embeddings
def store_embeddings_in_faiss(job_embeddings):
    index.add(np.array(job_embeddings))  # Store all embeddings in FAISS
    
    # Save FAISS index to the app directory inside the project
    index_file_path = os.path.join(os.path.dirname(__file__), "job_faiss_index.index")
    faiss.write_index(index, index_file_path)  # Save FAISS index to disk
    
    print(f"Stored {len(job_embeddings)} job embeddings in FAISS.")
    print(f"FAISS index size: {index.ntotal} embeddings stored.")

# Retrieve some embeddings for inspection
def retrieve_embeddings(num_to_retrieve=5):
    D, I = index.search(np.random.rand(1, dimension).astype('float32'), k=num_to_retrieve)  # Dummy query
    print(f"Indices of retrieved embeddings: {I}")
    print(f"Distances of retrieved embeddings: {D}")

if __name__ == "__main__":
    jobs = load_jobs()  # Step 1: Load jobs
    job_embeddings = embed_job_descriptions(jobs)  # Step 2: Convert to embeddings
    store_embeddings_in_faiss(job_embeddings)  # Step 3: Store in FAISS
    
    # Step 4: Retrieve embeddings to verify
    retrieve_embeddings(5)
