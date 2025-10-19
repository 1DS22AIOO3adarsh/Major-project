import os
import fitz  # PyMuPDF
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import warnings

# Suppress a common warning from the transformers library
warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub.file_download")

print("🚀 Initializing FULL PAPER processing and embedding pipeline...")

# --- Configuration ---
ACCEPTED_FOLDER = 'IEEE'
REJECTED_FOLDER = 'arXiv/new-2'
CHUNK_SIZE = 400  # Number of words per chunk
CHUNK_OVERLAP = 50 # Number of words to overlap between chunks

# --- 1. Load the BGE-M3 model ---
print("Loading BAAI/bge-m3 embedding model...")
model = SentenceTransformer('BAAI/bge-m3')
print("✅ Model loaded successfully.")

def extract_full_text(pdf_path):
    """Extracts the full plain text from a PDF document."""
    try:
        doc = fitz.open(pdf_path)
        full_text = ""
        for page in doc:
            full_text += page.get_text("text")
        doc.close()
        return full_text
    except Exception as e:
        print(f"   - ⚠️  Warning: Could not process {os.path.basename(pdf_path)}. Error: {e}")
        return None

def chunk_text(text, chunk_size, overlap):
    """Splits text into overlapping chunks."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunks.append(" ".join(words[i:i + chunk_size]))
    return chunks

# --- 2. The Main Processing Loop ---
embeddings = []
labels = []
folders_to_process = {ACCEPTED_FOLDER: 1, REJECTED_FOLDER: 0}

for folder, label in folders_to_process.items():
    print(f"⚙️  Processing files in '{folder}' (Label: {label})...")
    if not os.path.isdir(folder):
        print(f"   - ❌ Error: Folder '{folder}' not found.")
        continue
    
    for filename in os.listdir(folder):
        if filename.lower().endswith(".pdf"):
            path = os.path.join(folder, filename)
            full_text = extract_full_text(path)
            
            if full_text and len(full_text) > 100:
                # Step A: Chunk the full text
                text_chunks = chunk_text(full_text, CHUNK_SIZE, CHUNK_OVERLAP)
                
                # Step B: Embed all chunks at once (more efficient)
                chunk_embeddings = model.encode(text_chunks, show_progress_bar=False)
                
                # Step C: Average the embeddings to get a single vector for the whole paper
                mean_embedding = np.mean(chunk_embeddings, axis=0)
                
                embeddings.append(mean_embedding)
                labels.append(label)
                print(f"   - Embedded: {filename} ({len(text_chunks)} chunks)")

# --- 3. Save the Final Dataset ---
if embeddings:
    np.save('final_full_paper_embeddings.npy', np.array(embeddings))
    pd.DataFrame(labels, columns=['status']).to_csv('final_full_paper_labels.csv', index=False)
    print("\n✅ Success! Your full-paper dataset is now ready for training.")
    print(f"   - Embeddings saved to 'final_full_paper_embeddings.npy'")
    print(f"   - Labels saved to 'final_full_paper_labels.csv'")
else:
    print("\n❌ No data was created. Check your PDF folders.")