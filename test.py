import fitz  # PyMuPDF
import numpy as np
import joblib
from sentence_transformers import SentenceTransformer
import warnings

# Suppress common warnings for a cleaner output
warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub.file_download")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.base")

print("🚀 Initializing prediction script...")

# --- Configuration ---
PDF_TO_TEST = 'NOISE LEVEL ESTIMATION AND MODEL FILTERING FOR THROAT CANCER DETECTION.pdf' # <--- CHANGE THIS to the name of your PDF file
MODEL_FILE = 'ai_reviewer_model_tuned.joblib'
EMBEDDING_MODEL_NAME = 'BAAI/bge-m3'
CHUNK_SIZE = 400
CHUNK_OVERLAP = 50

# --- 1. Load the Trained AI Model and the Embedding Model ---
print("Loading trained AI reviewer model...")
try:
    classifier_model = joblib.load(MODEL_FILE)
except FileNotFoundError:
    print(f"❌ Error: Model file '{MODEL_FILE}' not found. Please run 'train_model.py' first.")
    exit()

print("Loading sentence embedding model...")
try:
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
except Exception as e:
    print(f"❌ Error loading embedding model. Ensure an internet connection. Details: {e}")
    exit()
print("✅ Models loaded successfully.")

# --- 2. Create the Same Processing Functions from Training ---
def extract_full_text(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        full_text = "".join(page.get_text("text") for page in doc)
        doc.close()
        return full_text
    except Exception as e:
        print(f"❌ Error reading PDF file '{pdf_path}'. Error: {e}")
        return None

def chunk_text(text, chunk_size, overlap):
    words = text.split()
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size - overlap)]

# --- 3. Process the New Paper ---
print(f"\n⚙️  Processing '{PDF_TO_TEST}'...")
full_text = extract_full_text(PDF_TO_TEST)

if full_text and len(full_text) > 100:
    # A. Chunk the text
    text_chunks = chunk_text(full_text, CHUNK_SIZE, CHUNK_OVERLAP)
    
    # B. Create embeddings for all chunks
    chunk_embeddings = embedding_model.encode(text_chunks, show_progress_bar=False)
    
    # C. Average the embeddings to get a single vector
    paper_embedding = np.mean(chunk_embeddings, axis=0)
    
    # --- 4. Make a Prediction ---
    print("🔮 Making a prediction...")
    
    # The model expects a 2D array, so we reshape our single embedding
    prediction = classifier_model.predict([paper_embedding])
    prediction_proba = classifier_model.predict_proba([paper_embedding])

    # --- 5. Show the Result ---
    result_label = "Accepted" if prediction[0] == 1 else "Rejected"
    confidence_score = prediction_proba[0][prediction[0]] * 100
    
    print("\n" + "="*30)
    print("          PREDICTION RESULT")
    print("="*30)
    print(f"The model predicts that this paper would be:  **{result_label}**")
    print(f"Confidence: {confidence_score:.2f}%")
    print("="*30)

else:
    print("Could not extract enough text from the PDF to make a prediction.")