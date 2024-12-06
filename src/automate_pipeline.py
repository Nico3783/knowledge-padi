import sys
import os
import hashlib
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from preprocess import preprocess_texts
from embeddings import generate_embeddings
from vector_store import initialize_vector_store, save_embeddings_to_astra
from rag_pipeline import RAGPipeline
from dotenv import load_dotenv

load_dotenv()

# Constants
PDF_DIRECTORY = "data/raw_pdfs"
TEXT_DIRECTORY = "data/processed_texts"


def get_document_hash(document):
    """
    Generates a unique hash for the document text to check for duplicates.
    """
    return hashlib.sha256(document.encode('utf-8')).hexdigest()

# 1. Preprocess PDFs into text documents
print("Loading and preprocessing PDFs...")
documents = preprocess_texts(PDF_DIRECTORY)

# 2. Save processed documents to TEXT_DIRECTORY (optional step)
os.makedirs(TEXT_DIRECTORY, exist_ok=True)
for i, doc in enumerate(documents):
    file_path = os.path.join(TEXT_DIRECTORY, f"document_{i}.txt")
    with open(file_path, 'w', encoding='utf-8') as f:  # Specify UTF-8 encoding
        f.write(doc)
print(f"Processed texts saved to {TEXT_DIRECTORY}")

# 3. Generate Embeddings for the documents
print("Generating embeddings...")
embeddings = generate_embeddings(documents)

# 4. Initialize Astra DB vector store
print("Initializing Astra DB vector store...")
session = initialize_vector_store()

# 5. Save embeddings to Astra DB to avoid duplicates
print("Saving embeddings to Astra DB...")
doc_hashes = [get_document_hash(doc) for doc in documents]  # Generate hashes for documents
save_embeddings_to_astra(session, embeddings, documents, doc_hashes)

# 6. Initialize the RAG pipeline
print("Initializing RAG pipeline...")
rag_pipeline = RAGPipeline(
    azure_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_LLM_ENDPOINT"),
    azure_deployment_name=os.getenv("AZURE_LLM_DEPLOYMENT"),
    vector_store=session  # Pass the Astra DB session as the vector store
)

# 7. Query the RAG pipeline (example)
query = "What is the purpose of the RAG pipeline?"
print(f"Query: {query}")
response = rag_pipeline.run(query)

print(f"Response: {response}")
