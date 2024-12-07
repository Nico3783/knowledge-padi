# Knowledge Padi AI

Knowledge Padi AI is a Retrieval-Augmented Generation (RAG) chatbot designed to answer questions based on UKNIAF-related documents. It integrates Azure OpenAI for embeddings and language models, with AstraDB as the vector store.

## Features
- **Azure OpenAI Integration**: Supports embeddings and language models.
- **AstraDB**: Manages vector store operations.
- **Flask API**: Provides a RESTful endpoint for querying the chatbot.
- **Pipeline**: Preprocesses PDF data, generates embeddings, and performs RAG.

## Project Structure

Project Directory and Files:

data/
    raw_pdfs/ -           Directory containing original PDF documents.
    processed_texts/

src/
   app.py -               Main Flask app for serving the RAG pipeline.
   preprocess.py -        Handles PDF parsing and preprocessing.
   vector_store.py -      Integration with AstraDB for vector storage.
   embeddings.py -        Embedding generation using Azure OpenAI.
   rag_pipeline.py -      Core logic for the RAG pipeline.
   automate_pipeline.py
   local_agent_test.py    Main logic to test agent locally on CLI terminal
   static/
         scripts.js
         styles.css

   templates/
            index.html
            widget.html
 

.env -                    Environment variables (e.g., API keys, config settings).
requirements.txt -        Python dependencies for the project.
README.md -               Project documentation.
setup.py -                Setup script for packaging and installation.
.gitignore



## Setup Instructions

### Prerequisites
- Python 3.10+
- Azure OpenAI API credentials
- AstraDB credentials

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Nico3783/knowledge-padi.git
   cd KnowledgePadi
