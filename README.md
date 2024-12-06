# Knowledge Padi AI

Knowledge Padi AI is a Retrieval-Augmented Generation (RAG) chatbot designed to answer questions based on UKNIAF-related documents. It integrates Azure OpenAI for embeddings and language models, with AstraDB as the vector store.

## Features
- **Azure OpenAI Integration**: Supports embeddings and language models.
- **AstraDB**: Manages vector store operations.
- **Flask API**: Provides a RESTful endpoint for querying the chatbot.
- **Pipeline**: Preprocesses PDF data, generates embeddings, and performs RAG.

## Project Structure

KnowledgePadi/ ├── data/ │ ├── raw_pdfs/ # Original PDFs ├── src/ │ ├── app.py # Main Flask App │ ├── preprocess.py # PDF parsing and preprocessing │ ├── vector_store.py # AstraDB vector store integration │ ├── embeddings.py # Embeddings generation with Azure OpenAI │ ├── rag_pipeline.py # RAG pipeline logic ├── tests/ │ ├── test_preprocess.py # Unit tests for preprocessing │ ├── test_vector_store.py # Unit tests for vector store │ ├── test_pipeline.py # Unit tests for RAG pipeline ├── .env # Environment variables ├── requirements.txt # Python dependencies ├── README.md # Documentation └── setup.py


## Setup Instructions

### Prerequisites
- Python 3.10+
- Azure OpenAI API credentials
- AstraDB credentials

### Installation
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd KnowledgePadi
