import openai
from preprocess import preprocess_texts
from langchain_astradb import AstraDBVectorStore
from langchain_openai.embeddings.azure import AzureOpenAIEmbeddings
from rag_pipeline import RAGPipeline
from dotenv import load_dotenv
import os

load_dotenv()

# Load environment variables
ASTRA_DB_API_ENDPOINT = os.getenv("ASTRA_DB_API_ENDPOINT")
ASTRA_DB_APPLICATION_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
AZURE_EMBEDDING_API_KEY = os.getenv("AZURE_EMBEDDING_API_KEY")
AZURE_EMBEDDING_ENDPOINT = os.getenv("AZURE_EMBEDDING_ENDPOINT")
AZURE_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_EMBEDDING_DEPLOYMENT")

# Set OpenAI/Azure environment variables
openai.api_type = "azure"
openai.api_base = AZURE_EMBEDDING_ENDPOINT
openai.api_key = AZURE_EMBEDDING_API_KEY
openai.api_version = "2023-05-15"  
def main():
    print("Loading and preprocessing PDFs...")
    documents = preprocess_texts("data/raw_pdfs")

    print("Initializing Astra vector store...")
    embeddings = AzureOpenAIEmbeddings(
        model="text-embedding-ada-002",  
        azure_endpoint=AZURE_EMBEDDING_ENDPOINT,
        api_key=AZURE_EMBEDDING_API_KEY,
        openai_api_version="2023-05-15",
    )

    vector_store = AstraDBVectorStore(
        collection_name="KnowledgePadi",
        embedding=embeddings,
        api_endpoint=ASTRA_DB_API_ENDPOINT,
        token=ASTRA_DB_APPLICATION_TOKEN,
        namespace="default_keyspace",
    )

    print("Indexing documents...")
    vector_store.add_documents(documents)

    print("Testing RAG pipeline...")
    rag_pipeline = RAGPipeline(
        azure_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        azure_endpoint=os.getenv("AZURE_LLM_ENDPOINT"),
        azure_deployment_name=os.getenv("AZURE_LLM_DEPLOYMENT"),
        vector_store=vector_store
    )

    while True:
        query = input("Enter your query (type 'exit' to quit): ")
        if query.lower() == "exit":
            break

        if not query:
            print("Error: Query cannot be empty. Please enter a valid query.")
            continue
        
        # Perform similarity search to retrieve relevant documents
        relevant_docs = vector_store.similarity_search(query, k=3)

        # Pass the relevant documents along with the query to the RAG pipeline
        response = rag_pipeline.run({"query": query, "relevant_docs": relevant_docs})
        print(f"Agent Response: {response}")

if __name__ == "__main__":
    main()
