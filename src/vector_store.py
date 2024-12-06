from astrapy import DataAPIClient
from langchain_astradb import AstraDBVectorStore
from langchain_openai.embeddings.azure import AzureOpenAIEmbeddings
from dotenv import load_dotenv
import os 

# Load environment variables
load_dotenv()

Astra_token= os.getenv("ASTRA_DB_APPLICATION_TOKEN"),
Astra_endpoint= os.getenv("ASTRA_DB_API_ENDPOINT"),
Astra_collection= os.getenv("ASTRA_COLLECTION_NAME"),

# Initialize the client with your application token
client = DataAPIClient(Astra_token)
database = client.get_database(Astra_endpoint)
# Access the collection by name
collection = database.get_collection(Astra_collection)


def initialize_vector_store():
    """
    Initializes the Astra DB vector store and Azure OpenAI embeddings.
    """
    azure_key = os.getenv("AZURE_OPENAI_API_KEY")
    endpoint = os.getenv("AZURE_EMBEDDING_ENDPOINT")

    if not azure_key or not endpoint:
        raise ValueError("Azure OpenAI credentials must be provided.")

    # Initialize Azure OpenAI embeddings
    embeddings = AzureOpenAIEmbeddings(
        model="text-embedding-ada-002",
        azure_endpoint=endpoint,
        api_key=azure_key,
        openai_api_version="2023-05-15",
    )

    # Initialize the Astra DB vector store
    vector_store = AstraDBVectorStore(
        collection_name="KnowledgePadi",
        token= os.getenv("ASTRA_DB_TOKEN"),
        api_endpoint= os.getenv("ASTRA_DB_ENDPOINT"),
        embedding=embeddings
    )

    print(f"Astra DB vector store initialized for collection: KnowledgePadi.")
    return vector_store, embeddings


def save_embeddings_to_astra(collection, embeddings, documents, document_hashes):
    """
    Saves the embeddings and documents to the Astra DB to avoid duplicates.
    """
    collection = database.get_collection("KnowledgePadi")

    for doc, emb, doc_hash in zip(documents, embeddings, document_hashes):
        try:
            # Check if the document hash already exists
            existing = collection.find_one({"hash": doc_hash})
            if existing:
                print(f"Document with hash {doc_hash} already exists in Astra DB. Skipping.")
                continue

            # Prepare the document data for insertion
            doc_data = {
                "id": f"doc_{hash(doc)}",
                "hash": doc_hash,
                "content": doc,
                "embedding": emb
            }

            # Insert into the collection
            collection.insert(doc_data)
            print(f"Inserted document with hash {doc_hash} into Astra DB.")
        except Exception as e:
            print(f"Error saving document with hash {doc_hash}: {e}")

    print("All embeddings have been processed and saved.")

class CustomDocument:
    """
    A class to represent a document with required attributes for LangChain.
    """
    def __init__(self, id, page_content, embedding, metadata=None):
        self.id = id
        self.page_content = page_content
        self.embedding = embedding
        self.metadata = metadata if metadata is not None else {}

class AstraVectorStore:
    """
    Class to handle Astra vector store interactions using Astrapy.
    """

    def __init__(self):
        self.vector_store, self.embeddings = initialize_vector_store()
        self.db = self.initialize_db_client()

    def initialize_db_client(self):
        """
        Initializes the Astrapy DataAPIClient for Astra DB.
        """
        astra_db_token = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
        astra_db_endpoint = os.getenv("ASTRA_DB_API_ENDPOINT")

        if not astra_db_token or not astra_db_endpoint:
            raise ValueError("Astra DB token and endpoint must be provided.")

        # Initialize the Astrapy client
        client = DataAPIClient(astra_db_token)
        db = client.get_database_by_api_endpoint(astra_db_endpoint)
        print(f"Connected to Astra DB. Collections available: {db.list_collection_names()}")
        return db

    def index_documents(self, documents, metadatas=None):
        """
        Add documents and their embeddings to the vector store.
        """
        chunk_embeddings = self.embeddings.embed_documents(documents)

        if metadatas is None:
            metadatas = [{} for _ in documents]
        elif len(metadatas) != len(documents):
            raise ValueError("Length of metadatas must match the length of documents.")

        formatted_documents = [
            CustomDocument(
                id=f"doc_{i}", 
                page_content=doc, 
                embedding=emb, 
                metadata=meta
            )
            for i, (doc, emb, meta) in enumerate(zip(documents, chunk_embeddings, metadatas))
        ]

        for doc in formatted_documents[:3]:
            print(f"Document ID: {doc.id}, Metadata: {doc.metadata}, Content Length: {len(doc.page_content)}")

        self.vector_store.add_documents(formatted_documents)
        print(f"Successfully indexed {len(documents)} documents into the vector store.")

        for doc in formatted_documents:
            result = self.query(doc.id)
            if result:
                print(f"Document with ID {doc.id} is present in Astra DB.")
            else:
                print(f"Document with ID {doc.id} is NOT found in Astra DB.")

    def query(self, document_id):
        """
        Query the Astra DB to check if a document with the given ID exists.
        """
        collection = self.db.get_collection("KnowledgePadi")
        try:
            results = collection.find({"id": document_id})
            return results[0] if results else None
        except Exception as e:
            print(f"Error querying the database: {e}")
            return None
