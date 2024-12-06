from langchain_openai.embeddings.azure import AzureOpenAIEmbeddings
import os
from dotenv import load_dotenv

load_dotenv()

def generate_embeddings(texts):
    """
    Generate embeddings for a list of texts using Azure OpenAI.
    """
    try:
        azure_key = os.getenv("AZURE_OPENAI_API_KEY"),
        endpoint =  os.getenv("AZURE_OPENAI_ENDPOINT")
        if not azure_key or not endpoint:
            raise ValueError("Azure OpenAI credentials are missing. Check your .env file.")
        
        embeddings = AzureOpenAIEmbeddings(
            model="text-embedding-ada-002",
            azure_endpoint=endpoint,
            openai_api_key=azure_key,
            openai_api_version="2023-05-15",
        )
        return embeddings.embed_documents(texts)
    except ValueError as ve:
        print(f"ValueError: {ve}")
        raise
    except Exception as e:
        print(f"An error occurred: {e}")
        raise
