import os
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_astradb.vectorstores import AstraDBVectorStore
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI  
from langchain_core.runnables import RunnableSequence
import requests

load_dotenv()

class RAGPipeline:
    """
    RAGPipeline class for executing Retrieval-Augmented Generation (RAG) tasks.
    It integrates with the vector store for document retrieval and OpenAI's Azure LLM for text generation.
    """
    def __init__(self, azure_api_key, azure_endpoint, azure_deployment_name, vector_store=None, model_name="gpt-4"):
        """
        Initializes the RAGPipeline with Azure OpenAI API credentials, model name, and a vector store.
        
        Args:
            azure_api_key (str): The API key for Azure OpenAI service.
            azure_endpoint (str): The endpoint for Azure OpenAI service.
            azure_deployment_name (str): The deployment name for Azure OpenAI service.
            vector_store (object, optional): A vector store instance. If None, it will initialize using `initialize_vector_store()`.
            model_name (str, optional): The Azure OpenAI model name. Default is "gpt-4".
        """
        self.azure_api_key = azure_api_key
        self.azure_endpoint = azure_endpoint
        self.azure_deployment_name = azure_deployment_name
        self.vector_store = vector_store
        self.model_name = model_name

        if not all([self.azure_api_key, self.azure_endpoint, self.azure_deployment_name]):
            raise ValueError("Azure API key, endpoint, and deployment name are required.")

        # Initialize vector store if not provided
        if vector_store is None:
            self.vector_store = self.initialize_vector_store()
        else:
            self.vector_store = vector_store

    def initialize_vector_store(self):
        """
        Initializes the Astra DB vector store and Azure OpenAI embeddings.
        """
        embeddings = AzureOpenAIEmbeddings(
            model="text-embedding-ada-002",
            azure_endpoint= os.getenv("AZURE_EMBEDDING_ENDPOINT"),
            api_key= os.getenv("AZURE_OPENAI_API_KEY"),
            openai_api_version="2023-05-15",
        )

        vector_store = AstraDBVectorStore(
            collection_name="KnowledgePadi",
            token= os.getenv("ASTRA_DB_APPLICATION_TOKEN"),
            api_endpoint= os.getenv("ASTRA_DB_API_ENDPOINT"),
            embedding=embeddings
        )

        print(f"Astra DB vector store initialized for collection: KnowledgePadi.")
        return vector_store

    def initialize_llm(self):
        """
        Initializes the Azure OpenAI LLM for text generation with a custom prompt template.
        """
        prompt_template = """
        You are an AI assistant specifically designed to answer questions about UKNIAF, its work, and related activities, initiatives, and research documents. You are tasked with providing accurate, professional, and contextually relevant answers based strictly on the uploaded UKNIAF documents and user-provided instructions. Your primary goal is to ensure accuracy, reliability, and professionalism while minimizing any risk of inaccuracies or hallucinations.

        KEY INSTRUCTIONS:
        1.Knowledge Boundaries:
        Respond strictly based on the uploaded documents and user-provided instructions.
        Do not use external sources or make assumptions unless explicitly instructed.
        If the required information is unavailable, clearly state that the data is insufficient to provide an answer and suggest clarifications if needed.

        2.Purpose and Audience:
        Support stakeholders and project partners by addressing their questions with precision and clarity.
        Use a tone that is professional but approachable, maintaining a positive and collaborative demeanor.

        3.Handling Uncertainty:
        When unsure of an answer, clearly state that further verification is required.
        Always cite specific documents or sections from the provided dataset where relevant.

        4.Response Quality:
        Provide concise, accurate, and structured responses (e.g., bullet points or numbered lists) to improve clarity.
        Avoid speculation and confirm details when in doubt.
        Ensure all responses are specifically tailored to the needs of the audience.

        5.Tone and Style:
        Maintain a balance between professionalism and friendliness, avoiding overly technical jargon unless necessary.
        Address users respectfully and adapt to the context of their questions.

        6.Contextual Reference:
        Use basic information about UKNIAF’s program from its official website only when relevant: https://ukniaf.ng/.
        For detailed responses, rely solely on the uploaded documents and specific instructions.

        7.Program Overview:
        UKNIAF focuses on two main components: Infrastructure Finance (IF) and Power. Ensure responses align with the specific context of these components.
        Reference Task Orders (TOs) and milestones based on their codes and descriptions as provided in the 'Task Order Reference List.'

        8.Avoiding Hallucinations:
        Base all responses on verifiable data from the provided documents.
        Clearly state document names, sections, or relevant references when applicable.
        Avoid generating speculative or inaccurate information.

        EXAMPLE OF KNOWLEDGE BASE CONTEXT:
        1.UKNIAF Project Background:
        UKNIAF, delivered by TetraTech International Development (formerly WYG International), is a technical assistance facility funded by the UK’s Foreign, Commonwealth, and Development Office (FCDO). It aims to improve lives in Nigeria by enhancing infrastructure design and delivery. The program works across Infrastructure Finance (IF) and Power sectors, collaborating with federal and state governments, investors, and international partners.

        2.Task Order and Milestones:
        Task Orders (TOs) summarize specific projects or workstreams. They are coded as IF00XX for Infrastructure Finance and PW00XX for Power.
        Milestones (Mx or TMx) track the progress of these Task Orders.


        RESPONSE PRIORITIES:
        Accuracy: Ensure high precision in all responses.
        Completeness: Address all parts of user queries thoroughly.
        Professionalism: Use a tone that fosters trust and professionalism.
        By following these instructions, your role as Knowledge Padi is to deliver accurate, reliable, and context-aware responses that meet user expectations while adhering strictly to the provided knowledge base and scope.

        Relevant documents:
        {documents}

        Question: {input}

        Based on the above context, provide a detailed, clear, and concise response.
        """
        
        prompt = PromptTemplate(
            input_variables=["documents", "input"],
            template=prompt_template
        )
        
        # Initialize the Azure OpenAI LLM for chat completion
        llm = AzureChatOpenAI(
            azure_endpoint= os.getenv("AZURE_LLM_ENDPOINT"),
            azure_deployment="gpt-4",
            openai_api_key= os.getenv("AZURE_OPENAI_API_KEY"),
            openai_api_version="2024-08-01",
        )
        
        # Combine prompt with the LLM using the '|' operator (RunnableSequence)
        chain = RunnableSequence(first=prompt, last=llm)
        
        return chain

    def run(self, inputs):
        """
        Executes the RAG pipeline for a given query using a custom prompt.
        """
        query = inputs.get("query")

        if not query:
            raise ValueError("Query is required.")

        # Retrieve relevant documents from AstraDB
        retriever = self.vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
        retrieved_docs = retriever.invoke(query)

        if not retrieved_docs or len(retrieved_docs) == 0:
            raise ValueError("No relevant documents retrieved. Ensure the vector store is indexed.")

        # Format documents into plain text
        documents = "\n\n".join([f"Document {i+1}: {doc.page_content}" for i, doc in enumerate(retrieved_docs)])

        # Prepare the LLM prompt using the documents and query
        prompt = {
            "messages": [
                {"role": "user", "content": f"Based on the following documents, answer the question: {query}"},
                {"role": "system", "content": documents}  # Add the documents here to guide the LLM
            ]
        }

        # Prepare headers and payload for the API request
        headers = {
            "Content-Type": "application/json",
            "api-key": self.azure_api_key
        }

        # Make the request to the API
        response = requests.post(self.azure_endpoint, headers=headers, json=prompt)

        # Check the API response
        if response.status_code == 200:
            # If the response is successful, return the response content
            api_response = response.json()
            answer = api_response.get("choices", [{}])[0].get("message", {}).get("content", "No response content.")
            return answer
        else:
            # If the response fails, raise an error
            raise ValueError(f"API request failed with status code {response.status_code}: {response.text}")
