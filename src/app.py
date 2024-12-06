from flask import Flask, render_template, request, jsonify
from langchain_astradb import AstraDBVectorStore
from langchain_openai.embeddings.azure import AzureOpenAIEmbeddings
from rag_pipeline import RAGPipeline
from dotenv import load_dotenv
import os

load_dotenv()

# Flask app initialization
app = Flask(__name__, template_folder="templates", static_folder="static")

# Initialize RAG components (reuse from local-agent_test.py)
# Ensure embeddings and vector store initialization is outside route functions
embeddings = AzureOpenAIEmbeddings(
    model="text-embedding-ada-002",
    azure_endpoint=os.getenv("AZURE_EMBEDDING_ENDPOINT"),
    api_key=os.getenv("AZURE_EMBEDDING_API_KEY"),
    openai_api_version="2023-05-15",
)
vector_store = AstraDBVectorStore(
    collection_name="KnowledgePadi",
    embedding=embeddings,
    api_endpoint=os.getenv("ASTRA_DB_API_ENDPOINT"),
    token=os.getenv("ASTRA_DB_APPLICATION_TOKEN"),
    namespace="default_keyspace",
)
rag_pipeline = RAGPipeline(
    azure_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_LLM_ENDPOINT"),
    azure_deployment_name=os.getenv("AZURE_LLM_DEPLOYMENT"),
    vector_store=vector_store,
)

# Routes
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/index")
def full_page_chat():
    return render_template("index.html")

@app.route("/widget")
def widget_chat():
    return render_template("widget.html")

@app.route("/api/query", methods=["POST"])
def api_query():
    data = request.get_json()
    query = data.get("query", "")
    if not query:
        return jsonify({"error": "No query provided"}), 400

    # Retrieve relevant documents and generate response
    relevant_docs = vector_store.similarity_search(query, k=3)
    response = rag_pipeline.run({"query": query, "relevant_docs": relevant_docs})
    return jsonify({"response": response})

@app.route("/get_session_id", methods=["GET"])
def get_session_id():
    session_id = request.cookies.get("session_id", "guest")
    return jsonify({"session_id": session_id})

if __name__ == "__main__":
    app.run(debug=True)
