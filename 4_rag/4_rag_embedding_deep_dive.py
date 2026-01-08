import os

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_mistralai.embeddings import MistralAIEmbeddings

# -----------------------------
# 0) Goal of this script (RAG perspective)
# -----------------------------
# We are building a tiny "retrieval system" over a book (The Odyssey).
# The pipeline is:
#   (1) Load raw text
#   (2) Split into chunks (so we can embed + retrieve efficiently)
#   (3) Embed chunks into vectors (two options: Mistral API vs local HuggingFace)
#   (4) Store vectors in a vector DB (Chroma) on disk
#   (5) At query time: embed the query, retrieve the most similar chunks
#
# NOTE: This script demonstrates Retrieval. It does NOT call an LLM to generate an answer.
# In full RAG, we'd do: retrieved_chunks -> prompt -> LLM -> final answer.


# Define the directory containing the text file and the persistent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "books", "odyssey.txt")
db_dir = os.path.join(current_dir, "db")

# Check if the text file exists
if not os.path.exists(file_path):
    raise FileNotFoundError(
        f"The file {file_path} does not exist. Please check the path."
    )

# -----------------------------
# 2) Document loading (Ingestion step in RAG)
# -----------------------------
# TextLoader reads the entire text file and wraps it in LangChain Document objects.
# Each Document has:
#   - page_content: the raw text
#   - metadata: optional info (source, page, url, etc.)

# Read the text content from the file
loader = TextLoader(file_path, encoding="utf-8")
documents = loader.load()


# -----------------------------
# 3) Chunking / splitting (Indexing design choice)
# -----------------------------
# Why split?
# - Embedding models work better on reasonably sized text (not huge books).
# - Retrieval needs small "knowledge atoms" so results are precise.
#
# chunk_size=1000 characters is a common baseline for demo purposes.
# chunk_overlap=0 means no repeated text across chunks.
# (In production, overlap like 100-200 chars often helps prevent boundary issues.)

# Split the document into chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

# Display information about the split documents
print("\n--- Document Chunks Information ---")
print(f"Number of document chunks: {len(docs)}")
print(f"Sample chunk:\n{docs[0].page_content}\n")


# Function to create and persist vector store
def create_vector_store(docs, embeddings, store_name):
    """
    Create a Chroma vector store for a given set of chunks and a specific embedding model.

    RAG concept:
      - We embed each chunk -> vector
      - Store (vector, chunk_text, metadata) inside Chroma
      - Persist to disk so we don't recompute embeddings every run

    store_name matters:
      - Each embedding model should have its OWN vector DB
      - Because vectors from different embedding models live in different "vector spaces"
    """

    persistent_directory = os.path.join(db_dir, store_name)
    if not os.path.exists(persistent_directory):
        print(f"\n--- Creating vector store {store_name} ---")
        Chroma.from_documents(
            docs, embeddings, persist_directory=persistent_directory)
        print(f"--- Finished creating vector store {store_name} ---")
    else:
        print(
            f"Vector store {store_name} already exists. No need to initialize.")

# -----------------------------
# 5) Embeddings: compare two approaches
# -----------------------------

# 5.1 Mistral embeddings (cloud API)
# Pros:
# - Very strong general embeddings
# - Simple setup
# Cons:
# - Needs API key
# - Has cost
# - Network dependency


# 1. Mistral Embeddings
# Uses Mistral's embedding models.
# Useful for general-purpose embeddings with high accuracy.
print("\n--- Using Mistral Embeddings ---")
mistral_embeddings = MistralAIEmbeddings(model="mistral-embed")

create_vector_store(docs, mistral_embeddings, "chroma_db_openai")

# 5.2 HuggingFace embeddings (local)
# Pros:
# - Runs locally, no API cost
# - Good for privacy/offline
# Cons:
# - Needs compute (CPU/GPU)
# - Quality depends heavily on chosen model

# 2. Hugging Face Transformers
# Uses models from the Hugging Face library.
# Ideal for leveraging a wide variety of models for different tasks.
# Note: Running Hugging Face models locally on your machine incurs no direct cost other than using your computational resources.
# Note: Find other models at https://huggingface.co/models?other=embeddings
print("\n--- Using Hugging Face Transformers ---")
huggingface_embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)
create_vector_store(docs, huggingface_embeddings, "chroma_db_huggingface")

print("Embedding demonstrations for OpenAI and Hugging Face completed.")


# Function to query a vector store
def query_vector_store(store_name, query, embedding_function):
    """
    Query-time RAG retrieval:

      query_text -> embed(query_text) -> vector_q
      vector_q -> similarity search in Chroma -> top-k chunks

    Important:
      - Must use the same embedding_function that was used to build that store.
      - Otherwise the query vector and document vectors are incompatible.
    """

    persistent_directory = os.path.join(db_dir, store_name)
    if os.path.exists(persistent_directory):
        print(f"\n--- Querying the Vector Store {store_name} ---")
        # Re-open the persisted DB and tell Chroma how to embed the query
        db = Chroma(
            persist_directory=persistent_directory,
            embedding_function=embedding_function,
        )
        # Retriever configuration:
        # - similarity_score_threshold: filter out low-similarity results
        # - k=3: return up to 3 chunks
        # - score_threshold=0.1: very permissive; good for demos, not strict filtering

        retriever = db.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": 3, "score_threshold": 0.1},
        )

        # This triggers:
        #   - embed(query) using embedding_function
        #   - similarity search in DB

        relevant_docs = retriever.invoke(query)
        # Display the relevant results with metadata
        print(f"\n--- Relevant Documents for {store_name} ---")
        for i, doc in enumerate(relevant_docs, 1):
            print(f"Document {i}:\n{doc.page_content}\n")
            if doc.metadata:
                print(f"Source: {doc.metadata.get('source', 'Unknown')}\n")
    else:
        print(f"Vector store {store_name} does not exist.")


# Define the user's question
query = "Who is Odysseus' wife?"

# Query each vector store
query_vector_store("chroma_db_mistral", query, mistral_embeddings)
query_vector_store("chroma_db_huggingface", query, huggingface_embeddings)

print("Querying demonstrations completed.")
