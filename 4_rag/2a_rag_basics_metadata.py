# This script performs the indexing phase of a RAG system, transforming raw book files into
# chunked, embedded, source-aware vectors stored in Chroma for future semantic retrieval.
# Build a knowledge index so that future queries can be retrieved semantically.
# There is:
# ❌ No user query
# ❌ No retrieval
# ❌ No generation

import os

from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_mistralai.embeddings import MistralAIEmbeddings

# 1) Define the corpus and the index location
# Define the directory containing the text files and the persistent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
books_dir = os.path.join(current_dir, "books") # your knowledge corpus
db_dir = os.path.join(current_dir, "db")
persistent_directory = os.path.join(db_dir, "chroma_db_with_metadata") # your semantic index

# Corpus = human-readable
# Vector DB = machine-readable

print(f"Books directory: {books_dir}")
print(f"Persistent directory: {persistent_directory}")

# Check if the Chroma vector store already exists
if not os.path.exists(persistent_directory):
    print("Persistent directory does not exist. Initializing vector store...")

    # Ensure the books directory exists
    if not os.path.exists(books_dir):
        raise FileNotFoundError(
            f"The directory {books_dir} does not exist. Please check the path."
        )

    # List all text files in the directory
    book_files = [f for f in os.listdir(books_dir) if f.endswith(".txt")]

    # Read the text content from each file and store it with metadata
    documents = []
    for book_file in book_files:
        file_path = os.path.join(books_dir, book_file)
        # 3) Load raw documents + attach provenance metadata
        loader = TextLoader(file_path, encoding="utf-8")
        book_docs = loader.load()
        for doc in book_docs:
            # Add metadata to each document indicating its source
            doc.metadata = {"source": book_file}
            documents.append(doc)
            # Why metadata matters in RAG:
                # Source attribution
                # Debugging hallucinations
                # Citation in answers
                # Filtering (per book, per author, etc.)
            # Without metadata, RAG becomes a black box.

    # 4) Chunking = adapting human text to LLM constraints
    # LLMs cannot embed or reason over book-length text
    # Split the documents into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)

    # Display information about the split documents
    print("\n--- Document Chunks Information ---")
    print(f"Number of document chunks: {len(docs)}")

    # 5) Embedding = turning knowledge into geometry
    # Create embeddings
    print("\n--- Creating embeddings ---")
    embeddings = MistralAIEmbeddings(model="mistral-embed")
    print("\n--- Finished creating embeddings ---")

    # 6) Vector store creation = building the retrieval engine
    # 7) Persistence = making RAG usable across runs
    # Create the vector store and persist it
    print("\n--- Creating and persisting vector store ---")
    db = Chroma.from_documents(
        docs, embeddings, persist_directory=persistent_directory)
    print("\n--- Finished creating and persisting vector store ---")
    # Semantic meaning → position in vector space
    # Similar meaning → close vectors

else:
    print("Vector store already exists. No need to initialize.")
