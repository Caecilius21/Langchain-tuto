# This script runs only the “R” part of RAG:
# Retrieve(query) → return documents


import os

from langchain_community.vectorstores import Chroma
from langchain_mistralai.embeddings import MistralAIEmbeddings

# Define the persistent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
persistent_directory = os.path.join(current_dir, "db", "chroma_db")

# Define the embedding model
# You must embed the query using the same embedding model
embeddings = MistralAIEmbeddings(model="mistral-embed")

# 1) Vector store re-attachment (not creation)
# Load the existing vector store with the embedding function
db = Chroma(persist_directory=persistent_directory,
            embedding_function=embeddings) # Even though you are not embedding documents, you will embed the query.

# 2) Query = user intent
# Define the user's question
query = "Who is Odysseus' wife?"

# 3) Retriever configuration = retrieval policy
# Retrieve relevant documents based on the query
retriever = db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 3, "score_threshold": 0.9}, # play with threshold
)
# k = 3: At most 3 chunks - k closest doc
# Risky if embeddings are noisy

# 4) The critical step: query embedding
relevant_docs = retriever.invoke(query)

# query
#  ↓
# embed(query)  ← Mistral API call
#  ↓
# vector_q
#  ↓
# similarity search in Chroma
#  ↓
# top-k chunks

# 5) Output = raw retrieved context
# Display the relevant results with metadata
print("\n--- Relevant Documents ---")
for i, doc in enumerate(relevant_docs, 1):
    print(f"Document {i}:\n{doc.page_content}\n")
    if doc.metadata:
        print(f"Source: {doc.metadata.get('source', 'Unknown')}\n")


# In a full RAG system, this would feed into:
# Prompt = system + question + retrieved_docs
# LLM(prompt)