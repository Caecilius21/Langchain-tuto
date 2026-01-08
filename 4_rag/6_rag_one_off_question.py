# This script is a minimal end-to-end RAG pipeline

import os

from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings

# Load environment variables from .env
load_dotenv()

# -----------------------------
# 2) Locate the persisted vector store
# -----------------------------
# indexing was already done earlier
# (documents were chunked, embedded, and stored in Chroma)

# Define the persistent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
persistent_directory = os.path.join(
    current_dir, "db", "chroma_db_with_metadata")

# -----------------------------
# 3) Define the embedding model
# -----------------------------
# This MUST be the same embedding model that was used during indexing.
# Otherwise, query vectors and document vectors would be incompatible.

embeddings = MistralAIEmbeddings(model="mistral-embed")

# -----------------------------
# 4) Re-open the existing vector store
# -----------------------------
# No documents are embedded here.
# We are just reconnecting to an already-built semantic index.

db = Chroma(persist_directory=persistent_directory,
            embedding_function=embeddings)

# -----------------------------
# 5) User query (intent)
# -----------------------------

query = "How can I learn more about LangChain?"

# -----------------------------
# 6) Retrieval step (the "R" in RAG)
# -----------------------------
# Convert the vector store into a retriever interface
# search_type="similarity" = standard nearest-neighbor search
# k=1 = return the single most similar chunk

retriever = db.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 1},
)

# This does:
#   - embed(query)
#   - vector similarity search in Chroma
#   - return the most relevant document chunk(s)
relevant_docs = retriever.invoke(query)

# -----------------------------
# 7) Inspect retrieved context
# -----------------------------
# Display the relevant results with metadata
print("\n--- Relevant Documents ---")
for i, doc in enumerate(relevant_docs, 1):
    print(f"Document {i}:\n{doc.page_content}\n")

# -----------------------------
# 8) Manual prompt construction (augmentation)
# -----------------------------
# We explicitly combine:
#   - the user's question
#   - the retrieved documents
# And instruct the model to ONLY use that information.

# Combine the query and the relevant document contents
combined_input = (
    "Here are some documents that might help answer the question: "
    + query
    + "\n\nRelevant Documents:\n"
    + "\n\n".join([doc.page_content for doc in relevant_docs])
    + "\n\nPlease provide an answer based only on the provided documents. If the answer is not found in the documents, respond with 'I'm not sure'."
)

# -----------------------------
# 9) Create the chat model (the "G" in RAG)
# -----------------------------
# This is the generation model that will produce the final answer
# Create a Mistral model

model = ChatMistralAI(model="mistral-large-latest")

# -----------------------------
# 10) Build chat messages
# -----------------------------
# SystemMessage sets behavior
# HumanMessage contains the augmented prompt

# Define the messages for the model
messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content=combined_input),
]

# -----------------------------
# 11) Generate the answer
# -----------------------------
# Invoke the model with the combined input
result = model.invoke(messages)

# Display the full result and content only
print("\n--- Generated Response ---")
# print("Full result:")
# print(result)
print("Content only:")
print(result.content)
