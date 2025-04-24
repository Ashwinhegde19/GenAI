import os
import google.generativeai as genai
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient


api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)

llm = genai.GenerativeModel('gemini-2.0-flash')

question = "What is FS module?"

# Step 1: Generate a hypothetical answer using Gemini
hypo_prompt = f"Answer this question as if you were an expert: {question}"
hypo_response = llm.generate_content(hypo_prompt)
hypothetical_answer = hypo_response.text.strip()

# Step 2: Get embeddings for the hypothetical answer using Hugging Face
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
hypo_embedding = embeddings.embed_query(hypothetical_answer)

# Step 3: Connect to your existing Qdrant collection 
client = QdrantClient(host="localhost", port=6333)
db = QdrantVectorStore(
    client=client,
    collection_name="hyde_demo",
    embedding=embeddings,
)

# Step 4: Search your vector database using the hypothetical answer's embedding
results = db.similarity_search_by_vector(hypo_embedding, k=3)

# Step 5: Show the top results with page numbers
for i, doc in enumerate(results, 1):
    page = doc.metadata.get("page", "N/A")
    print(f"\nResult {i} (Page {page}):\n{doc.page_content}")