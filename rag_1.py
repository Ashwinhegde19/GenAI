from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pathlib import Path
from langchain_huggingface import HuggingFaceEmbeddings 
from langchain_qdrant import QdrantVectorStore

pdf_path = Path(__file__).parent / "nodejs.pdf"
loader = PyPDFLoader(pdf_path)
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)

split_docs = text_splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# vector_store = QdrantVectorStore.from_documents(
#     documents=[],
#     url = "http://localhost:6333",
#     collection_name="trying_langchain",
#     embedding=embeddings,
# )

# vector_store.add_documents(split_docs)

# print("Documents added to Qdrant collection.")

retriever = QdrantVectorStore.from_existing_collection(
    url = "http://localhost:6333",
    collection_name="trying_langchain",
    embedding=embeddings,
)

relevant_chunks = retriever.similarity_search(
    query="What is FS Module?"
    )

# Create formatted context with page content and page numbers
formatted_context = ""
for i, doc in enumerate(relevant_chunks):
    page_num = doc.metadata.get('page', 'Unknown')
    content = doc.page_content
    formatted_context += f"--- Document {i+1} (Page {page_num}) ---\n{content}\n\n"

SYSTEM_PROMPT = f"""
You are a helpful assistant. You will be provided with a question and a context. Your task is to answer the question based on the context provided.
If the context does not contain the answer, say "I don't know".

Context:
{formatted_context}
"""

print(formatted_context)
