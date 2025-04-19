from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.retrievers import MultiQueryRetriever
from langchain_qdrant import QdrantVectorStore
import logging
import os
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader


logging.basicConfig()
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)


# Ensure GOOGLE_API_KEY is set if GEMINI_API_KEY is available
if "GOOGLE_API_KEY" not in os.environ and "GEMINI_API_KEY" in os.environ:
    os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]

# 1. Load and Prepare Data
pdf_path = Path(__file__).parent / "nodejs.pdf"
loader = PyPDFLoader(pdf_path)
data = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
splits = text_splitter.split_documents(data)

# 2. Embed and Store in Qdrant Vector Database
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
qdrant_url = "http://localhost:6333"
collection_name = "langchain_gemini_demo"
db = QdrantVectorStore.from_documents(
    documents=splits,
    embedding=embeddings,
    url=qdrant_url,
    collection_name=collection_name,
)

# 3. Define the Gemini LLM
llm = GoogleGenerativeAI(model="models/gemini-2.0-flash", temperature=0.0)

# 4. Use MultiQueryRetriever
retriever = MultiQueryRetriever.from_llm(
    retriever=db.as_retriever(), llm=llm
)

# 5. Get relevant documents for the user query
user_query = "What is FS Module?"  
unique_docs = retriever.invoke(user_query) 

# 6. Pass Unique Chunks and Original Question to Gemini for Final Answer
context = "\n".join([doc.page_content for doc in unique_docs])
final_prompt = f"""Here is some context:
{context}

Answer the following question using the context provided:
{user_query}"""

final_answer = llm.invoke(final_prompt)
print(final_answer)