from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
import logging
import os
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate

logging.basicConfig()
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)

if "GEMINI_API_KEY" in os.environ:
    os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]

# 1. Load and Prepare Data
pdf_path = Path(__file__).parent / "nodejs.pdf"
loader = PyPDFLoader(pdf_path)
data = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
splits = text_splitter.split_documents(data)

# 2. Embed and Store in Qdrant Vector Database
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
db = QdrantVectorStore.from_documents(
    documents=splits,
    embedding=embeddings,
    url="http://localhost:6333",
    collection_name="langchain_gemini_demo_1",
)

# 3. Define the Gemini LLM
llm = GoogleGenerativeAI(model="models/gemini-2.0-flash", temperature=0.0)

# 4. Expand the user query (no LLMChain, use RunnableSequence)
prompt = PromptTemplate.from_template(
    "Generate 3 diverse rephrasings for this user question:\n{question}"
)
user_query = "What is FS Module?"

# Use RunnableSequence: prompt | llm
chain = prompt | llm
response = chain.invoke({"question": user_query})
expanded_queries = [q for q in response.split("\n") if q.strip()]

# 5. Retrieve top-k docs for each expanded query
retriever = db.as_retriever(search_kwargs={"k": 5})
rankings = []
doc_map = {}

for q in expanded_queries:
    docs = retriever.invoke(q)
    ranking = []
    for doc in docs:
        # Use a unique identifier for each doc (preferably from metadata)
        doc_id = doc.metadata.get("id", str(hash(doc.page_content)))
        ranking.append(doc_id)
        doc_map[doc_id] = doc
    rankings.append(ranking)

# 6. Reciprocal Rank Fusion
def reciprocal_rank_fusion(rankings, k=60):
    scores = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            score = 1 / (k + rank + 1)
            scores[doc_id] = scores.get(doc_id, 0) + score
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)

fused = reciprocal_rank_fusion(rankings)
top_ids = [doc_id for doc_id, _ in fused[:5]]

# 7. Fetch top docs by ID (from doc_map)
top_docs = [doc_map[doc_id] for doc_id in top_ids if doc_id in doc_map]

# 8. Final Answer
context = "\n".join([doc.page_content for doc in top_docs])
final_prompt = f"""Here is some context:
{context}

Answer the following question using the context provided:
{user_query}"""

final_answer = llm.invoke(final_prompt)
print(final_answer)