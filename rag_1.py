from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pathlib import Path
from langchain_huggingface import HuggingFaceEmbeddings 
from langchain_google_genai import GoogleGenerativeAIEmbeddings  
from langchain_qdrant import QdrantVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
import os

api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise ValueError("Please set the GEMINI_API_KEY environment variable")

pdf_path = Path(__file__).parent / "nodejs.pdf"

loader = PyPDFLoader(pdf_path)
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)

split_docs = text_splitter.split_documents(documents)

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)

# embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

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
    collection_name="trying_langchain_gemini",
    embedding=embeddings,
)

# Set up Google Gemini API
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise ValueError("Please set the GEMINI_API_KEY environment variable")
    
# Fixed: Explicitly pass the API key to the ChatGoogleGenerativeAI constructor
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.2, google_api_key=api_key)

def get_response(query):
    # Retrieve relevant documents
    relevant_chunks = retriever.similarity_search(query)
    
    # Format the context
    formatted_context = ""
    for i, doc in enumerate(relevant_chunks):
        page_num = doc.metadata.get('page', 'Unknown')
        content = doc.page_content
        formatted_context += f"--- Document {i+1} (Page {page_num}) ---\n{content}\n\n"
    
    # Create the system prompt
    system_prompt = f"""
    You are a helpful assistant for answering questions about Node.js based on the documentation provided.
    You will be provided with a question and context from the Node.js documentation. 
    Your task is to answer the question based on the context provided.
    If the context does not contain the answer, say "I don't know based on the available documentation."
    Always cite the page numbers you're referencing in your answer.

    Context:
    {formatted_context}
    """
    
    # Get response from LLM
    response = llm.invoke([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query}
    ])
    
    return response.content, formatted_context

def main():
    print("🤖 Node.js Documentation Assistant (Powered by Gemini)")
    print("Type 'exit' to end the conversation\n")
    
    while True:
        user_query = input("\n👤 What would you like to know about Node.js? ")
        
        if user_query.lower() in ['exit', 'quit', 'bye']:
            print("\n🤖 Goodbye! Have a great day!")
            break
        
        print("\n🔍 Searching documentation...")
        response, context = get_response(user_query)
        
        print("\n🤖 AI:")
        print(response)
        
        # Uncomment to show the retrieved context for debugging
        # print("\nRetrieved context:")
        # print(context)

if __name__ == "__main__":
    main()
