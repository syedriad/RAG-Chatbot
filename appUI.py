import os
import shutil
from dotenv import load_dotenv

load_dotenv()

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.agents import create_agent
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
import chromadb
from chromadb.errors import NotFoundError
import streamlit as st

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="DocuTalk - RAG PDF Chatbot",
    page_icon="📚",
    layout="centered"
)

# Header Section
st.title("📚 DocuTalk - RAG PDF Chatbot")
st.caption("Upload your PDFs and chat strictly with the document's contents (Powered by Gemini).")
st.divider()

### Data in st session
if "document_uploaded" not in st.session_state:
    st.session_state.document_uploaded = False

if "agent" not in st.session_state:
    st.session_state.agent = None

if "messages" not in st.session_state:
    st.session_state.messages = []


def process_document(path):
    # Initialize persistent Chroma client
    client = chromadb.PersistentClient(path="chroma_DB")
    
    # Safely clear old collection if it exists
    try:
        client.delete_collection("pdf_rag_collection")
    except (ValueError, NotFoundError, Exception):
        pass

    ## 1. Load the documents
    loader = PyPDFDirectoryLoader(path)
    docs = loader.load()

    if not docs:
        st.error("No valid documents found in the directory!")
        return

    ## 2. Split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents=docs)

    if not chunks:
        st.error("Could not extract text from the provided PDFs.")
        return

    ## 3. Embeddings and Vector DB (Switched to Google Embeddings)
    embedding_model = GoogleGenerativeAIEmbeddings(model="text-embedding-004")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        client=client,
        collection_name="pdf_rag_collection"
    )

    ## 4. Retriever (MMR search)
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 4,
            "fetch_k": 10,
            "lambda_mult": 0.5
        }
    )

    ## 5. Agent Setup with Gemini LLM
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

    @tool
    def retrieve_context(query: str) -> str:
        """Retrieve relevant information from the document knowledge base using MMR search.
        
        ALWAYS use this tool when answering questions about the uploaded PDF document.
        """
        retrieved_docs = retriever.invoke(query)

        if not retrieved_docs:
            return "NO_CONTEXT_FOUND"

        contents = [doc.page_content for doc in retrieved_docs]
        return "\n\n".join(contents)

    system_prompt = """You are a strict QA assistant. Your ONLY purpose is to answer questions based on the uploaded PDF document.

CRITICAL INSTRUCTIONS:
1. You MUST ALWAYS call `retrieve_context` for EVERY question asked.
2. BASE YOUR ANSWER ENTIRELY ON THE RETRIEVED CONTEXT.
3. NEVER use your own pre-trained knowledge or general knowledge to answer questions.
4. If `retrieve_context` returns "NO_CONTEXT_FOUND" or if the retrieved text does not contain relevant information, reply EXACTLY with:
   "I couldn't find any information about that in the uploaded document."
"""

    tools = [retrieve_context]
    memory = InMemorySaver()

    agent = create_agent(
        model=llm,
        tools=tools,
52    system_prompt=system_prompt,
        checkpointer=memory
    )

    st.session_state.agent = agent
    st.session_state.document_uploaded = True


### Upload UI
if not st.session_state.document_uploaded:
    st.subheader("Upload Documents")
    uploaded = st.file_uploader(label="Select PDF Files", type=["pdf"], accept_multiple_files=True)
    if uploaded:
        with st.spinner("Processing documents..."):
            target_dir = "doc_files"
            os.makedirs(target_dir, exist_ok=True)

            for file in uploaded:
                file_path = os.path.join(target_dir, file.name)
                with open(file_path, "wb") as f:
                    f.write(file.getvalue())

            process_document(target_dir)

            if st.session_state.document_uploaded:
                st.rerun()


### Chat UI
if st.session_state.document_uploaded and st.session_state.agent:
    st.success("Documents processed successfully! Ask your questions below.")
    
    for message in st.session_state.messages:
        role = message.get("role")
        content = message.get("content")
        st.chat_message(role).markdown(content)

    query = st.chat_input("Ask anything related to uploaded documents...")
    if query:
        st.session_state.messages.append({"role": "user", "content": query})
        st.chat_message("user").markdown(query)

        response = st.session_state.agent.invoke(
            {"messages": [{"role": "user", "content": query}]},
            {"configurable": {"thread_id": "1"}}
        )

        answer = response["messages"][-1].content
        st.chat_message("ai").markdown(answer)
        st.session_state.messages.append({"role": "ai", "content": answer})
