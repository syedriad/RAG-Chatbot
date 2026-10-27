# import os
# import shutil
# from dotenv import load_dotenv

# load_dotenv()

# from langchain_community.document_loaders import PyPDFDirectoryLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
# from langchain_community.vectorstores import Chroma
# from langchain.agents import create_agent
# from langchain.tools import tool
# from langgraph.checkpoint.memory import InMemorySaver
# import chromadb
# from chromadb.errors import NotFoundError
# import streamlit as st

# # Set Streamlit Page Configuration
# st.set_page_config(
#     page_title="DocuTalk - RAG PDF Chatbot",
#     page_icon="📚",
#     layout="centered"
# )

# # Header Section
# st.title("📚 DocuTalk - RAG PDF Chatbot")
# st.caption("Upload your PDFs and chat strictly with the document's contents (Powered by Gemini).")
# st.divider()

# ### Data in st session
# if "document_uploaded" not in st.session_state:
#     st.session_state.document_uploaded = False

# if "agent" not in st.session_state:
#     st.session_state.agent = None

# if "messages" not in st.session_state:
#     st.session_state.messages = []


# def process_document(path):
#     # Initialize persistent Chroma client
#     client = chromadb.PersistentClient(path="chroma_DB")
    
#     # Safely clear old collection if it exists
#     try:
#         client.delete_collection("pdf_rag_collection")
#     except (ValueError, NotFoundError, Exception):
#         pass

#     ## 1. Load the documents
#     loader = PyPDFDirectoryLoader(path)
#     docs = loader.load()

#     if not docs:
#         st.error("No valid documents found in the directory!")
#         return

#     ## 2. Split into chunks
#     splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#     chunks = splitter.split_documents(documents=docs)

#     if not chunks:
#         st.error("Could not extract text from the provided PDFs.")
#         return

#     ## 3. Embeddings and Vector DB (Switched to Google Embeddings)
#     embedding_model = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")
#     vectorstore = Chroma.from_documents(
#         documents=chunks,
#         embedding=embedding_model,
#         client=client,
#         collection_name="pdf_rag_collection"
#     )

#     ## 4. Retriever (MMR search)
#     retriever = vectorstore.as_retriever(
#         search_type="mmr",
#         search_kwargs={
#             "k": 4,
#             "fetch_k": 10,
#             "lambda_mult": 0.5
#         }
#     )

#     ## 5. Agent Setup with Gemini LLM
#     llm = ChatGoogleGenerativeAI(model="gemini-3.8-flash")

#     @tool
#     def retrieve_context(query: str) -> str:
#         """Retrieve relevant information from the document knowledge base using MMR search.
        
#         ALWAYS use this tool when answering questions about the uploaded PDF document.
#         """
#         retrieved_docs = retriever.invoke(query)

#         if not retrieved_docs:
#             return "NO_CONTEXT_FOUND"

#         contents = [doc.page_content for doc in retrieved_docs]
#         return "\n\n".join(contents)

#     system_prompt = """You are a strict QA assistant. Your ONLY purpose is to answer questions based on the uploaded PDF document.

# CRITICAL INSTRUCTIONS:
# 1. You MUST ALWAYS call `retrieve_context` for EVERY question asked.
# 2. BASE YOUR ANSWER ENTIRELY ON THE RETRIEVED CONTEXT.
# 3. NEVER use your own pre-trained knowledge or general knowledge to answer questions.
# 4. If `retrieve_context` returns "NO_CONTEXT_FOUND" or if the retrieved text does not contain relevant information, reply EXACTLY with:
#    "I couldn't find any information about that in the uploaded document."
# """

#     tools = [retrieve_context]
#     memory = InMemorySaver()

#     agent = create_agent(
#         model=llm,
#         tools=tools,
#         system_prompt=system_prompt,
#         checkpointer=memory
#     )

#     st.session_state.agent = agent
#     st.session_state.document_uploaded = True


# ### Upload UI
# if not st.session_state.document_uploaded:
#     st.subheader("Upload Documents")
#     uploaded = st.file_uploader(label="Select PDF Files", type=["pdf"], accept_multiple_files=True)
#     if uploaded:
#         with st.spinner("Processing documents..."):
#             target_dir = "doc_files"
#             os.makedirs(target_dir, exist_ok=True)

#             for file in uploaded:
#                 file_path = os.path.join(target_dir, file.name)
#                 with open(file_path, "wb") as f:
#                     f.write(file.getvalue())

#             process_document(target_dir)

#             if st.session_state.document_uploaded:
#                 st.rerun()


# ### Chat UI
# if st.session_state.document_uploaded and st.session_state.agent:
#     st.success("Documents processed successfully! Ask your questions below.")
    
#     for message in st.session_state.messages:
#         role = message.get("role")
#         content = message.get("content")
#         st.chat_message(role).markdown(content)

#     query = st.chat_input("Ask anything related to uploaded documents...")
#     if query:
#         st.session_state.messages.append({"role": "user", "content": query})
#         st.chat_message("user").markdown(query)

#         response = st.session_state.agent.invoke(
#             {"messages": [{"role": "user", "content": query}]},
#             {"configurable": {"thread_id": "1"}}
#         )



import os
import uuid
import streamlit as st
import chromadb

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)
from langchain_community.vectorstores import Chroma
from langgraph.prebuilt import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain.tools import tool


# --------------------------------------------------
# Streamlit page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="DocuTalk - PDF Q&A",
    page_icon="📄",
    layout="wide",
)

st.title("📄 DocuTalk")
st.caption("Upload your PDFs and chat with their contents.")


# --------------------------------------------------
# Session state initialization
# --------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "agent" not in st.session_state:
    st.session_state.agent = None

if "processed_files" not in st.session_state:
    st.session_state.processed_files = []


# --------------------------------------------------
# PDF processing and RAG setup
# --------------------------------------------------

def process_document(path, collection_name):
    """Load PDFs, create embeddings, and build the retrieval agent."""

    # Load all PDFs from the selected directory
    loader = PyPDFDirectoryLoader(path)
    documents = loader.load()

    if not documents:
        raise ValueError(
            "No readable PDF content was found. "
            "Please upload a valid PDF and try again."
        )

    # Split documents into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )

    chunks = text_splitter.split_documents(documents)

    if not chunks:
        raise ValueError(
            "No text chunks could be created from the uploaded PDFs."
        )

    # Generate Google Gemini embeddings
    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-2-preview"
    )

    # Create a uniquely named Chroma collection.
    # This avoids deleting or reusing another session's collection.
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory="chroma_DB",
    )

    # Retrieve relevant chunks using MMR
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 4,
            "fetch_k": 10,
            "lambda_mult": 0.5,
        },
    )

    # Tool used by the agent to retrieve PDF context
    @tool
    def retrieve_context(query: str) -> str:
        """Retrieve relevant information from the uploaded PDF documents.
        ALWAYS use this tool when answering questions about the documents.
        """

        retrieved_docs = retriever.invoke(query)

        if not retrieved_docs:
            return "NO_CONTEXT_FOUND"

        context = "\n\n".join(
            [
                f"Document: {doc.metadata.get('source', 'Uploaded PDF')}\n"
                f"Page: {doc.metadata.get('page', 'Unknown')}\n"
                f"Content: {doc.page_content}"
                for doc in retrieved_docs
            ]
        )

        return context

    # Gemini chat model
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.8-flash"
    )

    # Instructions to keep answers grounded in uploaded PDFs
    system_prompt = """
    You are DocuTalk, a PDF question-answering assistant.

    Rules:
    1. ALWAYS use the retrieve_context tool to answer questions
       about uploaded documents.
    2. Answer using only information found in the retrieved PDF content.
    3. Do not invent facts or use outside knowledge to fill gaps.
    4. If the retrieved content does not contain the answer, clearly
       say that the information could not be found in the uploaded PDFs.
    5. Give clear, direct, and helpful answers.
    6. You may summarize or explain the retrieved content, but do not
       introduce unsupported facts.
    """

    # Preserve the existing LangGraph agent architecture
    checkpointer = InMemorySaver()

    agent = create_agent(
        model=llm,
        tools=[retrieve_context],
        system_prompt=system_prompt,
        checkpointer=checkpointer,
    )

    return agent


# --------------------------------------------------
# Sidebar: PDF upload
# --------------------------------------------------

with st.sidebar:
    st.header("Upload Documents")

    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type=["pdf"],
        accept_multiple_files=True,
    )

    process_button = st.button(
        "Process PDFs",
        type="primary",
        disabled=not uploaded_files,
        use_container_width=True,
    )

    if process_button:
        try:
            # Save uploaded PDFs to a directory
            upload_dir = "uploaded_pdfs"
            os.makedirs(upload_dir, exist_ok=True)

            # Clear previous uploaded files from this working directory
            for filename in os.listdir(upload_dir):
                file_path = os.path.join(upload_dir, filename)

                if os.path.isfile(file_path):
                    os.remove(file_path)

            # Save the currently selected files
            for uploaded_file in uploaded_files:
                safe_name = os.path.basename(uploaded_file.name)
                save_path = os.path.join(upload_dir, safe_name)

                with open(save_path, "wb") as file:
                    file.write(uploaded_file.getvalue())

            # Unique collection name prevents collection collisions
            collection_name = f"pdf_rag_{uuid.uuid4().hex}"

            with st.spinner("Processing PDFs and creating embeddings..."):
                agent = process_document(
                    upload_dir,
                    collection_name,
                )

            # Store the newly created agent for this session
            st.session_state.agent = agent
            st.session_state.collection_name = collection_name
            st.session_state.processed_files = [
                file.name for file in uploaded_files
            ]

            # Start a fresh conversation for the new documents
            st.session_state.messages = []

            st.success("PDFs processed successfully!")

        except Exception as e:
            st.error(
                f"PDF processing failed: {type(e).__name__}: {e}"
            )

    if st.session_state.processed_files:
        st.subheader("Processed PDFs")

        for filename in st.session_state.processed_files:
            st.write(f"📄 {filename}")


# --------------------------------------------------
# Chat interface
# --------------------------------------------------

if st.session_state.agent is None:
    st.info(
        "Upload one or more PDFs from the sidebar and click "
        "**Process PDFs** to start chatting."
    )

else:
    # Display previous conversation messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept a new question
    query = st.chat_input("Ask a question about your PDFs...")

    if query:
        # Display and save the user's question
        st.session_state.messages.append(
            {"role": "user", "content": query}
        )

        with st.chat_message("user"):
            st.markdown(query)

        # Invoke the agent and display its answer
        with st.chat_message("assistant"):
            with st.spinner("Searching your PDFs and generating an answer..."):
                try:
                    response = st.session_state.agent.invoke(
                        {
                            "messages": [
                                {
                                    "role": "user",
                                    "content": query,
                                }
                            ]
                        },
                        {
                            "configurable": {
                                "thread_id": "1"
                            }
                        },
                    )

                    # Extract the final assistant message
                    final_message = response["messages"][-1]
                    content = final_message.content

                    # Handle both plain text and content-block responses
                    if isinstance(content, str):
                        answer = content

                    elif isinstance(content, list):
                        text_parts = []

                        for block in content:
                            if (
                                isinstance(block, dict)
                                and block.get("type") == "text"
                            ):
                                text_parts.append(block.get("text", ""))

                        answer = "\n\n".join(
                            part for part in text_parts if part
                        )

                    else:
                        answer = str(content)

                    if not answer.strip():
                        answer = (
                            "I couldn't generate a text answer. "
                            "Please try asking your question again."
                        )

                    st.markdown(answer)

                    # Save the answer for chat history
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": answer,
                        }
                    )

                except Exception as e:
                    st.error(
                        f"Request failed: {type(e).__name__}: {e}"
                    )


#         answer = response["messages"][-1].text
#         st.chat_message("ai").markdown(answer)
#         st.session_state.messages.append({"role": "ai", "content": answer})
