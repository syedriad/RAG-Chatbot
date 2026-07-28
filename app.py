from dotenv import load_dotenv
load_dotenv()



from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mistralai import MistralAIEmbeddings,ChatMistralAI
from langchain_community.vectorstores import Chroma

from langgraph.checkpoint.memory import InMemorySaver

# AGENT
from langchain.agents import create_agent
from langchain.tools import tool  #convert a function into tool


embedding_model = MistralAIEmbeddings(model="mistral-embed")

# Load the Documents
loader = PyPDFLoader("CRM in Marketing.pdf")
docs = loader.load()


# Split into chunks

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap=200
)

chunks = splitter.split_documents(docs)

# embeddings and vectorDB


vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory="chroma_DB"
)

# retreiver 
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k":4,
        "fetch_k":10,
        "lambda_mult":0.5
    }
)

# create Agent - (for that we need tool, llm, prompt?)

# llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
llm= ChatMistralAI(model="mistral-small-2506")
@tool
def retrieve_context(query:str):  #(query:str) iska matlb query lega string ki type mai
    """Retrieve relevant information from the document knowledge base using MMR search.
    
    ALWAYS use this tool when answering questions about the uploaded PDF document.
    """

    docs = retriever.invoke(query)

    if not docs:
        return "No relevant content found in the document!"


    
    contents = []
    for doc in docs:
        contents.append(doc.page_content)
        
    context = "\n\n".join(contents)
    
    return context

system_prompt = """You are a helpful assistant that answers questions using retrieved context.
My knowledge base consists of details from the uploaded document.
ALWAYS use the `retrieve_context` tool for questions requiring document knowledge.
If the context doesn't contain the answer, state clearly that you could not find it in the document."""

memory = InMemorySaver()  #is a chat history memory manager provided by LangGraph

# AGENT
agent = create_agent(
    model=llm,
    tools=[retrieve_context],
    system_prompt= system_prompt,
    checkpointer=memory
)

print("RAG CHATBOT")
print("press exit to quit")

while True:
    query= input("You: ")
    if query.lower() == "exit":
        break

    # calling our agent
    response = agent.invoke(
        {"messages":[{"role": "user", "content": query}]},
        {"configurable": {"thread_id":1}}

            )

    result = response["messages"][-1].content  #messages object ka last item [-1]

    print("AI: ", result)