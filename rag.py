from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from html_loader import *
from document_chunk import document_chunking

from langchain_core.documents import Document
from langchain import hub
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict
import os

from langchain.chat_models import init_chat_model
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document

#llm = init_chat_model("gpt-4o-mini", model_provider="openai")
#embeddings = DeterministicFakeEmbedding(size=4096)
print("Loading LLM and embeddings...")
#curl -fsSL https://ollama.com/install.sh | sh
#ollama pull llama3.1

llm = ChatOllama(model="llama3.1", temperature=0)
# Use Ollama for embeddings
embeddings = OllamaEmbeddings(model="llama3.1")



# Index chunks
PERSIST_DIR = "./vector_store"

if not os.path.exists(PERSIST_DIR):
    os.makedirs(PERSIST_DIR)
    # Load HTML docs
    html_docs = load_html_docs()
    print(f"Loaded {len(html_docs)} HTML documents.")
    print(html_docs[0].page_content[:200])
    
    # Chunk documents
    document_chunks = document_chunking(html_docs)
    print(f"Split blog post into {len(document_chunks)} sub-documents.")
    
    # Create Chroma DB and persist
    vector_store = Chroma.from_documents(documents=document_chunks, embedding=embeddings, persist_directory=PERSIST_DIR)
    vector_store.persist()
else:
    vector_store = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)


# 4. Load prompt from LangChain hub

prompt = hub.pull("rlm/rag-prompt")


## Define state for application
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str


# Define application steps
def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}


def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}


# Compile application and test
graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()

print(graph)
# Test the application

while True:
    print("Say 'exit' or 'quit' to exit the loop")
    question = input('User question: ')
    print(f"Question: {question}")
    if question.lower() in ["exit", "quit"]:
        print("Exiting the conversation. Goodbye!")
        break
    response = graph.invoke({"question": question})
    print(response["answer"])



