from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings

from langchain_openai import OpenAIEmbeddings
from html_loader import *
from document_chunk import document_chunking

from langchain.chat_models import init_chat_model
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document

from langchain import hub
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict

#llm = init_chat_model("gpt-4o-mini", model_provider="openai")
#embeddings = DeterministicFakeEmbedding(size=4096)
print("Loading LLM and embeddings...")
#curl -fsSL https://ollama.com/install.sh | sh
#ollama pull llama3.1

llm = ChatOllama(model="llama3.1", temperature=0)
# Use Ollama for embeddings
embeddings = OllamaEmbeddings(model="llama3.1")

vector_store = InMemoryVectorStore(embeddings)

print("Loading HTML documents...")

html_docs = load_html_docs()

print(f"Loaded {len(html_docs)} HTML documents.")
#print(html_docs[0].page_content[:200])  # Print first 200 characters of the first document


document_chunking = document_chunking(html_docs)
print(f"Split blog post into {len(document_chunking)} sub-documents.")

# Index chunks
_ = vector_store.add_documents(documents=document_chunking)

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
    #formatted_prompt = prompt.format(context=retrieved_context, question=question)
    #response_from_model = model.invoke(formatted_prompt)
    #parsed_response = parser.parse(response_from_model)
    #print(f"Answer: {parsed_response}")
    #print()


response = graph.invoke({"question": "What is 5G?"})
print(response["answer"])

