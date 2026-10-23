"""
.env
AZURE_OPENAI_API_KEY=1234
AZURE_OPENAI_ENDPOINT=https://eus2.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT_NAME_LLM=EGPT-4.1
AZURE_OPENAI_API_VERSION=2025-03-01-preview

AZURE_OPENAI_EMBEDDING_MODEL= "text-embedding-3-small" # DEPLOYMENT
AZURE_OPENAI_EMBEDDING_MODEL_API_VERSION="2024-12-01-preview"

The three key technologies
Technology	              Role in your application
Azure embedding model	    Converts questions and document chunks into vectors
In-memory vector store	  Finds chunks similar to the question
Azure chat model	        Writes an answer using those chunks
LangGraph	                Coordinates the processing steps and shared state

Customer Support Agent using LangGraph with RAG.

Handles customer queries using a knowledge base (product docs).
Routes complex issues to human escalation.
Retrieval uses LangChain Core's in-memory vector store with cosine similarity.

Usage:
    python agent.py
    python agent.py --kb-dir docs/          # load custom knowledge base
"""

import argparse
import os
from pathlib import Path
from typing import Annotated, Literal, TypedDict

from dotenv import load_dotenv
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings, AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

load_dotenv()

SAMPLE_KB = [
    "Product: CloudSync Pro. Features: real-time sync across 5 devices, 1TB storage, offline mode, version history 30 days.",
    "Pricing: Basic $9/mo (100GB, 2 devices), Pro $19/mo (1TB, 5 devices), Business $49/mo (5TB, unlimited devices).",
    "Cancellation: Cancel anytime from Account > Subscription > Cancel. Refunds available within 14 days of charge.",
    "Password reset: Go to login page, click 'Forgot Password', enter email. Reset link expires in 1 hour.",
    "Sync issues: Check internet connection, ensure app is updated, try Sign Out and Sign In. If persists, contact support.",
    "Supported platforms: Windows 10+, macOS 12+, iOS 15+, Android 10+, Linux (Beta).",
    "Data security: AES-256 encryption at rest and in transit. SOC 2 Type II certified. Zero-knowledge architecture.",
    "File size limit: Individual files up to 10GB (Pro/Business), 2GB (Basic). No limit on total number of files.",
]

ESCALATION_KEYWORDS = ["refund", "lawsuit", "furious", "fraud", "broken", "data loss", "cancel account", "charge", "billing error"]

"""
Shared state: SupportState - The graph passes a shared dictionary between its processing steps.

Field	              Purpose
messages	          Conversation history
user_input	        Current customer question
retrieved_context	  Relevant knowledge-base text
response	          Answer to display
escalate	          Whether the question matches escalation rules

This declaration is especially important:  messages: Annotated[list, add_messages]
  add_messages tells LangGraph how to merge message updates into the existing history. New messages normally get appended; matching message IDs can update existing messages.

Other fields use ordinary replacement: a new response, for example, replaces the previous answer.

Important distinction: graph nodes return only the fields they update. LangGraph merges those updates into the shared state.
"""
class SupportState(TypedDict):
    messages: Annotated[list, add_messages]
    user_input: str
    retrieved_context: str
    response: str
    escalate: bool

# Retrieval: retrieve_context - This function finds documentation relevant to the question.
def retrieve_context(state: SupportState) -> SupportState:
    query = state["user_input"]
    if not hasattr(retrieve_context, "vectorstore"):
        # Gets the loaded knowledge-base text.
        texts = getattr(retrieve_context, "kb_texts", SAMPLE_KB)
        # Splits it into chunks. targets chunks up to 200 characters, not tokens. 20 - overlapping text between adjacent chunks to help preserve context.
        splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
        docs_split = splitter.create_documents(texts)
        embeddings = AzureOpenAIEmbeddings(model=os.environ['AZURE_OPENAI_EMBEDDING_MODEL']) #OpenAIEmbeddings()
        # Embeds the chunks and stores them in memory.
        # The embedding model converts text into numerical vectors representing meaning.
        # This builds a searchable collection of document chunks and their vectors.
        retrieve_context.vectorstore = InMemoryVectorStore.from_documents(docs_split, embeddings)

    # Every question: search the vector store. This embeds the question and retrieves up to three similar chunks, using cosine similarity.
    docs = retrieve_context.vectorstore.similarity_search(query, k=3)
    # The chunks are combined into one string
    context = "\n".join(d.page_content for d in docs)
    return {"retrieved_context": context}

"""  
Question	                        Escalates?
“How do I reset my password?”	    No
“I need a refund.”	              Yes
“My account was charged twice.”	  Yes—contains charge
"""
def check_escalation(state: SupportState) -> SupportState:
    text = state["user_input"].lower()
    needs_escalation = any(kw in text for kw in ESCALATION_KEYWORDS)
    return {"escalate": needs_escalation}

"""
A low temperature generally aims for less varied responses; it does not guarantee factual accuracy.

The prompt contains:
  System instructions + retrieved documentation
  Previous conversation
  Current customer question
"""
def generate_response(state: SupportState) -> SupportState:
    #llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
    llm = AzureChatOpenAI(
        azure_deployment=os.environ['AZURE_OPENAI_DEPLOYMENT_NAME_LLM'],
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT_LLM"],
        api_version="2025-03-01-preview",
        temperature=0.2,
    )
    conversation = state["messages"][:-1]  # exclude latest user msg
    
    if state.get("escalate"):
        response_text = "I understand your concern and I want to make sure this gets the attention it deserves. I'm connecting you with a senior support specialist who can resolve this directly. You'll hear back within 2 hours. Your case ID is #" + str(hash(state["user_input"]) % 100000) + "."
    else:
        messages = [
            SystemMessage(content=f"""You are a helpful customer support agent for CloudSync Pro.
Use this knowledge base context to answer accurately:
{state['retrieved_context']}

Be friendly, concise, and solution-focused. If unsure, say so honestly."""),
            *conversation,
            HumanMessage(content=state["user_input"]),
        ]
        response = llm.invoke(messages)
        response_text = response.content

    return {"response": response_text, "messages": [AIMessage(content=response_text)]}


def route_after_escalation_check(state: SupportState) -> Literal["generate", "generate"]:
    return "generate"

"""
Workflow: build_graph - This connects the three functions into a LangGraph workflow: retrieve → check_escalation → generate → END
  add_node() registers a processing step.
  set_entry_point() selects the first step.
  add_edge() defines execution order.
  compile() creates an executable graph.
"""  
def build_graph():
    graph = StateGraph(SupportState)
    graph.add_node("retrieve", retrieve_context)
    graph.add_node("check_escalation", check_escalation)
    graph.add_node("generate", generate_response)
    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "check_escalation")
    graph.add_edge("check_escalation", "generate")
    graph.add_edge("generate", END)
    return graph.compile()


def load_kb_texts(kb_dir: str | None) -> list[str]:
    if not kb_dir:
        return SAMPLE_KB

    root = Path(kb_dir)
    if not root.is_dir():
        raise ValueError(f"Knowledge base directory does not exist: {kb_dir}")

    texts = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in {".txt", ".md"}:
            texts.append(path.read_text(encoding="utf-8"))

    if not texts:
        raise ValueError(f"No .txt or .md files found in knowledge base directory: {kb_dir}")

    return texts


def main():
    parser = argparse.ArgumentParser(description="Customer Support Agent")
    parser.add_argument("--kb-dir", help="Directory containing .txt or .md support knowledge base files")
    args = parser.parse_args()

    retrieve_context.kb_texts = load_kb_texts(args.kb_dir)
    if hasattr(retrieve_context, "vectorstore"):
        delattr(retrieve_context, "vectorstore")
    
    agent = build_graph()
   
    state = {"messages": [], "user_input": "", "retrieved_context": "", "response": "", "escalate": False}

    print("\n🎧 Customer Support Agent (CloudSync Pro)")
    print("Type 'quit' to exit\n")

    while True:
        user_input = input("Customer: ").strip()
        if user_input.lower() in ("quit", "exit", "q"):
            break
        if not user_input:
            continue

        state["user_input"] = user_input
        state["messages"].append(HumanMessage(content=user_input))
        state = agent.invoke(state)

        escalation_indicator = " [ESCALATED]" if state.get("escalate") else ""
        print(f"\nAgent{escalation_indicator}: {state['response']}\n")


if __name__ == "__main__":
    main()
