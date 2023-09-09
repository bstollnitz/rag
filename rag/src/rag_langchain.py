"""
Bring your own data to an OpenAI LLM using Azure Cognitive Search with vector search.
Uses LangChain.
"""
import os

from chatbot_langchain import Chatbot
from dotenv import load_dotenv
from langchain.retrievers.azure_cognitive_search import AzureCognitiveSearchRetriever
from utils import log

# Config for Azure Search.
AZURE_SEARCH_KEY = os.getenv("AZURE_SEARCH_KEY")
AZURE_SEARCH_SERVICE_NAME = os.getenv("AZURE_SEARCH_SERVICE_NAME")
AZURE_SEARCH_INDEX_NAME = "blog-posts-index-2"


def get_context(query: str) -> list[str]:
    """
    Gets the relevant documents from Azure Cognitive Search.
    """
    retriever = AzureCognitiveSearchRetriever(
        api_key=AZURE_SEARCH_KEY,
        service_name=AZURE_SEARCH_SERVICE_NAME,
        index_name=AZURE_SEARCH_INDEX_NAME,
        top_k=1,
    )

    docs = retriever.get_relevant_documents(query)
    context = [doc.page_content for doc in docs]

    return context


def ask_question(chat: Chatbot, question: str):
    """
    Get the context for the user's question, and ask the Chatbot that question.
    """
    log("QUESTION", question)
    context_list = get_context(question)
    response = chat.ask(context_list, question)
    log("RESPONSE", response)


def main():
    load_dotenv()

    chat = Chatbot()
    ask_question(chat, "Explain in one or two sentences how attention works.")
    ask_question(chat, "Is it used by the GPT Transformer?")
    ask_question(chat, "Explain how whales communicate.")


if __name__ == "__main__":
    main()
