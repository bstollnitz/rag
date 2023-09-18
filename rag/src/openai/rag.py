"""
Bring your own data to an OpenAI LLM using Azure Cognitive Search with vector search
and semantic ranking.
"""
import os

import openai
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import Vector
from chatbot import Chatbot
from dotenv import load_dotenv

# Config for Azure Search.
AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
AZURE_SEARCH_KEY = os.getenv("AZURE_SEARCH_KEY")
AZURE_SEARCH_INDEX_NAME = "blog-posts-index-1"

# Config for Azure OpenAI.
OPENAI_API_TYPE = "azure"
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE")
OPENAI_API_VERSION = "2023-03-15-preview"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_EMBEDDING_DEPLOYMENT = "embedding-deployment"


def log(title: str, content: str) -> str:
    """
    Prints a title and content to the console.
    """
    print(f"*****\n{title.upper()}:\n{content}\n*****\n")


def get_context(question: str) -> str:
    """
    Gets the relevant documents from Azure Cognitive Search.
    """
    query_vector = Vector(
        value=openai.Embedding.create(
            engine=OPENAI_EMBEDDING_DEPLOYMENT, input=question
        )["data"][0]["embedding"],
        fields="Embedding",
    )

    search_client = SearchClient(
        endpoint=AZURE_SEARCH_ENDPOINT,
        index_name=AZURE_SEARCH_INDEX_NAME,
        credential=AzureKeyCredential(AZURE_SEARCH_KEY),
    )

    docs = search_client.search(
        search_text="",
        vectors=[query_vector],
    )
    context = [doc["Content"] for doc in docs]

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

    openai.api_type = OPENAI_API_TYPE
    openai.api_base = OPENAI_API_BASE
    openai.api_version = OPENAI_API_VERSION
    openai.api_key = OPENAI_API_KEY

    chat = Chatbot()
    ask_question(chat, "Explain in one or two sentences how attention works.")
    ask_question(chat, "Is it used by the GPT Transformer?")
    ask_question(chat, "Explain how whales communicate.")


if __name__ == "__main__":
    main()
