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
from utils import log

# Config for Azure Search.

# Go to https://portal.azure.com/, find your "Cognitive Search" resource,
# and find the "Url".
AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
# On the same resource page, click on "Settings", then "Keys", then copy the
# "Primary admin key".
AZURE_SEARCH_KEY = os.getenv("AZURE_SEARCH_KEY")
# This is the name of the index we created earlier.
AZURE_SEARCH_INDEX_NAME = "blog-posts-index-1"

# Config for Azure OpenAI.

OPENAI_API_TYPE = "azure"
# Go to https://oai.azure.com/, "Chat Playground", "View code", and find
# the API base in the code.
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE")
# In the same window, find the version in the code.
OPENAI_API_VERSION = "2023-03-15-preview"
# In the same window, copy the "Key" at the bottom.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# Go to https://oai.azure.com/, "Deployments", and find the deployment name.
OPENAI_EMBEDDING_DEPLOYMENT = "embedding-deployment"

openai.api_type = OPENAI_API_TYPE
openai.api_base = OPENAI_API_BASE
openai.api_version = OPENAI_API_VERSION
openai.api_key = OPENAI_API_KEY


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

    chat = Chatbot()
    ask_question(chat, "Explain in one or two sentences how attention works.")
    ask_question(chat, "Is it used by the GPT Transformer?")
    ask_question(chat, "Explain how whales communicate.")


if __name__ == "__main__":
    main()
