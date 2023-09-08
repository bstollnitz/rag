"""
Bring your own data to an OpenAI LLM using Azure Cognitive Search with vector search.
Uses Semantic Kernel.
"""
import asyncio
import os

import semantic_kernel as sk
from chatbot_semantic_kernel import Chatbot
from dotenv import load_dotenv
from semantic_kernel.connectors.ai.open_ai import OpenAITextEmbedding
from semantic_kernel.connectors.memory.azure_cognitive_search import (
    AzureCognitiveSearchMemoryStore,
)
from utils import log

# Config for Azure Search.

# Go to https://portal.azure.com/, find your "Cognitive Search" resource,
# and find the "Url".
AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
# On the same resource page, click on "Settings", then "Keys", then copy the
# "Primary admin key".
AZURE_SEARCH_KEY = os.getenv("AZURE_SEARCH_KEY")
# The name of the index we'll create.
AZURE_SEARCH_INDEX_NAME = "blog-posts-index-3"


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


def get_context(query: str) -> list[str]:
    """
    Gets the relevant documents from Azure Cognitive Search.
    """
    kernel = sk.Kernel()
    kernel.add_text_embedding_generation_service(
        "openai-embedding",
        OpenAITextEmbedding(
            model_id=OPENAI_EMBEDDING_DEPLOYMENT,
            api_key=OPENAI_API_KEY,
            endpoint=OPENAI_API_BASE,
            api_type=OPENAI_API_TYPE,
            api_version=OPENAI_API_VERSION,
        ),
    )
    kernel.register_memory_store(
        memory_store=AzureCognitiveSearchMemoryStore(
            vector_size=1536,
            search_endpoint=AZURE_SEARCH_ENDPOINT,
            admin_key=AZURE_SEARCH_KEY,
        )
    )

    docs = asyncio.run(
        kernel.memory.search_async(AZURE_SEARCH_INDEX_NAME, query, limit=1)
    )
    context = [doc.text for doc in docs]

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
