"""
Entry point for the chatbot.
"""
import asyncio
import os

import semantic_kernel as sk
from chatbot_3 import Chatbot
from dotenv import load_dotenv
from semantic_kernel.connectors.ai.open_ai import OpenAITextEmbedding
from semantic_kernel.connectors.memory.azure_cognitive_search import (
    AzureCognitiveSearchMemoryStore,
)

# Config for Azure Search.
AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
AZURE_SEARCH_KEY = os.getenv("AZURE_SEARCH_KEY")
AZURE_SEARCH_INDEX_NAME = "products-index-3"

# Config for Azure OpenAI.
AZURE_OPENAI_API_TYPE = "azure"
AZURE_OPENAI_API_BASE = os.getenv("AZURE_OPENAI_API_BASE")
AZURE_OPENAI_API_VERSION = "2023-03-15-preview"
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")


def log(title: str, content: str) -> str:
    """
    Prints a title and content to the console.
    """
    print(f"*****\n{title.upper()}:\n{content}\n*****\n")


async def get_context(query: str) -> list[str]:
    """
    Gets the relevant documents from Azure Cognitive Search.
    """
    kernel = sk.Kernel()
    kernel.add_text_embedding_generation_service(
        "openai-embedding",
        OpenAITextEmbedding(
            model_id=AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
            api_key=AZURE_OPENAI_API_KEY,
            endpoint=AZURE_OPENAI_API_BASE,
            api_type=AZURE_OPENAI_API_TYPE,
            api_version=AZURE_OPENAI_API_VERSION,
        ),
    )
    kernel.register_memory_store(
        memory_store=AzureCognitiveSearchMemoryStore(
            vector_size=1536,
            search_endpoint=AZURE_SEARCH_ENDPOINT,
            admin_key=AZURE_SEARCH_KEY,
        )
    )

    docs = await kernel.memory.search_async(AZURE_SEARCH_INDEX_NAME, query, limit=1)
    context = [doc.text for doc in docs]

    return context


async def ask_question(chat: Chatbot, question: str):
    """
    Get the context for the user's question, and ask the Chatbot that question.
    """
    log("QUESTION", question)
    context_list = await get_context(question)
    response = await chat.ask(context_list, question)
    log("RESPONSE", response)


async def main():
    load_dotenv()

    chat = Chatbot()
    await ask_question(chat, "I need a large backpack. Which one do you recommend?")
    await ask_question(chat, "How much does that backpack cost?")
    await ask_question(chat, "Explain how whales communicate.")


if __name__ == "__main__":
    asyncio.run(main())
