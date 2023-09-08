"""
Initializes an Azure Cognitive Search index with our custom data, using vector search.
Uses Semantic Kernel.

To run this code, you must already have a "Cognitive Search" and an "OpenAI"
resource created in Azure.
"""
import asyncio
import os

import semantic_kernel as sk
from dotenv import load_dotenv
from semantic_kernel.connectors.ai.open_ai import OpenAITextEmbedding
from semantic_kernel.connectors.memory.azure_cognitive_search import (
    AzureCognitiveSearchMemoryStore,
)
from utils import load_and_split_documents

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

# Config for Azure Search.

# Go to https://portal.azure.com/, find your "Cognitive Search" resource,
# and find the "Url".
AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
# On the same resource page, click on "Settings", then "Keys", then copy the
# "Primary admin key".
AZURE_SEARCH_KEY = os.getenv("AZURE_SEARCH_KEY")
# The name of the index we'll create.
AZURE_SEARCH_INDEX_NAME = "blog-posts-index-3"


async def initialize(memory_store: AzureCognitiveSearchMemoryStore):
    """
    Initializes an Azure Cognitive Search index with our custom data.
    """
    # Load our data.
    docs = load_and_split_documents()

    # Create an Azure Cognitive Search index.
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
    kernel.register_memory_store(memory_store)

    # Upload our data to the index.
    for doc in docs:
        await kernel.memory.save_information_async(
            AZURE_SEARCH_INDEX_NAME, id=doc["Id"], text=doc["Content"]
        )


async def delete(memory_store: AzureCognitiveSearchMemoryStore):
    """
    Deletes the Azure Cognitive Search index.
    """
    await memory_store.delete_collection_async(AZURE_SEARCH_INDEX_NAME)


async def main():
    load_dotenv()

    memory_store = AzureCognitiveSearchMemoryStore(
        vector_size=1536,
        search_endpoint=AZURE_SEARCH_ENDPOINT,
        admin_key=AZURE_SEARCH_KEY,
    )

    await initialize(memory_store)
    # await delete(memory_store)


if __name__ == "__main__":
    asyncio.run(main())
