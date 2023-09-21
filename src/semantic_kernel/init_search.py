"""
Initializes an Azure Cognitive Search index with our custom data, using vector search.
Uses Semantic Kernel.

To run this code, you must already have a "Cognitive Search" and an "OpenAI"
resource created in Azure.
"""
import asyncio
import ntpath
import os

import semantic_kernel as sk
from dotenv import load_dotenv
from langchain.document_loaders import DirectoryLoader, UnstructuredMarkdownLoader
from langchain.text_splitter import Language, RecursiveCharacterTextSplitter
from semantic_kernel.connectors.ai.open_ai import OpenAITextEmbedding
from semantic_kernel.connectors.memory.azure_cognitive_search import (
    AzureCognitiveSearchMemoryStore,
)

# Config for Azure OpenAI.
AZURE_OPENAI_API_TYPE = "azure"
AZURE_OPENAI_API_BASE = os.getenv("AZURE_OPENAI_API_BASE")
AZURE_OPENAI_API_VERSION = "2023-03-15-preview"
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")

# Config for Azure Search.
AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
AZURE_SEARCH_KEY = os.getenv("AZURE_SEARCH_KEY")
AZURE_SEARCH_INDEX_NAME = "blog-posts-index-3"


DATA_DIR = "data/"


def load_and_split_documents() -> list[dict]:
    """
    Loads our documents from disc and split them into chunks.
    Returns a list of dictionaries.
    """
    # Load our data.
    loader = DirectoryLoader(
        DATA_DIR, loader_cls=UnstructuredMarkdownLoader, show_progress=True
    )
    docs = loader.load()

    # Split our documents.
    splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.MARKDOWN, chunk_size=6000, chunk_overlap=100
    )
    split_docs = splitter.split_documents(docs)

    # Convert our LangChain Documents to a list of dictionaries.
    final_docs = []
    for i, doc in enumerate(split_docs):
        doc_dict = {
            "id": str(i),
            "content": doc.page_content,
            "sourcefile": ntpath.basename(doc.metadata["source"]),
        }
        final_docs.append(doc_dict)

    return final_docs


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
            model_id=AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
            api_key=AZURE_OPENAI_API_KEY,
            endpoint=AZURE_OPENAI_API_BASE,
            api_type=AZURE_OPENAI_API_TYPE,
            api_version=AZURE_OPENAI_API_VERSION,
        ),
    )
    kernel.register_memory_store(memory_store)

    # Upload our data to the index.
    for doc in docs:
        await kernel.memory.save_information_async(
            AZURE_SEARCH_INDEX_NAME, id=doc["id"], text=doc["content"]
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
