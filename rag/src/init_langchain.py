"""
Initializes an Azure Cognitive Search index with our custom data, using vector search.
Uses LangChain.

To run this code, you must already have a "Cognitive Search" and an "OpenAI"
resource created in Azure.
"""
import os

import openai
from dotenv import load_dotenv
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.azuresearch import AzureSearch
from utils import load_and_split_documents_langchain

# Config for Azure OpenAI.
OPENAI_API_TYPE = "azure"
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE")
OPENAI_API_VERSION = "2023-03-15-preview"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_EMBEDDING_DEPLOYMENT = "embedding-deployment"

# Config for Azure Search.
AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
AZURE_SEARCH_KEY = os.getenv("AZURE_SEARCH_KEY")
AZURE_SEARCH_INDEX_NAME = "blog-posts-index-2"


def initialize():
    """
    Initializes an Azure Cognitive Search index with our custom data.
    """
    # Load our data.
    docs = load_and_split_documents_langchain()

    # Create an Azure Cognitive Search index.
    embeddings_parameters = {"engine": OPENAI_EMBEDDING_DEPLOYMENT}
    embeddings = OpenAIEmbeddings(model_kwargs=embeddings_parameters)
    vector_store = AzureSearch(
        azure_search_endpoint=AZURE_SEARCH_ENDPOINT,
        azure_search_key=AZURE_SEARCH_KEY,
        index_name=AZURE_SEARCH_INDEX_NAME,
        embedding_function=embeddings.embed_query,
    )

    # Upload our data to the index.
    vector_store.add_documents(documents=docs)


def main():
    load_dotenv()

    openai.api_type = OPENAI_API_TYPE
    openai.api_base = OPENAI_API_BASE
    openai.api_version = OPENAI_API_VERSION
    openai.api_key = OPENAI_API_KEY

    initialize()


if __name__ == "__main__":
    main()
