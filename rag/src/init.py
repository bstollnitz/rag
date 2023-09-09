"""
Initializes an Azure Cognitive Search index with our custom data, using vector search 
and semantic ranking.

To run this code, you must already have a "Cognitive Search" and an "OpenAI"
resource created in Azure.
"""
import os

import openai
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    HnswParameters,
    HnswVectorSearchAlgorithmConfiguration,
    PrioritizedFields,
    SearchableField,
    SearchField,
    SearchFieldDataType,
    SearchIndex,
    SemanticConfiguration,
    SemanticField,
    SemanticSettings,
    SimpleField,
    VectorSearch,
)
from dotenv import load_dotenv
from utils import load_and_split_documents

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


def get_index(name: str) -> SearchIndex:
    """
    Returns an Azure Cognitive Search index with the given name.
    """
    # The fields we want to index. The "Embedding" field is a vector field that will
    # be used for vector search.
    fields = [
        SimpleField(name="Id", type=SearchFieldDataType.String, key=True),
        SimpleField(name="Filename", type=SearchFieldDataType.String),
        SearchableField(name="Content", type=SearchFieldDataType.String),
        SearchField(
            name="Embedding",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            # Size of the vector created by the text-embedding-ada-002 model.
            vector_search_dimensions=1536,
            vector_search_configuration="default",
        ),
    ]

    # The "Content" field should be prioritized for semantic ranking.
    semantic_settings = SemanticSettings(
        configurations=[
            SemanticConfiguration(
                name="default",
                prioritized_fields=PrioritizedFields(
                    title_field=None,
                    prioritized_content_fields=[SemanticField(field_name="Content")],
                ),
            )
        ]
    )

    # For vector search, we want to use the HNSW (Hierarchical Navigable Small World)
    # algorithm (a type of approximate nearest neighbor search algorithm) with cosine
    # distance.
    vector_search = VectorSearch(
        algorithm_configurations=[
            HnswVectorSearchAlgorithmConfiguration(
                name="default",
                kind="hnsw",
                hnsw_parameters=HnswParameters(metric="cosine"),
            )
        ]
    )

    # Create the search index.
    index = SearchIndex(
        name=name,
        fields=fields,
        semantic_settings=semantic_settings,
        vector_search=vector_search,
    )

    return index


def initialize(search_index_client: SearchIndexClient):
    """
    Initializes an Azure Cognitive Search index with our custom data, using vector
    search.
    """
    # Load our data.
    docs = load_and_split_documents()
    for doc in docs:
        doc["Embedding"] = openai.Embedding.create(
            engine=OPENAI_EMBEDDING_DEPLOYMENT, input=doc["Content"]
        )["data"][0]["embedding"]

    # Create an Azure Cognitive Search index.
    index = get_index(AZURE_SEARCH_INDEX_NAME)
    search_index_client.create_or_update_index(index)

    # Upload our data to the index.
    search_client = SearchClient(
        endpoint=AZURE_SEARCH_ENDPOINT,
        index_name=AZURE_SEARCH_INDEX_NAME,
        credential=AzureKeyCredential(AZURE_SEARCH_KEY),
    )
    search_client.upload_documents(docs)


def delete(search_index_client: SearchIndexClient):
    """
    Deletes the Azure Cognitive Search index.
    """
    search_index_client.delete_index(AZURE_SEARCH_INDEX_NAME)


def main():
    load_dotenv()

    openai.api_type = OPENAI_API_TYPE
    openai.api_base = OPENAI_API_BASE
    openai.api_version = OPENAI_API_VERSION
    openai.api_key = OPENAI_API_KEY

    search_index_client = SearchIndexClient(
        AZURE_SEARCH_ENDPOINT, AzureKeyCredential(AZURE_SEARCH_KEY)
    )

    initialize(search_index_client)
    # delete(search_index_client)


if __name__ == "__main__":
    main()
