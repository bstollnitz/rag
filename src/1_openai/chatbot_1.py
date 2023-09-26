"""
Chatbot with context and memory.
"""
import os

from dotenv import load_dotenv
import openai
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import Vector

# Config for Azure Search.
AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
AZURE_SEARCH_KEY = os.getenv("AZURE_SEARCH_KEY")
AZURE_SEARCH_INDEX_NAME = "products-index-1"

# Config for Azure OpenAI.
AZURE_OPENAI_API_TYPE = "azure"
AZURE_OPENAI_API_BASE = os.getenv("AZURE_OPENAI_API_BASE")
AZURE_OPENAI_API_VERSION = "2023-03-15-preview"
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_CHATGPT_DEPLOYMENT = os.getenv("AZURE_OPENAI_CHATGPT_DEPLOYMENT")
AZURE_OPENAI_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")

# Chat roles
SYSTEM = "system"
USER = "user"
ASSISTANT = "assistant"


class Chatbot:
    """Chat with an LLM using RAG. Keeps chat history in memory."""

    chat_history_list = []

    def __init__(self):
        load_dotenv()
        openai.api_type = AZURE_OPENAI_API_TYPE
        openai.api_base = AZURE_OPENAI_API_BASE
        openai.api_version = AZURE_OPENAI_API_VERSION
        openai.api_key = AZURE_OPENAI_API_KEY

    def _summarize_chat_history_and_query(self, query: str) -> str:
        """
        Creates a user message by summarizing the chat history and user query.
        """
        chat_history = ""
        for entry in self.chat_history_list:
            chat_history += f"{entry['role']}: {entry['content']}\n"
        user_intent_message = {
            "role": USER,
            "content": (
                "You're an AI assistant reading the transcript of a conversation "
                "between a user and an assistant. Given the chat history and user's "
                "query, infer user real intent."
                f"Chat history: ```{chat_history}```\n"
                f"User's query: ```{query}```\n"
            ),
        }
        chat_intent_completion = openai.ChatCompletion.create(
            deployment_id=AZURE_OPENAI_CHATGPT_DEPLOYMENT,
            messages=[user_intent_message],
            temperature=0.7,
            max_tokens=1024,
            n=1,
        )
        user_intent = chat_intent_completion.choices[0].message.content

        return user_intent

    def _get_context(self, user_intent: str) -> list[str]:
        """
        Gets the relevant documents from Azure Cognitive Search.
        """
        query_vector = Vector(
            value=openai.Embedding.create(
                engine=AZURE_OPENAI_EMBEDDING_DEPLOYMENT, input=user_intent
            )["data"][0]["embedding"],
            fields="embedding",
        )

        search_client = SearchClient(
            endpoint=AZURE_SEARCH_ENDPOINT,
            index_name=AZURE_SEARCH_INDEX_NAME,
            credential=AzureKeyCredential(AZURE_SEARCH_KEY),
        )

        docs = search_client.search(search_text="", vectors=[query_vector], top=1)
        context_list = [doc["content"] for doc in docs]

        return context_list

    def _rag(self, context_list: list[str], user_intent: str) -> str:
        """
        Asks the LLM to answer the user's query with the context provided.
        """
        context = "\n\n".join(context_list)
        system_message = {
            "role": SYSTEM,
            "content": (
                "You're a helpful assistant.\n"
                "Please answer the user's question using only information you can find "
                "in the context.\n"
                "If the user's question is unrelated to the information in the "
                "context, say you don't know.\n"
                f"Context: ```{context}```\n"
            ),
        }
        user_intent_message = {"role": USER, "content": user_intent}
        rag_messages = [system_message, user_intent_message]

        chat_completion = openai.ChatCompletion.create(
            deployment_id=AZURE_OPENAI_CHATGPT_DEPLOYMENT,
            messages=rag_messages,
            temperature=0.7,
            max_tokens=1024,
            n=1,
        )
        response = chat_completion.choices[0].message.content

        return response

    def _update_chat_history(self, query: str, response: str) -> None:
        user_message = {"role": USER, "content": query}
        assistant_message = {"role": ASSISTANT, "content": response}
        self.chat_history_list.append(user_message)
        self.chat_history_list.append(assistant_message)

    def ask(self, query: str) -> str:
        """
        Queries an LLM using RAG.
        """
        user_intent = self._summarize_chat_history_and_query(query)
        context_list = self._get_context(user_intent)
        response = self._rag(context_list, user_intent)
        self._update_chat_history(query, response)
        print(
            "*****\n"
            f"QUESTION:\n{query}\n"
            f"USER INTENT:\n{user_intent}\n"
            f"RESPONSE:\n{response}\n"
            "*****\n"
        )

        return response
