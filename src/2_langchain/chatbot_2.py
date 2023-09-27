"""
Chatbot with context and memory, using LangChain.
"""
import os

from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain.chat_models import AzureChatOpenAI
from langchain.memory import ChatMessageHistory
from langchain.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate
from langchain.retrievers.azure_cognitive_search import AzureCognitiveSearchRetriever

# Config for Azure OpenAI.
AZURE_OPENAI_API_TYPE = "azure"
AZURE_OPENAI_API_BASE = os.getenv("AZURE_OPENAI_API_BASE")
AZURE_OPENAI_API_VERSION = "2023-03-15-preview"
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_CHATGPT_DEPLOYMENT = os.getenv("AZURE_OPENAI_CHATGPT_DEPLOYMENT")

# Config for Azure Search.
AZURE_SEARCH_KEY = os.getenv("AZURE_SEARCH_KEY")
AZURE_SEARCH_SERVICE_NAME = os.getenv("AZURE_SEARCH_SERVICE_NAME")
AZURE_SEARCH_INDEX_NAME = "products-index-2"


class Chatbot:
    """Chat with an LLM using LangChain. Keeps chat history in memory."""

    history = ChatMessageHistory()

    def __init__(self):
        load_dotenv()

    def _summarize_user_intent(self, query: str) -> str:
        """
        Creates a user message containing the user intent, by summarizing the chat
        history and user query.
        """
        llm = AzureChatOpenAI(
            deployment_name=AZURE_OPENAI_CHATGPT_DEPLOYMENT,
            openai_api_type=AZURE_OPENAI_API_TYPE,
            openai_api_base=AZURE_OPENAI_API_BASE,
            openai_api_version=AZURE_OPENAI_API_VERSION,
            openai_api_key=AZURE_OPENAI_API_KEY,
            temperature=0.7,
        )

        chat_history_str = ""
        for entry in self.history.messages:
            chat_history_str += entry.type + ": " + entry.content + "\n "
        system_template = (
            "You're an AI assistant reading the transcript of a conversation "
            "between a user and an assistant. Given the chat history and "
            "user's query, infer user real intent."
            "Chat history: ```{chat_history_str}```\n"
            "User's query: ```{query}```\n"
        )
        chat_template = ChatPromptTemplate(
            messages=[SystemMessagePromptTemplate.from_template(system_template)]
        )

        llm_chain = LLMChain(llm=llm, prompt=chat_template)
        user_intent = llm_chain({"chat_history_str": chat_history_str, "query": query})[
            "text"
        ]

        return user_intent

    def _get_context(self, user_intent: str) -> list[str]:
        """
        Gets the relevant documents from Azure Cognitive Search.
        """
        retriever = AzureCognitiveSearchRetriever(
            api_key=AZURE_SEARCH_KEY,
            service_name=AZURE_SEARCH_SERVICE_NAME,
            index_name=AZURE_SEARCH_INDEX_NAME,
            top_k=3,
        )

        docs = retriever.get_relevant_documents(user_intent)
        context = [doc.page_content for doc in docs]

        return context

    def _rag(self, context_list: list[str], query: str) -> str:
        """
        Asks the LLM to answer the user's query with the context provided.
        """
        self.history.add_user_message(query)

        llm = AzureChatOpenAI(
            deployment_name=AZURE_OPENAI_CHATGPT_DEPLOYMENT,
            openai_api_type=AZURE_OPENAI_API_TYPE,
            openai_api_base=AZURE_OPENAI_API_BASE,
            openai_api_version=AZURE_OPENAI_API_VERSION,
            openai_api_key=AZURE_OPENAI_API_KEY,
            temperature=0.7,
        )

        context = "\n\n".join(context_list)
        system_template = (
            "You're a helpful assistant.\n"
            "Please answer the user's question using only information you can "
            "find in the context.\n"
            "If the user's question is unrelated to the information in the "
            "context, say you don't know.\n"
            "Context: ```{context}```\n"
        )
        chat_template = ChatPromptTemplate(
            messages=[SystemMessagePromptTemplate.from_template(system_template)]
            + self.history.messages
        )

        llm_chain = LLMChain(llm=llm, prompt=chat_template)
        response = llm_chain({"context": context})["text"]
        self.history.add_ai_message(response)

        return response

    def ask(self, query: str) -> str:
        """
        Queries an LLM using RAG.
        """
        user_intent = self._summarize_user_intent(query)
        context_list = self._get_context(user_intent)
        response = self._rag(context_list, query)
        print(
            "*****\n"
            f"QUESTION:\n{query}\n"
            f"USER INTENT:\n{user_intent}\n"
            f"RESPONSE:\n{response}\n"
            "*****\n"
        )
