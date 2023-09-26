"""
Entry point for the chatbot.
"""
import os

from chatbot_2 import Chatbot
from dotenv import load_dotenv
from langchain.retrievers.azure_cognitive_search import AzureCognitiveSearchRetriever

# Config for Azure Search.
AZURE_SEARCH_KEY = os.getenv("AZURE_SEARCH_KEY")
AZURE_SEARCH_SERVICE_NAME = os.getenv("AZURE_SEARCH_SERVICE_NAME")
AZURE_SEARCH_INDEX_NAME = "products-index-2"


def log(title: str, content: str) -> str:
    """
    Prints a title and content to the console.
    """
    print(f"*****\n{title.upper()}:\n{content}\n*****\n")


def get_context(query: str) -> list[str]:
    """
    Gets the relevant documents from Azure Cognitive Search.
    """
    retriever = AzureCognitiveSearchRetriever(
        api_key=AZURE_SEARCH_KEY,
        service_name=AZURE_SEARCH_SERVICE_NAME,
        index_name=AZURE_SEARCH_INDEX_NAME,
        top_k=3,
    )

    docs = retriever.get_relevant_documents(query)
    context = [doc.page_content for doc in docs]

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
    ask_question(chat, "I need a large backpack. Which one do you recommend?")
    ask_question(chat, "How much does that backpack cost?")
    ask_question(chat, "Explain how whales communicate.")


if __name__ == "__main__":
    main()
