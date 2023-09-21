"""
Chatbot with context and memory, using LangChain.
"""
import os

from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import AzureChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessage,
)
from langchain.schema.document import Document

# Config for Azure OpenAI.
AZURE_OPENAI_API_TYPE = "azure"
AZURE_OPENAI_API_BASE = os.getenv("AZURE_OPENAI_API_BASE")
AZURE_OPENAI_API_VERSION = "2023-03-15-preview"
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_CHATGPT_DEPLOYMENT = os.getenv("AZURE_OPENAI_CHATGPT_DEPLOYMENT")


class Chatbot:
    """Chat with an LLM using LangChain. Keeps chat history in memory."""

    chain = None

    def __init__(self):
        system_message = (
            "You're an assistant helping users learn about GPT models.\n"
            "Please answer the user's question using only information you can find in "
            "the chat history and context, which are enclosed by back ticks in the "
            "user prompt.\n"
            "If the user's question is unrelated to GPT models, say you don't know.\n"
        )

        user_template = (
            "Here's the chat history: ```{chat_history}```\n"
            "Here's the context: ```{context}```\n"
            "Here's my question: ```{question}```\n"
        )

        # Create a chat prompt template.
        chat_template = ChatPromptTemplate(
            messages=[
                SystemMessage(content=system_message),
                MessagesPlaceholder(variable_name="chat_history"),
                HumanMessagePromptTemplate.from_template(user_template),
            ]
        )

        # Create an LLM.
        llm = AzureChatOpenAI(
            deployment_name=AZURE_OPENAI_CHATGPT_DEPLOYMENT,
            openai_api_type=AZURE_OPENAI_API_TYPE,
            openai_api_base=AZURE_OPENAI_API_BASE,
            openai_api_version=AZURE_OPENAI_API_VERSION,
            openai_api_key=AZURE_OPENAI_API_KEY,
            temperature=0.7,
        )

        # Create the memory for our chat history.
        memory = ConversationBufferMemory(
            memory_key="chat_history", input_key="question", return_messages=True
        )

        # Create an LLM chain.
        self.chain = load_qa_chain(llm=llm, prompt=chat_template, memory=memory)

    def ask(self, context_list: list[str], question: str) -> str:
        """
        Queries the LLM including relevant context from our own data.
        """
        input_documents = [Document(page_content=context) for context in context_list]
        response = self.chain(
            {"question": question, "input_documents": input_documents}
        )

        return response["output_text"]
