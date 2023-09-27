"""
Chatbot with context and memory, using Semantic Kernel.
"""
import os

import semantic_kernel as sk
from semantic_kernel.connectors.ai.open_ai import (
    AzureChatCompletion,
    OpenAITextEmbedding,
)
from semantic_kernel.connectors.memory.azure_cognitive_search import (
    AzureCognitiveSearchMemoryStore,
)
from semantic_kernel.semantic_functions.chat_prompt_template import ChatPromptTemplate
from semantic_kernel.semantic_functions.prompt_template_config import (
    PromptTemplateConfig,
)
from semantic_kernel.semantic_functions.semantic_function_config import (
    SemanticFunctionConfig,
)

# Config for Azure OpenAI.
AZURE_OPENAI_API_TYPE = "azure"
AZURE_OPENAI_API_BASE = os.getenv("AZURE_OPENAI_API_BASE")
AZURE_OPENAI_API_VERSION = "2023-03-15-preview"
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_CHATGPT_DEPLOYMENT = os.getenv("AZURE_OPENAI_CHATGPT_DEPLOYMENT")
AZURE_OPENAI_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")

# Config for Azure Search.
AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
AZURE_SEARCH_KEY = os.getenv("AZURE_SEARCH_KEY")
AZURE_SEARCH_INDEX_NAME = "products-index-3"

# Chat roles
SYSTEM = "system"
USER = "user"
ASSISTANT = "assistant"

PLUGIN_NAME = "rag_plugin"


class Chatbot:
    """Chat with an LLM. Keeps chat history in memory."""

    kernel = None
    variables = None

    def __init__(self):
        # Create a kernel and adds the Azure OpenAI connector to it.
        self.kernel = sk.Kernel()
        self.kernel.add_chat_service(
            "azureopenai",
            AzureChatCompletion(
                deployment_name=AZURE_OPENAI_CHATGPT_DEPLOYMENT,
                endpoint=AZURE_OPENAI_API_BASE,
                api_key=AZURE_OPENAI_API_KEY,
            ),
        )

        # Create the variables to pass to the chat functions.
        self.variables = sk.ContextVariables()
        self.variables["chat_history"] = ""

    async def _summarize_user_intent(self, query: str) -> str:
        """
        Creates a user message containing the user intent, by summarizing the chat
        history and user query.
        """
        # Define the user template.
        self.variables["query"] = query
        user_template = (
            "You're an AI assistant reading the transcript of a conversation "
            "between a user and an assistant. Given the chat history and "
            "user's query, infer user real intent."
            "Chat history: ```{{$chat_history}}```\n"
            "User's query: ```{{$query}}```\n"
        )

        # Create a semantic function.
        prompt_config_dict = {
            "type": "completion",
            "description": "An AI assistant that infers user intent.",
            "completion": {
                "temperature": 0.7,
                "top_p": 0.5,
                "max_tokens": 200,
                "number_of_responses": 1,
                "presence_penalty": 0,
                "frequency_penalty": 0,
            },
            "input": {
                "parameters": [
                    {
                        "name": "query",
                        "description": "The question asked by the user.",
                        "defaultValue": "",
                    },
                    {
                        "name": "chat_history",
                        "description": "All the user and assistant messages so far .",
                        "defaultValue": "",
                    },
                ]
            },
        }
        prompt_config = PromptTemplateConfig.from_dict(prompt_config_dict)
        prompt_template = ChatPromptTemplate(
            template=user_template,
            prompt_config=prompt_config,
            template_engine=self.kernel.prompt_template_engine,
        )
        user_intent_function_config = SemanticFunctionConfig(
            prompt_config, prompt_template
        )
        user_intent_function = self.kernel.register_semantic_function(
            skill_name=PLUGIN_NAME,
            function_name="user_intent_function",
            function_config=user_intent_function_config,
        )

        # Run the semantic function.
        response = await self.kernel.run_async(
            user_intent_function, input_vars=self.variables
        )

        return str(response)

    async def _get_context(self, query: str) -> list[str]:
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

    async def _rag(self, context_list: list[str], query: str) -> str:
        """
        Asks the LLM to answer the user's query with the context provided.
        """
        # Define the system template.
        context = "\n\n".join(context_list)
        self.variables["context"] = context
        system_template = (
            "You're a helpful assistant.\n"
            "Please answer the user's question using only information you can "
            "find in the context.\n"
            "If the user's question is unrelated to the information in the "
            "context, say you don't know.\n"
            "Context: ```{{$context}}```\n"
        )

        # Define the user template.
        self.variables["query"] = query
        user_template = "{{$chat_history}}" + f"{USER}: " + "{{$query}}\n"

        # Create a semantic function.
        prompt_config_dict = {
            "type": "completion",
            "description": "A chatbot that's a helpful assistant.",
            "completion": {
                "temperature": 0.7,
                "top_p": 0.5,
                "max_tokens": 200,
                "number_of_responses": 1,
                "chat_system_prompt": system_template,
                "presence_penalty": 0,
                "frequency_penalty": 0,
            },
            "input": {
                "parameters": [
                    {
                        "name": "query",
                        "description": "The question asked by the user.",
                        "defaultValue": "",
                    },
                    {
                        "name": "context",
                        "description": "The context for the assistant's answer.",
                        "defaultValue": "",
                    },
                    {
                        "name": "chat_history",
                        "description": "All the user and assistant messages so far.",
                        "defaultValue": "",
                    },
                ]
            },
        }
        prompt_config = PromptTemplateConfig.from_dict(prompt_config_dict)
        prompt_template = ChatPromptTemplate(
            template=user_template,
            prompt_config=prompt_config,
            template_engine=self.kernel.prompt_template_engine,
        )
        rag_function_config = SemanticFunctionConfig(prompt_config, prompt_template)
        rag_function = self.kernel.register_semantic_function(
            skill_name=PLUGIN_NAME,
            function_name="rag_function",
            function_config=rag_function_config,
        )

        # Run the semantic function.
        response = await self.kernel.run_async(rag_function, input_vars=self.variables)
        self.variables["chat_history"] += f"{USER}: {query}\n{ASSISTANT}: {response}\n"

        return str(response)

    async def ask(self, query: str) -> str:
        """
        Queries an LLM using RAG.
        """
        user_intent = await self._summarize_user_intent(query)
        context_list = await self._get_context(user_intent)
        response = await self._rag(context_list, query)
        print(
            "*****\n"
            f"QUESTION:\n{query}\n"
            f"USER INTENT:\n{user_intent}\n"
            f"RESPONSE:\n{response}\n"
            "*****\n"
        )

        return response
