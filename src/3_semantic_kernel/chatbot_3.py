"""
Chatbot with context and memory, using Semantic Kernel.
"""
import os

import semantic_kernel as sk
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
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

# Chat roles
SYSTEM = "system"
USER = "user"
ASSISTANT = "assistant"


class Chatbot:
    """Chat with an LLM. Keeps chat history in memory."""

    kernel = None
    chat_function = None
    variables = None

    def __init__(self):
        system_message = (
            f"{SYSTEM}:\n"
            "You're a helpful assistant.\n"
            "Please answer the user's question using only information you can find in "
            "the chat history and context, which are enclosed by back ticks in the "
            "user prompt.\n"
            "If the user's question is unrelated to that information, "
            "say you don't know.\n"
        )

        user_template = (
            f"{USER}:\n"
            "Here's the chat history: ```{{$chat_history}}```\n"
            "Here's the context: ```{{$context}}```\n"
            "Here's the user's question: {{$question}}\n"
            f"{ASSISTANT}:\n"
        )

        # Creates a kernel and adds the Azure OpenAI connector to it.
        self.kernel = sk.Kernel()
        self.kernel.add_chat_service(
            "azureopenai",
            AzureChatCompletion(
                deployment_name=AZURE_OPENAI_CHATGPT_DEPLOYMENT,
                endpoint=AZURE_OPENAI_API_BASE,
                api_key=AZURE_OPENAI_API_KEY,
            ),
        )

        # Creates a chat function.
        prompt_config_dict = {
            "type": "completion",
            "description": "A chatbot that's a helpful assistant.",
            "completion": {
                "temperature": 0.7,
                "top_p": 0.5,
                "max_tokens": 200,
                "number_of_responses": 1,
                "chat_system_prompt": system_message,
                "presence_penalty": 0,
                "frequency_penalty": 0,
            },
            "input": {
                "parameters": [
                    {
                        "name": "question",
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

        chat_function_config = SemanticFunctionConfig(prompt_config, prompt_template)

        self.chat_function = self.kernel.register_semantic_function(
            skill_name="chat_plugin",
            function_name="chat_function",
            function_config=chat_function_config,
        )

        # Creates the variables to pass to the chat function.
        self.variables = sk.ContextVariables()
        self.variables["chat_history"] = ""

    async def ask(self, context_list: list[str], question: str) -> str:
        """
        Queries the LLM including relevant context from our own data.
        """
        self.variables["question"] = question
        context = "\n\n".join(context_list)
        self.variables["context"] = context
        response = await self.kernel.run_async(
            self.chat_function, input_vars=self.variables
        )
        self.variables[
            "chat_history"
        ] += f"\n{USER}: {question}\n{ASSISTANT}: {response}\n"

        return str(response)
