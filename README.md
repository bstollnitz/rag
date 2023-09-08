## Bringing your own data to LLMs

This project shows how to create a Chatbot that can converse with your own data, using Azure Cognitive Search, and that remembers the message history. It shows 3 approaches to the problem: interacting directly with OpenAI APIs, using LangChain, and using Semantic Kernel.  


## Pre-requisites
- You need to have an Azure subscription. You can get a [free subscription](https://azure.microsoft.com/en-us/free) to try it out.
- Create a "Cognitive Search" resource on Azure.
- Create an "OpenAI" resource on Azure. Create two deployments within this resource: one for the "text-embedding-ada-002" model called "embedding-deployment", and another for the "gpt-35-turbo" model called "chatgpt-deployment".
- Add a ".env" file to the project with the following variables set:
    - OPENAI_API_BASE
    - OPENAI_API_KEY
    - AZURE_SEARCH_ENDPOINT
    - AZURE_SEARCH_KEY
    - AZURE_SEARCH_SERVICE_NAME

## How to run

### Code that interacts directly with OpenAI APIs

- *Run init.py* by opening the file and pressing F5. This creates an Azure Cognitive Search index (a vector database) containing the data in the "data" directory.
- *Run rag.py.* This asks an LLM a sequence of questions, using our data as context, and keeping chat history.

### Code that uses LangChain

- *Run init_langchain.py.*
- *Run rag_langchain.py.*

### Code that uses Semantic Kernel

- *Run init_semantic_kernel.py.*
- *Run rag_semantic_kernel.py.*
