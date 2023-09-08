"""
Utilities for interacting with LLMs.
"""
from langchain.document_loaders import DirectoryLoader, UnstructuredMarkdownLoader
from langchain.text_splitter import Language, RecursiveCharacterTextSplitter
from langchain.vectorstores.utils import Document
import ntpath


DATA_DIR = "rag/data/"


def log(title: str, content: str) -> str:
    """
    Prints a title and content to the console.
    """
    print(f"*****\n{title.upper()}:\n{content}\n*****\n")


def load_and_split_documents_langchain() -> list[Document]:
    """
    Load our documents from disc and split them into chunks.
    Returns a list of LancChain Documents.
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

    return split_docs


def load_and_split_documents() -> list[dict]:
    """
    Load our documents from disc and split them into chunks.
    Returns a list of dictionaries.
    """
    docs_langchain = load_and_split_documents_langchain()

    # Convert our LangChain Documents to a list of dictionaries.
    final_docs = []
    for i, doc in enumerate(docs_langchain):
        doc_dict = {
            "Id": str(i),
            "Content": doc.page_content,
            "Filename": ntpath.basename(doc.metadata["source"]),
        }
        final_docs.append(doc_dict)

    return final_docs
