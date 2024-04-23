# Reference Document: https://python.langchain.com/docs/integrations/vectorstores/chroma/

import traceback

import chromadb
from chromadb.config import Settings
from langchain_chroma import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from constants import OLLAMA_BASE_URL, CHROMA_DB_LOCAL_PATH, CHROMA_SERVER_HOST, CHROMA_SERVER_HTTP_PORT, CHROMA_USE_LOCAL

_OEMBED = OllamaEmbeddings(base_url=OLLAMA_BASE_URL, model="nomic-embed-text")

CLIENT_SETTINGS = {
    "allow_reset": False,
    "anonymized_telemetry": False,
    "is_persistent": True,
}


def connect_chroma():
    """
    Connects to the Chroma DB.
    Returns:
        chromadb.PersistentClient or chromadb.HttpClient
    """
    _chroma = None
    if CHROMA_USE_LOCAL:
        _chroma = chromadb.PersistentClient(
            path=CHROMA_DB_LOCAL_PATH,
            settings=Settings(**CLIENT_SETTINGS),
        )
    else:
        _chroma = chromadb.HttpClient(
            host=CHROMA_SERVER_HOST,
            port=CHROMA_SERVER_HTTP_PORT,
            settings=Settings(**CLIENT_SETTINGS),
        )
    return _chroma


def get_chroma_instance():
    """
    Returns an instance of Chroma with configured embedding function and client.
    """
    return Chroma(embedding_function=_OEMBED, client=connect_chroma())


def get_chroma_collection(instance=get_chroma_instance()):
    """
    Returns the Chroma collection instance.
    """
    return instance._collection


def count_all_documents():
    """
    Counts all documents in the Chroma database.

    Returns:
        int: Total count of documents.
    """
    return get_chroma_collection().count()


def generate_metadata(file):
    """
    Generates metadata for a given file.

    Args:
        file: File object for which metadata is to be generated.

    Returns:
        dict: Metadata dictionary.
    """
    return {"filename": file.name}


def generate_documents(file, raw_text):
    """
    Generates Document objects from raw text content and file metadata.

    Args:
        file: File object containing metadata.
        raw_text: Raw text content to be stored.

    Returns:
        list: List of Document objects.
    """
    return [Document(page_content=raw_text, metadata=generate_metadata(file))]

