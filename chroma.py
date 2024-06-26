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


def store_text_in_vector(file, raw_text):
    """
    Stores text data into the Chroma database after processing.

    Args:
        file: File object containing metadata.
        raw_text: Raw text content to be stored.

    Returns:
        bool: True if storing is successful, False otherwise.
    """
    try:
        deleted_docs = delete_existing_file_documents(file)
        print("Total documents deleted are: %s" % deleted_docs)

        print("Total documents before insert are: %s" % count_all_documents())

        documents = generate_documents(file, raw_text)
        print("Generated documents : %s" % documents)

        store_documents(documents=documents)

        print("Total documents after insert are: %s" % count_all_documents())
        return True
    except Exception as e:
        print("Error while storing documents in Chroma db: %s" % e)
        print(traceback.format_exc())
        return False


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


def split_documents(documents, chunk_size=1000, chunk_overlap=200):
    """
    Splits documents into smaller chunks for processing.

    Args:
        documents: List of Document objects.
        chunk_size: Optional. Size of each chunk. Defaults to 1000.
        chunk_overlap: Optional. Overlap size between chunks. Defaults to 200.

    Returns:
        list: List of Document objects with smaller chunks of text.
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    ).split_documents(documents)


def store_documents(documents):
    """
    Stores Document objects into the Chroma database.

    Args:
        documents: List of Document objects to be stored.

    Returns:
        Chroma instance
    """
    return Chroma.from_documents(
        documents=split_documents(documents), embedding=_OEMBED, client=connect_chroma()
    )


def search_documents(question, vector_store=get_chroma_instance()):
    """
    Searches for documents related to a given question in the Chroma database.

    Args:
        question: Query string to search for.
        vector_store: Optional. Instance of Chroma. Defaults to the global instance.

    Returns:
        List: List of relevant documents.
    """
    return vector_store.similarity_search(question)


def retriever(vector_store=get_chroma_instance()):
    """
    Retrieves documents relevant to a given question from the Chroma database.

    Args:
        question: Query string for retrieval.
        vector_store: Optional. Instance of Chroma. Defaults to the global instance.

    Returns:
        List: List of retrieved documents.
    """
    return vector_store.as_retriever()


def find_document_by_file(file, vector_store=get_chroma_instance()):
    """
    Finds a document in the Chroma database based on the associated file.

    Args:
        file: File object to find the associated document.
        vector_store: Optional. Instance of Chroma. Defaults to the global instance.

    Returns:
        dict: Document information.
    """
    return vector_store.get(where=generate_metadata(file))


def delete_document(documents):
    """
    Deletes documents from the Chroma database.

    Args:
        documents: Dictionary containing document IDs to delete.

    Returns:
        int: Total count of documents deleted.
    """
    ids = documents.get("ids", [])
    total_ids = len(ids)
    if total_ids:
        get_chroma_collection().delete(ids=ids)
    return total_ids


def delete_existing_file_documents(file):
    """
    Deletes documents associated with a specific file from the Chroma database.

    Args:
        file: File object to delete associated documents.

    Returns:
        int: Total count of documents deleted.
    """
    return delete_document(find_document_by_file(file))