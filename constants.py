import os
from dotenv import load_dotenv

load_dotenv()

# AI Constants
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://0.0.0.0:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")

# Chroma DB
CHROMA_SERVER_HOST = os.getenv("CHROMA_SERVER_HOST", "0.0.0.0")
CHROMA_SERVER_HTTP_PORT = os.getenv("CHROMA_SERVER_HTTP_PORT", "8000")
CHROMA_DB_LOCAL_PATH = os.getenv("CHROMA_DB_LOCAL_PATH", "chroma_db")
CHROMA_DB_COLLECTION = os.getenv("CHROMA_DB_COLLECTION", "tngpt")

if os.getenv("CHROMA_USE_LOCAL", "false") == "true":
    CHROMA_USE_LOCAL = True
else:
    CHROMA_USE_LOCAL = False
