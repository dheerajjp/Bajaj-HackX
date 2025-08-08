
import os

# === Runtime Flags ===
VECTOR_MODE = os.getenv("VECTOR_MODE", "faiss")  # "faiss" or "pinecone"
EMBEDDER = os.getenv("EMBEDDER", "sentence")     # "sentence" or "openai"

# === Auth ===
# Fall back to the provided team token if none set
ALLOWED_BEARER_TOKEN = os.getenv(
    "ALLOWED_BEARER_TOKEN",
    "dc409f1e339103877a936ed2ef35093ce8d2623e7eb5a1bc58a7ee165ea44135"
)

# === OpenAI ===
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# === Pinecone ===
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV", "us-west1-gcp")
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "policy-index")

# === Storage ===
STORAGE_DIR = os.getenv("STORAGE_DIR", "./storage")
os.makedirs(STORAGE_DIR, exist_ok=True)

# === Retrieval Params ===
TOP_K = int(os.getenv("TOP_K", "5"))
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.65"))
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "1200"))  # per chunk trim for token efficiency

# === Server ===
API_PREFIX = "/api/v1"
