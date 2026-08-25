# src/config.py
import os
import yaml

from dotenv import load_dotenv

# 1. Load the secret keys from the .env file into the system environment
load_dotenv()

TEST_DOCUMENT_PATH = os.getenv("TEST_DOCUMENT_PATH")

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")

OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# 2. Parse the structural hyperparameters from the YAML configuration
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "configs", "config.yaml")
with open(CONFIG_PATH, "r") as f:
    _yaml_config = yaml.safe_load(f)


# Expose them as clean, accessible Python variables
LLM_SOURCE = _yaml_config["provider"]["llm_source"].lower()
EMBEDDING_SOURCE = _yaml_config["provider"]["embedding_source"].lower()

if LLM_SOURCE not in ["google", "ollama"]:
    raise ValueError(
        f"Unsupported LLM source: {LLM_SOURCE}. Supported sources are 'google' and 'ollama'."
    )

if EMBEDDING_SOURCE not in ["google", "ollama"]:
    raise ValueError(
        f"Unsupported embedding source: {EMBEDDING_SOURCE}. Supported sources are 'google' and 'ollama'."
    )


GOOGLE_LLM = _yaml_config["models"]["google_llm"]
GOOGLE_EMBEDDING = _yaml_config["models"]["google_embedding"]
OLLAMA_LLM = _yaml_config["models"]["ollama_llm"]
OLLAMA_EMBEDDING = _yaml_config["models"]["ollama_embedding"]
LLM_TEMPERATURE = _yaml_config["models"].get(
    "temperature", 0.0
)  # Default to 0.0 if not specified

PARSER_STRATEGY = _yaml_config["ingestion"]["parser_strategy"].lower()
SPLITTER_TYPE = _yaml_config["ingestion"]["splitter_type"].lower()
CHUNK_SIZE = _yaml_config["ingestion"]["chunk_size"]
CHUNK_OVERLAP = _yaml_config["ingestion"]["chunk_overlap"]

SEARCH_TYPE = _yaml_config["retrieval"].get("search_type", "mmr").lower()
FETCH_K = _yaml_config["retrieval"].get("fetch_k", 10)
BASE_TOP_K = _yaml_config["retrieval"]["base_k"]
RERANK_TOP_N = _yaml_config["retrieval"]["rerank_top_n"]
RERANK_MODEL = _yaml_config["retrieval"]["rerank_model"]

# 1. Get the absolute path of the directory containing config.py (src/)
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))

# 2. Go up one level to the main project root directory
BASE_DIR = os.path.dirname(_SRC_DIR)

# 3. Explicitly define the absolute path to your data/ directory
DATA_DIR = os.path.join(BASE_DIR, _yaml_config["vectorstore"]["data_dir"])

# Ensure the data directories physically exist on the machine automatically
os.makedirs(DATA_DIR, exist_ok=True)

golden_dataset_path = _yaml_config["evaluation"]["golden_ragas_dataset_path"]
GOLDEN_DATASET_PATH = os.path.join(BASE_DIR, golden_dataset_path)
