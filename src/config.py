# src/config.py
import os
import yaml
import ollama

from dotenv import load_dotenv

# 1. Load the secret keys from the .env file into the system environment
load_dotenv()

TEST_DOCUMENT_PATH = os.getenv("TEST_DOCUMENT_PATH")

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL")
OLLAMA_URL = os.getenv("OLLAMA_BASE_URL")

# 2. Test the connection to Ollama with the provided configuration
client = ollama.Client(host=OLLAMA_URL)
response = client.generate(model=OLLAMA_MODEL, prompt="Hello, Llama!")
print(response['response'])

# 3. Parse the structural hyperparameters from the YAML configuration
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "configs", "config.yaml")
with open(CONFIG_PATH, "r") as f:
    _yaml_config = yaml.safe_load(f)


# Expose them as clean, accessible Python variables
LLM_SOURCE = _yaml_config["provider"]["llm_source"].lower()
EMBEDDING_SOURCE = _yaml_config["provider"]["embedding_source"].lower()

if LLM_SOURCE not in ["google", "ollama"]:
    raise ValueError(f"Unsupported LLM source: {LLM_SOURCE}. Supported sources are 'google' and 'ollama'.")

if EMBEDDING_SOURCE not in ["google", "ollama"]:
    raise ValueError(f"Unsupported embedding source: {EMBEDDING_SOURCE}. Supported sources are 'google' and 'ollama'.")


GOOGLE_LLM = _yaml_config["models"]["google_llm"]
GOOGLE_EMBEDDING = _yaml_config["models"]["google_embedding"]
OLLAMA_LLM = _yaml_config["models"]["ollama_llm"]
OLLAMA_EMBEDDING = _yaml_config["models"]["ollama_embedding"]

PARSER_STRATEGY = _yaml_config["ingestion"]["parser_strategy"].lower()
SPLITTER_TYPE = _yaml_config["ingestion"]["splitter_type"].lower()
CHUNK_SIZE = _yaml_config["ingestion"]["chunk_size"]
CHUNK_OVERLAP = _yaml_config["ingestion"]["chunk_overlap"]

RERANK_TOP_N = _yaml_config["retrieval"]["rerank_top_n"]
MODEL_NAME = _yaml_config["generation"]["model_name"]