# tests/conftest.py
import os

# Global testing gatekeeper: Inject mock/dummy variables
# so tests stay isolated from active cloud billings or local network drops
os.environ["GOOGLE_API_KEY"] = "mock_testing_key_abcdefg"
os.environ["COHERE_API_KEY"] = "mock_cohere_key_123456"
os.environ["LOCAL_OLLAMA_URL"] = "http://mock-localhost:11434"
