# src/api/main.py
from fastapi import FastAPI
from src.database import initialize_vectorstore
from src.pipeline.chains import AgenticRAGCore

app = FastAPI(
    title="Enterprise Agentic RAG Platform",
    description="Headless production API gateway running dynamic routing and structured Pydantic guardrails.",
    version="1.0.0"
)

# Initialize global application components on server boot
try:
    vectorstore = initialize_vectorstore()
    agent_system = AgenticRAGCore(vectorstore=vectorstore)
except Exception as e:
    print(f"⚠️ Warning during database startup hook: {e}")
    agent_system = None

@app.get("/health")
async def health_check():
    """
    Sub-second service health probe. Used by container orchestrators 
    (like Docker/Kubernetes) to verify system availability.
    """
    return {
        "status": "healthy", 
        "database_connected": vectorstore is not None,
        "pipeline_active": agent_system is not None
    }