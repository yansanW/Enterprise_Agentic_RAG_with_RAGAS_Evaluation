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


from pydantic import BaseModel

class QueryRequest(BaseModel):
    question: str
    session_id: str = "default_session"


from fastapi import HTTPException
from src.database import get_session_history
from src.pipeline.schemas import GuardedAnswerSchema

@app.post("/api/v1/query", response_model=GuardedAnswerSchema)
async def process_agent_query(payload: QueryRequest):
    """
    Primary API gateway endpoint. Orchestrates stateful memory lookups,
    contextual query rewriting, and structured agentic execution loops.
    """
    if agent_system is None:
        raise HTTPException(
            status_code=503, 
            detail="AI pipeline core components failed to initialize properly."
        )
    
    try:
        # 1. Fetch persistent history tracking matrices from SQLite
        session_history = get_session_history(payload.session_id)
        past_messages = session_history.messages
        
        # 2. Execute the cognitive routing and query rewriting chain
        structured_response = await agent_system.aexecute_pipeline(
            query=payload.question, 
            chat_history=past_messages
        )
        
        # 3. Append the new conversational turn state back to the SQL disk table
        session_history.add_user_message(payload.question)
        session_history.add_ai_message(structured_response.answer)
        
        return structured_response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Pipeline Error: {str(e)}")