# src/pipeline/chains.py
'''
the pure engine room of your RAG. 
It has no awareness of Streamlit or FastAPI; it just takes input parameters and builds the LangChain components.
'''

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_cohere import CohereRerank
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from src import config

# Intercept string-like token objects coming from the parser to protect the embedding model
class SafeGoogleEmbeddings(GoogleGenerativeAIEmbeddings):
    def embed_query(self, text: str) -> list[float]:
        return super().embed_query(str(text))

def build_agentic_rag_chain(vectorstore):
    """Assembles the entire history-aware retrieval and rerank pipeline."""
    
    # 1. Base retriever pulls a broad selection (e.g., k=10)
    base_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
    
    # 2. Reranker narrows down to top 3 semantic matches
    # reranker = CohereRerank(top_n=3, model="rerank-v3.5", cohere_api_key=config.COHERE_API_KEY)
    # No hardcoding, completely dynamic!
    reranker = CohereRerank(
        top_n=config.RERANK_TOP_N, 
        model="rerank-v3.5", 
        cohere_api_key=config.COHERE_API_KEY
    )
    
    # 3. Compression pipeline combines them
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=reranker, 
        base_retriever=base_retriever
    )
    
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=config.GOOGLE_API_KEY)
    
    # ── Contextualization prompt (Query rewriter) ──
    contextualize_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Given the chat history and the latest user question, "
         "rewrite the question to be fully self-contained string. "
         "Do NOT answer it, and do NOT include any introductory filler text. "
         "If it is already self-contained, return it unchanged."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    history_aware_retriever = create_history_aware_retriever(
        llm, compression_retriever, contextualize_prompt
    )
    
    # ── Answer extraction prompt ──
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Use the retrieved context below to answer the question. "
         "If you don't know, say so — don't make things up.\n\n"
         "{context}"),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    combine_docs_chain = create_stuff_documents_chain(llm, qa_prompt)
    
    # Complete multi-stage chain
    return create_retrieval_chain(history_aware_retriever, combine_docs_chain)