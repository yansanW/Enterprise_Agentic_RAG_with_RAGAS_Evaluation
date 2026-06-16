# src/pipeline/chains.py
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import PydanticOutputParser
from langchain_cohere import CohereRerank  # Professional integration package
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_ollama import ChatOllama

from src import config
from src.pipeline.schemas import GuardedAnswerSchema
from src.factory import ModelFactory
from typing import Optional, Any


def _get_llm_client():
    """Internal Factory helper to instantiate the exact model provider chosen in configs."""
    return ModelFactory.get_llm()
    

class AgenticRAGCore:
    def __init__(self, vectorstore):
        self.vectorstore = vectorstore

        # 1. Base Retrieval Phase: Pull a wide pool of candidates
        base_retriever = vectorstore.as_retriever(
            search_type=config.SEARCH_TYPE,
            search_kwargs={"k": config.BASE_TOP_K, "fetch_k": config.FETCH_K},
        )

        # 2. Rerank Phase: Core cross-encoder model setup
        compressor = CohereRerank(
            model=config.RERANK_MODEL,
            cohere_api_key=config.COHERE_API_KEY,
            top_n=config.RERANK_TOP_N,  # Truncate down to top 3 elite matches
        )

        # 3. Layer the compression router onto the base retriever stream
        self.retriever = ContextualCompressionRetriever(
            base_compressor=compressor, base_retriever=base_retriever
        )

        # Instantiate base LLM client with strict deterministic controls
        self.llm = _get_llm_client()

        # Explicit parser interface for logging and debugging
        self.output_parser = PydanticOutputParser(pydantic_object=GuardedAnswerSchema)
        self.structured_generator = self.llm.with_structured_output(GuardedAnswerSchema)

    async def aroute_query(self, query: str) -> str:
        """Asynchronously routes the query to optimize multi-user connection queues."""
        router_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are an elite triage agent for an enterprise knowledge graph. "
                    "Analyze the user input string carefully.\n\n"
                    "CRITERIA:\n"
                    "- Reply with 'RETRIEVE' if the user asks for data facts, metrics, documents, or technical analysis.\n"
                    "- Reply with 'CHAT' if the user is greeting you, asking casual questions, or making small talk.\n\n"
                    "Respond with exactly one word: either 'RETRIEVE' or 'CHAT'. Do not include punctuation.",
                ),
                ("human", "{query}"),
            ]
        )

        decision_chain = router_prompt | self.llm
        # Using .ainvoke() processes this concurrently without blocking your server loop!
        response = await decision_chain.ainvoke({"query": query})
        decision = response.content.strip().upper()
        return "RETRIEVE" if "RETRIEVE" in decision else "CHAT"

    async def aexecute_pipeline(
        self, query: str, chat_history: Optional[list[Any]] = None
    ) -> GuardedAnswerSchema:
        """
        Asynchronous Core Pipeline. Cohesive routing, conversational query rewriting,
        context parsing, and structured generation guardrails.
        """
        if chat_history is None:
            chat_history = []

        route = await self.aroute_query(query)
        print(f"🔀 Cognitive Router selected execution track: {route}")

        if route == "CHAT":
            return GuardedAnswerSchema(
                answer="Hello! I am your enterprise agentic intelligence core. How can I assist your research today?",
                is_supported_by_context=True,
                citations=[],
            )

        # --- NEW: CONVERSATIONAL QUERY REWRITING NODE ---
        search_query = query
        if len(chat_history) > 0:
            print(
                "🔄 Past conversation history detected. Executing Query Rewrite Node..."
            )
            rewrite_prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        "You are an elite query reformulation assistant. "
                        "Analyze the conversation history and the latest user follow-up question. "
                        "If the latest question contains pronouns (it, they, their, she, he) or references past topics, "
                        "rewrite it into a standalone, fully detailed search query optimized for vector database lookups. "
                        "Do not answer the question. Respond with ONLY the rewritten search query text string.",
                    ),
                    MessagesPlaceholder(variable_name="history"),
                    ("human", "{query}"),
                ]
            )

            rewrite_chain = rewrite_prompt | self.llm
            rewrite_response = await rewrite_chain.ainvoke(
                {"history": chat_history, "query": query}
            )
            search_query = rewrite_response.content.strip()
            print(
                f"🔍 Original query '{query}' successfully rewritten to: '{search_query}'"
            )

        # --- ASYNCHRONOUS RETRIEVAL TRACK ---
        # Pass the optimized search_query to the retriever instead of the raw user input!
        docs = await self.retriever.ainvoke(search_query)
        context_str = "\n\n".join(
            [
                f"[Source: {doc.metadata.get('source', 'Unknown')} | Page: {doc.metadata.get('page', 'N/A')}]\n{doc.page_content}"
                for doc in docs
            ]
        )

        # Professional prompt template engineering with strict behavioral rules
        qa_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a sovereign enterprise AI assistant bound to a strict knowledge contract.\n\n"
                    "INSTRUCTIONS:\n"
                    "1. Build your answer using ONLY the verified context fragments provided below.\n"
                    "2. If the context does not contain sufficient facts to answer the question, set 'is_supported_by_context' to false and output exactly: 'Information not found within verified knowledge base.'\n"
                    "3. Do not assume, extrapolate, or hallucinate metrics.\n"
                    "4. Populate the 'citations' array with exact string snippets used from the context.\n\n"
                    "VERIFIED REPOSITORY CONTEXT:\n{context}",
                ),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{query}"),
            ]
        )

        # Construct message payload matrix
        formatted_messages = qa_prompt.format_messages(
            context=context_str, history=chat_history, query=query
        )

        # Invoke the structured generation decoder asynchronously
        structured_response = await self.structured_generator.ainvoke(
            formatted_messages
        )
        return structured_response
