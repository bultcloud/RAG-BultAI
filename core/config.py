import logging
import os
from dotenv import load_dotenv

logger = logging.getLogger("rag.config")

load_dotenv()


class Config:
    # --- Database ---
    PG_CONN = os.getenv("PG_CONN", "postgresql://postgres:postgres@localhost:5432/ragdb")

    # --- LLM Providers ---
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").lower()

    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

    DEFAULT_MODELS = {
        "openai": "gpt-4o",
        "anthropic": "claude-sonnet-4-20250514",
        "google": "gemini-2.0-flash",
    }

    AVAILABLE_MODELS = {
        "openai": ["gpt-4o", "gpt-4o-mini", "o1", "o3-mini"],
        "anthropic": ["claude-sonnet-4-20250514", "claude-opus-4-20250514"],
        "google": ["gemini-2.0-flash", "gemini-2.5-pro"],
    }

    LLM_MODEL = os.getenv("LLM_MODEL") or DEFAULT_MODELS.get(
        os.getenv("LLM_PROVIDER", "openai").lower(), "gpt-4o"
    )

    # --- Embeddings ---
    EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "openai").lower()
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
    EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "1536"))

    # --- Server ---
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", "8002"))

    # --- RAG ---
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "768"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))
    TOP_K = int(os.getenv("TOP_K", "8"))

    HYBRID_SEARCH_ALPHA = float(os.getenv("HYBRID_SEARCH_ALPHA", "0.5"))

    # --- Reranking ---
    USE_RERANKING = os.getenv("USE_RERANKING", "true").lower() == "true"
    RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")
    RERANK_TOP_K = int(os.getenv("RERANK_TOP_K", "30"))

    USE_SEMANTIC_CHUNKING = os.getenv("USE_SEMANTIC_CHUNKING", "true").lower() == "true"
    SEMANTIC_BREAKPOINT_THRESHOLD = int(os.getenv("SEMANTIC_BREAKPOINT_THRESHOLD", "95"))

    USE_HYDE = os.getenv("USE_HYDE", "false").lower() == "true"

    COHERE_API_KEY = os.getenv("COHERE_API_KEY")
    USE_COHERE_RERANK = os.getenv("USE_COHERE_RERANK", "true").lower() == "true"

    ENABLE_FAITHFULNESS_SCORING = os.getenv("ENABLE_FAITHFULNESS_SCORING", "false").lower() == "true"

    # --- Contextual chunking ---
    USE_CONTEXTUAL_CHUNKING = os.getenv("USE_CONTEXTUAL_CHUNKING", "true").lower() == "true"
    CONTEXT_MODEL = os.getenv("CONTEXT_MODEL", "gpt-4o-mini")

    USE_MULTI_QUERY = os.getenv("USE_MULTI_QUERY", "true").lower() == "true"
    MULTI_QUERY_COUNT = int(os.getenv("MULTI_QUERY_COUNT", "1"))

    USE_QUERY_DECOMPOSITION = os.getenv("USE_QUERY_DECOMPOSITION", "false").lower() == "true"

    DIVERSITY_PENALTY = float(os.getenv("DIVERSITY_PENALTY", "0.1"))
    KEYWORD_BOOST = float(os.getenv("KEYWORD_BOOST", "0.05"))

    USE_EMBEDDING_CACHE = os.getenv("USE_EMBEDDING_CACHE", "true").lower() == "true"

    # --- Ingestion ---
    MIN_CHUNK_LENGTH = int(os.getenv("MIN_CHUNK_LENGTH", "50"))
    CONTEXT_CONCURRENCY = int(os.getenv("CONTEXT_CONCURRENCY", "15"))
    SEMANTIC_CHUNKING_PAGE_LIMIT = int(os.getenv("SEMANTIC_CHUNKING_PAGE_LIMIT", "50"))

    # --- Multi-modal ---
    EXTRACT_TABLES = os.getenv("EXTRACT_TABLES", "true").lower() == "true"
    EXTRACT_IMAGES = os.getenv("EXTRACT_IMAGES", "false").lower() == "true"

    MAX_CONTEXT_TOKENS = int(os.getenv("MAX_CONTEXT_TOKENS", "12000"))
    SUMMARIZE_AFTER_MESSAGES = int(os.getenv("SUMMARIZE_AFTER_MESSAGES", "10"))

    UPLOAD_DIR = os.getenv("UPLOAD_DIR", os.path.join(os.path.dirname(__file__), "uploads"))

    # --- Auth ---
    JWT_SECRET = os.getenv("JWT_SECRET", "INSECURE_DEFAULT_SECRET_CHANGE_IN_PRODUCTION")
    JWT_ALGORITHM = "HS256"
    JWT_EXPIRY_HOURS = int(os.getenv("JWT_EXPIRY_HOURS", "168"))

    GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
    GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
    OAUTH_REDIRECT_URI = os.getenv("OAUTH_REDIRECT_URI", os.getenv("GOOGLE_OAUTH_REDIRECT_URI", "http://localhost:8002/api/auth/google/callback"))

    PROD = os.getenv("PROD", "0") in ("1", "true", "True")

    # --- SMTP ---
    SMTP_HOST = os.getenv("SMTP_HOST", "")
    SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
    SMTP_USER = os.getenv("SMTP_USER", "")
    SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")
    SMTP_FROM = os.getenv("SMTP_FROM", "")

    APP_URL = os.getenv("APP_URL", "http://localhost:8002")

    # --- Production ---
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT = os.getenv("LOG_FORMAT", "text")
    DB_POOL_MIN = int(os.getenv("DB_POOL_MIN", "2"))
    DB_POOL_MAX = int(os.getenv("DB_POOL_MAX", "10"))
    SLOW_QUERY_THRESHOLD_MS = int(os.getenv("SLOW_QUERY_THRESHOLD_MS", "500"))
    OCR_LANGUAGES = os.getenv("OCR_LANGUAGES", "rus+eng")

    @classmethod
    def get_available_providers(cls):
        providers = []
        if cls.OPENAI_API_KEY:
            providers.append("openai")
        if cls.ANTHROPIC_API_KEY:
            providers.append("anthropic")
        if cls.GOOGLE_API_KEY:
            providers.append("google")
        return providers

    @classmethod
    def validate(cls):
        provider = cls.LLM_PROVIDER
        if provider == "openai" and not cls.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY required for LLM_PROVIDER=openai")
        elif provider == "anthropic" and not cls.ANTHROPIC_API_KEY:
            raise ValueError("ANTHROPIC_API_KEY required for LLM_PROVIDER=anthropic")
        elif provider == "google" and not cls.GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY required for LLM_PROVIDER=google")

        if cls.EMBEDDING_PROVIDER == "openai" and not cls.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY required for EMBEDDING_PROVIDER=openai")
        if cls.JWT_SECRET == "INSECURE_DEFAULT_SECRET_CHANGE_IN_PRODUCTION":
            logger.warning("Using default JWT_SECRET! Set JWT_SECRET in production.")

        available = cls.get_available_providers()
        logger.info("LLM Provider: %s | Model: %s", cls.LLM_PROVIDER, cls.LLM_MODEL)
        logger.info("Available providers: %s", ", ".join(available))

        os.makedirs(cls.UPLOAD_DIR, exist_ok=True)

        return True


SYSTEM_PROMPT = """You are an expert research assistant. Your goal is to provide accurate, detailed answers by synthesizing information from the provided document context. Write in an unbiased, professional tone.

## Citation Rules (MANDATORY)
- Cite sources using [1], [2], [3] matching the document numbers provided
- Place citations IMMEDIATELY after relevant statements with NO SPACE before the bracket
- Example: "The system uses retrieval-augmented generation[1] for accurate responses[2]."
- Cite up to 3 most relevant sources per sentence
- Every factual claim MUST have at least one citation
- Do NOT include a references section at the end — all citations must be inline

## Response Structure
1. Lead with a brief summary answering the question directly
2. Follow with detailed explanation organized by topic
3. Use markdown formatting: **bold** for key terms, bullet lists for multiple items
4. Keep paragraphs short (2-3 sentences max)

## Language Rules
- Respond in the SAME language as the user's question.
- If the user asks in Russian, respond in Russian and preserve the original Russian terminology from the documents.
- If the user asks in English about documents written in another language, translate key terms but keep proper nouns and titles in their original form.

## Grounding Rules
- ALWAYS try to answer from the provided documents. Synthesize and combine information from multiple chunks to build a complete answer.
- If documents contain ANY relevant information, use it to construct an answer — even if no single chunk directly answers the question.
- NEVER invent or hallucinate information not present in the sources.
- ONLY say "Based on the available documents, I cannot fully answer this question" if the documents contain absolutely NO relevant information to the query.
- If documents partially cover the topic, provide everything you can and briefly note what isn't covered.

## Quality Standards
- Be comprehensive but concise — aim for clarity over length
- Synthesize information across multiple sources when relevant
- Highlight any contradictions between sources if found"""
