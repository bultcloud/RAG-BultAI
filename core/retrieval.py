"""Hybrid search, reranking, HyDE, and faithfulness scoring."""
import asyncio
import logging
import os
import re
import math
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional
from dataclasses import dataclass, field

from llama_index.core import Settings
from llama_index.llms.openai import OpenAI

from .config import Config
from .db import get_db

logger = logging.getLogger("rag.retrieval")

_reranker = None
_cohere_client = None
_has_tsvector_cache = None


@dataclass
class RetrievalConfig:
    hybrid_alpha: float = field(default_factory=lambda: Config.HYBRID_SEARCH_ALPHA)
    initial_top_k: int = field(default_factory=lambda: Config.RERANK_TOP_K)
    final_top_k: int = field(default_factory=lambda: Config.TOP_K)
    use_reranking: bool = field(default_factory=lambda: Config.USE_RERANKING)
    use_hyde: bool = field(default_factory=lambda: Config.USE_HYDE)
    use_cohere: bool = field(default_factory=lambda: Config.USE_COHERE_RERANK)


def get_reranker():
    global _reranker
    if _reranker is None:
        try:
            from sentence_transformers import CrossEncoder
            logger.info("Loading reranker model: %s", Config.RERANKER_MODEL)
            _reranker = CrossEncoder(Config.RERANKER_MODEL, max_length=512)
            logger.info("Reranker loaded: %s", Config.RERANKER_MODEL)
        except ImportError:
            logger.warning("sentence-transformers not installed, reranking disabled")
            return None
        except Exception as e:
            logger.warning("Failed to load reranker: %s", e)
            return None
    return _reranker


def get_cohere_client():
    global _cohere_client
    if _cohere_client is None and Config.COHERE_API_KEY:
        try:
            import cohere
            _cohere_client = cohere.Client(Config.COHERE_API_KEY)
            logger.info("Cohere client initialized")
        except ImportError:
            logger.warning("cohere package not installed")
            return None
        except Exception as e:
            logger.warning("Failed to initialize Cohere: %s", e)
            return None
    return _cohere_client


def preload_models():
    """Load reranker at startup to avoid cold-start on first query."""
    if Config.USE_RERANKING:
        get_reranker()
    if Config.USE_COHERE_RERANK and Config.COHERE_API_KEY:
        get_cohere_client()


def vector_retrieve(
    project_id: int,
    query_embedding: List[float],
    top_k: int = 20
) -> List[dict]:
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT c.id, c.content, c.chunk_index, c.document_id,
                       d.filename, 1 - (c.embedding <=> %s::vector) as similarity,
                       c.page_number
                FROM chunks c
                JOIN documents d ON d.id = c.document_id
                WHERE c.project_id = %s
                ORDER BY c.embedding <=> %s::vector
                LIMIT %s
            """, (query_embedding, project_id, query_embedding, top_k))
            rows = cur.fetchall()

    return [
        {
            "chunk_id": r[0],
            "content": r[1],
            "chunk_index": r[2],
            "document_id": r[3],
            "filename": r[4],
            "vector_score": float(r[5]),
            "keyword_score": 0.0,
            "combined_score": float(r[5]),
            "page_number": r[6]
        }
        for r in rows
    ]


def hybrid_retrieve(
    project_id: int,
    query: str,
    query_embedding: List[float],
    alpha: float = 0.5,
    top_k: int = 20
) -> List[dict]:
    global _has_tsvector_cache
    with get_db() as conn:
        with conn.cursor() as cur:
            if _has_tsvector_cache is None:
                cur.execute("""
                    SELECT EXISTS (
                        SELECT 1 FROM information_schema.columns
                        WHERE table_name = 'chunks' AND column_name = 'content_tsv'
                    )
                """)
                _has_tsvector_cache = cur.fetchone()[0]

            if not _has_tsvector_cache:
                return vector_retrieve(project_id, query_embedding, top_k)

            # 'simple' text search config for multilingual support
            cur.execute("""
                WITH vector_search AS (
                    SELECT id, 1 - (embedding <=> %s::vector) as vscore
                    FROM chunks
                    WHERE project_id = %s
                ),
                keyword_search AS (
                    SELECT id,
                           ts_rank_cd(content_tsv, plainto_tsquery('simple', %s), 32) as kscore
                    FROM chunks
                    WHERE project_id = %s
                      AND content_tsv @@ plainto_tsquery('simple', %s)
                ),
                -- Normalize scores to 0-1 range
                vector_normalized AS (
                    SELECT id, vscore,
                           (vscore - MIN(vscore) OVER()) /
                           NULLIF(MAX(vscore) OVER() - MIN(vscore) OVER(), 0) as vscore_norm
                    FROM vector_search
                ),
                keyword_normalized AS (
                    SELECT id, kscore,
                           (kscore - MIN(kscore) OVER()) /
                           NULLIF(MAX(kscore) OVER() - MIN(kscore) OVER(), 0) as kscore_norm
                    FROM keyword_search
                )
                SELECT c.id, c.content, c.chunk_index, c.document_id, d.filename,
                       COALESCE(v.vscore, 0) as vector_score,
                       COALESCE(k.kscore, 0) as keyword_score,
                       (COALESCE(v.vscore_norm, 0) * %s +
                        COALESCE(k.kscore_norm, 0) * %s) as combined_score,
                       c.page_number
                FROM chunks c
                LEFT JOIN vector_normalized v ON v.id = c.id
                LEFT JOIN keyword_normalized k ON k.id = c.id
                JOIN documents d ON d.id = c.document_id
                WHERE c.project_id = %s
                  AND (v.vscore IS NOT NULL OR k.kscore IS NOT NULL)
                ORDER BY combined_score DESC
                LIMIT %s
            """, (
                query_embedding, project_id,
                query, project_id, query,
                alpha, 1 - alpha,
                project_id, top_k
            ))
            rows = cur.fetchall()

    return [
        {
            "chunk_id": r[0],
            "content": r[1],
            "chunk_index": r[2],
            "document_id": r[3],
            "filename": r[4],
            "vector_score": float(r[5]) if r[5] else 0.0,
            "keyword_score": float(r[6]) if r[6] else 0.0,
            "combined_score": float(r[7]) if r[7] else 0.0,
            "page_number": r[8]
        }
        for r in rows
    ]


def normalize_scores_sigmoid(chunks: List[dict]) -> List[dict]:
    """Sigmoid normalization gives absolute relevance, not just ranking."""
    if not chunks:
        return chunks

    for chunk in chunks:
        raw = chunk.get("rerank_score", 0)
        chunk["rerank_score"] = 1.0 / (1.0 + math.exp(-raw))

    return chunks


def _apply_rerank_postprocessing(
    query: str,
    chunks: List[dict],
    diversity_penalty: float = None,
    keyword_boost: float = None,
) -> List[dict]:
    """Diversity penalty + keyword boost on reranked results."""
    if not chunks:
        return chunks

    if diversity_penalty is None:
        diversity_penalty = Config.DIVERSITY_PENALTY
    if keyword_boost is None:
        keyword_boost = Config.KEYWORD_BOOST

    is_cyrillic = _detect_has_cyrillic(query)
    stop_words = _STOP_WORDS_RU | _STOP_WORDS_EN if is_cyrillic else _STOP_WORDS_EN
    query_keywords = {
        w for w in query.lower().split()
        if w not in stop_words and len(w) > 2
    }

    if query_keywords:
        for chunk in chunks:
            content_lower = chunk.get("content", "").lower()
            matched = sum(1 for kw in query_keywords if kw in content_lower)
            if matched > 0:
                boost = keyword_boost * (matched / len(query_keywords))
                chunk["rerank_score"] = chunk.get("rerank_score", 0) + boost

    seen_sections: set = set()
    chunks.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)

    for chunk in chunks:
        doc_id = chunk.get("document_id")
        c_idx = chunk.get("chunk_index")
        if doc_id is not None and c_idx is not None:
            neighbours = {
                (doc_id, c_idx - 1),
                (doc_id, c_idx),
                (doc_id, c_idx + 1),
            }
            if neighbours & seen_sections:
                chunk["rerank_score"] = max(
                    0.0, chunk.get("rerank_score", 0) - diversity_penalty
                )
            seen_sections.add((doc_id, c_idx))

    chunks.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)
    return chunks


def rerank_with_cross_encoder(
    query: str,
    chunks: List[dict],
    top_k: int = 5
) -> List[dict]:
    if not chunks:
        return chunks

    reranker = get_reranker()
    if reranker is None:
        return chunks[:top_k]

    pairs = [[query, chunk["content"]] for chunk in chunks]
    try:
        scores = reranker.predict(pairs)
    except Exception as e:
        logger.warning("Reranking failed: %s", e)
        return chunks[:top_k]

    for chunk, score in zip(chunks, scores):
        chunk["rerank_score"] = float(score)

    reranked = sorted(chunks, key=lambda x: x.get("rerank_score", 0), reverse=True)
    normalize_scores_sigmoid(reranked)
    reranked = _apply_rerank_postprocessing(query, reranked)
    return reranked[:top_k]


def rerank_with_cohere(
    query: str,
    chunks: List[dict],
    top_k: int = 5
) -> List[dict]:
    if not chunks:
        return chunks

    client = get_cohere_client()
    if client is None:
        return rerank_with_cross_encoder(query, chunks, top_k)

    docs = [chunk["content"] for chunk in chunks]

    try:
        cohere_top_n = min(len(docs), max(top_k * 2, top_k + 10))
        response = client.rerank(
            query=query,
            documents=docs,
            top_n=cohere_top_n,
            model="rerank-v3.5"
        )

        reranked = []
        for result in response.results:
            chunk = chunks[result.index].copy()
            chunk["rerank_score"] = result.relevance_score
            reranked.append(chunk)

        reranked = _apply_rerank_postprocessing(query, reranked)
        return reranked[:top_k]

    except Exception as e:
        logger.warning("Cohere reranking failed, falling back to local: %s", e)
        return rerank_with_cross_encoder(query, chunks, top_k)


async def hyde_transform(query: str) -> str:
    llm = OpenAI(model=Config.LLM_MODEL, api_key=Config.OPENAI_API_KEY)

    prompt = f"""Write a short, detailed paragraph that would be a perfect answer
to this question. Be specific and factual. Do not say "I don't know" or
ask clarifying questions. Just provide the answer as if you know it.

Question: {query}

Answer:"""

    try:
        response = await llm.acomplete(prompt)
        return response.text.strip()
    except Exception as e:
        logger.warning("HyDE transform failed: %s", e)
        return query


async def generate_query_variations(query: str, count: int = 3) -> List[str]:
    if not Config.USE_MULTI_QUERY:
        return [query]

    llm = OpenAI(model=Config.LLM_MODEL, api_key=Config.OPENAI_API_KEY)

    prompt = f"""Generate {count} alternative phrasings of this search query.
Each variation should approach the topic from a different angle or use different keywords.
Keep each variation concise (under 30 words).

Original query: {query}

Return ONLY the variations, one per line, numbered 1-{count}. Do not include the original query."""

    try:
        response = await llm.acomplete(prompt)
        lines = [l.strip() for l in response.text.strip().split('\n') if l.strip()]
        variations = []
        for line in lines[:count]:
            cleaned = re.sub(r'^\d+[\.\)\-\:]\s*', '', line).strip()
            if cleaned:
                variations.append(cleaned)

        return [query] + variations
    except Exception as e:
        logger.warning("Multi-query generation failed: %s", e)
        return [query]


def is_complex_query(query: str) -> bool:
    """Heuristic: does this query need decomposition?"""
    q_lower = query.lower().strip()

    comparison_keywords = [
        "compare", "comparison", "contrast", "difference between",
        "differences between", "how does .* differ", "vs", "versus",
        "similarities between", "similar to",
        "сравни", "сравнение", "разница между", "различия между",
        "отличия между", "чем отличается",
    ]
    for kw in comparison_keywords:
        if re.search(kw, q_lower):
            return True

    multi_part_patterns = [
        r'\b(and|и)\b.+\?',            # "X and Y?" style
        r'\b(both|оба|обе|оба)\b',     # "both A and B"
        r'\b(as well as|а также)\b',
        r'\b(in addition|кроме того)\b',
    ]
    for pat in multi_part_patterns:
        if re.search(pat, q_lower):
            return True

    if query.count('?') >= 2:
        return True

    if re.search(r'\b(first|second|third|firstly|secondly|thirdly)\b', q_lower):
        return True
    if re.search(r'(\d+[\.\)]\s)', query):
        return True

    words = query.split()
    capitalized = [
        w for w in words[1:]
        if w[0].isupper() and w.lower() not in _STOP_WORDS_EN and len(w) > 1
    ] if len(words) > 1 else []
    if len(capitalized) >= 3:
        return True

    return False


async def decompose_query(query: str) -> List[str]:
    llm = OpenAI(model=Config.LLM_MODEL, api_key=Config.OPENAI_API_KEY)

    prompt = f"""Break the following complex question into 2-4 simpler, independent sub-questions.
Each sub-question should target one specific piece of information needed to answer the original question.
Keep each sub-question concise and self-contained.

Original question: {query}

Return ONLY the sub-questions, one per line, numbered 1-4. Do not include explanations."""

    try:
        response = await llm.acomplete(prompt)
        lines = [l.strip() for l in response.text.strip().split('\n') if l.strip()]
        sub_queries = []
        for line in lines[:4]:
            cleaned = re.sub(r'^\d+[\.\)\-\:]\s*', '', line).strip()
            if cleaned:
                sub_queries.append(cleaned)

        if len(sub_queries) >= 2:
            logger.info("Decomposed query into %d sub-queries", len(sub_queries))
            return sub_queries
        return [query]
    except Exception as e:
        logger.warning("Query decomposition failed: %s", e)
        return [query]


def reciprocal_rank_fusion(result_lists: List[List[dict]], k: int = 60) -> List[dict]:
    fused_scores = {}

    for results in result_lists:
        for rank, chunk in enumerate(results):
            chunk_id = chunk["chunk_id"]
            rrf_score = 1.0 / (k + rank + 1)

            if chunk_id in fused_scores:
                fused_scores[chunk_id] = (
                    fused_scores[chunk_id][0] + rrf_score,
                    fused_scores[chunk_id][1]
                )
            else:
                fused_scores[chunk_id] = (rrf_score, chunk)

    sorted_results = sorted(fused_scores.values(), key=lambda x: x[0], reverse=True)

    merged = []
    for score, chunk in sorted_results:
        chunk_copy = chunk.copy()
        chunk_copy["combined_score"] = score
        merged.append(chunk_copy)

    return merged


def _detect_has_cyrillic(text: str) -> bool:
    return bool(re.search(r'[а-яА-ЯёЁ]', text))
_STOP_WORDS_EN = {
    'the', 'a', 'an', 'is', 'are', 'was', 'were', 'what', 'which',
    'who', 'how', 'when', 'where', 'why', 'do', 'does', 'did',
    'can', 'could', 'would', 'should', 'of', 'in', 'on', 'at',
    'to', 'for', 'with', 'by', 'from', 'as', 'into', 'about',
    'that', 'this', 'it', 'they', 'them', 'their', 'there', 'be',
    'been', 'being', 'have', 'has', 'had', 'will', 'not', 'but',
    'or', 'and', 'if', 'so', 'my', 'your', 'his', 'her', 'its'
}

_STOP_WORDS_RU = {
    'что', 'как', 'в', 'на', 'с', 'по', 'для', 'от', 'из', 'за', 'о',
    'об', 'до', 'у', 'к', 'и', 'а', 'но', 'или', 'не', 'ни', 'да',
    'это', 'то', 'он', 'она', 'оно', 'они', 'мы', 'вы', 'я', 'ты',
    'его', 'её', 'их', 'мой', 'наш', 'ваш', 'свой', 'этот', 'тот',
    'весь', 'все', 'вся', 'всё', 'быть', 'был', 'была', 'было', 'были',
    'есть', 'будет', 'бы', 'же', 'ли', 'вот', 'так', 'уже', 'тоже',
    'только', 'ещё', 'при', 'через', 'между', 'когда', 'где', 'кто',
    'какой', 'какая', 'какое', 'какие', 'чем', 'чего', 'кого',
    'который', 'которая', 'которое', 'которые', 'происходит'
}


def filter_citations(
    chunks: List[dict],
    query: str,
    min_score: float = 0.4,
    max_citations: int = 5
) -> List[dict]:
    if not chunks:
        return chunks

    is_cyrillic = _detect_has_cyrillic(query)
    stop_words = _STOP_WORDS_RU | _STOP_WORDS_EN if is_cyrillic else _STOP_WORDS_EN

    query_words = set(query.lower().split())
    query_keywords = query_words - stop_words

    MIN_CONTENT_LENGTH = 80

    filtered = []
    for chunk in chunks:
        score = chunk.get("rerank_score", chunk.get("similarity", 0))

        if score < min_score:
            continue

        content = chunk.get("content", "")

        if len(content.strip()) < MIN_CONTENT_LENGTH:
            continue

        content_lower = content.lower()
        filename_lower = chunk.get("filename", "").lower()

        # Stem-based matching (first 4 chars) to handle inflected forms
        has_keyword_match = False
        if query_keywords:
            for kw in query_keywords:
                stem = kw[:4] if len(kw) > 4 else kw
                if stem in content_lower or stem in filename_lower:
                    has_keyword_match = True
                    break
        else:
            has_keyword_match = True

        if not has_keyword_match:
            chunk["rerank_score"] = score * 0.8
            if chunk["rerank_score"] < min_score:
                continue

        filtered.append(chunk)

    filtered.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)
    return filtered[:max_citations]


async def advanced_retrieve(
    project_id: int,
    query: str,
    config: Optional[RetrievalConfig] = None
) -> List[dict]:
    if config is None:
        config = RetrievalConfig()

    search_text = query
    if config.use_hyde:
        search_text = await hyde_transform(query)

    sub_queries: List[str] = []
    if Config.USE_QUERY_DECOMPOSITION and is_complex_query(search_text):
        sub_queries = await decompose_query(search_text)
        if len(sub_queries) < 2:
            sub_queries = []

    all_variations = []

    queries_to_expand = sub_queries if sub_queries else [search_text]
    for base_query in queries_to_expand:
        query_variations = await generate_query_variations(base_query)
        all_variations.extend(query_variations)

    try:
        all_embeddings = Settings.embed_model.get_text_embedding_batch(all_variations)
    except Exception:
        all_embeddings = [Settings.embed_model.get_text_embedding(q) for q in all_variations]

    if len(all_variations) > 1:
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor(max_workers=min(len(all_variations), 4)) as executor:
            retrieve_tasks = [
                loop.run_in_executor(
                    executor,
                    hybrid_retrieve,
                    project_id, q_var, q_emb,
                    config.hybrid_alpha, config.initial_top_k
                )
                for q_var, q_emb in zip(all_variations, all_embeddings)
            ]
            all_result_lists = list(await asyncio.gather(*retrieve_tasks))
    else:
        all_result_lists = [
            hybrid_retrieve(
                project_id=project_id,
                query=all_variations[0],
                query_embedding=all_embeddings[0],
                alpha=config.hybrid_alpha,
                top_k=config.initial_top_k
            )
        ] if all_variations else []

    if len(all_result_lists) > 1:
        candidates = reciprocal_rank_fusion(all_result_lists)[:config.initial_top_k]
    else:
        candidates = all_result_lists[0] if all_result_lists else []

    if config.use_reranking and len(candidates) > config.final_top_k:
        if config.use_cohere and Config.COHERE_API_KEY:
            reranked = rerank_with_cohere(query, candidates, config.final_top_k)
        else:
            reranked = rerank_with_cross_encoder(query, candidates, config.final_top_k)
    else:
        reranked = candidates[:config.final_top_k]

    # Adaptive threshold: multilingual cross-encoders produce low absolute scores
    results = filter_citations(
        reranked,
        query=query,
        min_score=0.15,
        max_citations=config.final_top_k
    )

    if not results and candidates:
        fallback = sorted(
            candidates[:config.final_top_k],
            key=lambda x: x.get("vector_score", 0),
            reverse=True
        )[:5]
        for chunk in fallback:
            chunk["similarity"] = chunk.get("vector_score", 0)
        results = fallback
    else:
        for chunk in results:
            if "rerank_score" in chunk:
                chunk["similarity"] = chunk["rerank_score"]
            elif "combined_score" in chunk:
                chunk["similarity"] = chunk["combined_score"]
            else:
                chunk["similarity"] = chunk.get("vector_score", 0)

    return results


def retrieve_context(project_id: int, query: str, top_k: int = None) -> List[dict]:
    """Sync wrapper for advanced_retrieve."""
    if top_k is None:
        top_k = Config.TOP_K

    config = RetrievalConfig(final_top_k=top_k)

    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If we're already in an async context, create a new task
            with ThreadPoolExecutor() as executor:
                future = executor.submit(
                    asyncio.run,
                    advanced_retrieve(project_id, query, config)
                )
                return future.result()
        else:
            return loop.run_until_complete(
                advanced_retrieve(project_id, query, config)
            )
    except RuntimeError:
        return asyncio.run(advanced_retrieve(project_id, query, config))


async def calculate_faithfulness_score(
    response: str,
    context_chunks: List[dict],
    client  # OpenAI client
) -> dict:
    """LLM-as-judge faithfulness score (0-1)."""
    if not context_chunks:
        return {"score": 0.0, "level": "low", "reason": "No sources provided"}

    context_text = "\n\n---\n".join([c.get("content", "")[:1000] for c in context_chunks[:3]])

    prompt = f"""Evaluate if this response is faithful to the source documents.
A faithful response only contains information that can be verified from the sources.
Penalize responses that add information not found in sources or contradict them.

SOURCES:
{context_text}

RESPONSE TO EVALUATE:
{response[:2000]}

Rate faithfulness from 0 to 100 where:
- 90-100: Fully faithful, all claims supported by sources
- 70-89: Mostly faithful, minor unsupported details
- 50-69: Partially faithful, some claims unsupported
- 0-49: Unfaithful, significant hallucination

Respond in this exact format:
SCORE: [number]
REASON: [one sentence explanation]"""

    try:
        result = await client.chat.completions.create(
            model="gpt-4o-mini",  # Fast and cheap for evaluation
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100,
            temperature=0
        )

        text = result.choices[0].message.content.strip()

        score_match = re.search(r'SCORE:\s*(\d+)', text)
        reason_match = re.search(r'REASON:\s*(.+)', text, re.DOTALL)

        score = int(score_match.group(1)) / 100 if score_match else 0.5
        score = max(0.0, min(1.0, score))  # Clamp to 0-1
        reason = reason_match.group(1).strip() if reason_match else "Unable to evaluate"

        if score >= 0.8:
            level = "high"
        elif score >= 0.5:
            level = "medium"
        else:
            level = "low"

        return {"score": round(score, 2), "level": level, "reason": reason}

    except Exception as e:
        logger.warning("Faithfulness scoring failed: %s", e)
        return {"score": 0.5, "level": "medium", "reason": "Evaluation unavailable"}
