"""Document processing: load, chunk, embed, store."""
import logging
import os
import hashlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from llama_index.core import Settings, SimpleDirectoryReader, Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.readers.file import PyMuPDFReader

from .config import Config
from .db import get_db

logger = logging.getLogger("rag.tasks")

_semantic_splitter = None


def sanitize_text(text: str) -> str:
    """Strip NUL bytes and control chars that break Postgres text columns."""
    if not text:
        return text
    text = text.replace('\x00', '')
    text = ''.join(char for char in text if char == '\n' or char == '\t' or char == '\r' or ord(char) >= 32)
    return text


def clean_pdf_artifacts(text: str) -> str:
    """Remove TOC leaders, standalone page numbers, repeated headers, etc."""
    import re

    if not text:
        return text

    text = re.sub(r'\.{3,}', ' ', text)
    text = re.sub(r'^[^\n]+\s*\.{2,}\s*\d{1,3}\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*\d{1,3}\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'\s+\d{1,3}\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*\d+\.\d+\s*$', '', text, flags=re.MULTILINE)
    # Join lines split mid-sentence
    text = re.sub(r'([a-zа-яё,])\n([a-zа-яё])', r'\1 \2', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)
    text = re.sub(r'^\s+$', '', text, flags=re.MULTILINE)
    lines = text.split('\n')
    seen_markers = set()
    cleaned_lines = []

    for line in lines:
        line_lower = line.lower().strip()
        if line_lower in seen_markers:
            continue
        if re.match(r'^(confidential|confidentially|конфиденциально)$', line_lower):
            if line_lower not in seen_markers:
                seen_markers.add(line_lower)
                cleaned_lines.append(line)
        elif len(line_lower) < 30 and line_lower:
            if line_lower not in seen_markers:
                seen_markers.add(line_lower)
                cleaned_lines.append(line)
        else:
            cleaned_lines.append(line)

    return '\n'.join(cleaned_lines).strip()


def merge_short_chunks(nodes, min_length):
    """Prepend short chunks into the next long-enough chunk."""
    result = []
    pending_prefix = ""
    for node in nodes:
        text = node.get_content().strip() if hasattr(node, 'get_content') else str(node).strip()
        if len(text) < min_length and text:  # short but non-empty
            pending_prefix += text + "\n"
        else:
            if pending_prefix and text:
                node.text = pending_prefix + text
                pending_prefix = ""
            result.append(node)
    # Trailing prefix -> append to last chunk
    if pending_prefix and result:
        last = result[-1]
        last.text = last.get_content() + "\n" + pending_prefix.strip()
    merged_count = len(nodes) - len(result)
    if merged_count > 0:
        logger.info("Merged %d short chunks (<%d chars) into adjacent chunks", merged_count, min_length)
    return result


def format_citation_text(text: str, max_length: int = 500) -> str:
    """Clean and truncate chunk text for citation preview."""
    import re

    if not text:
        return text

    text = clean_pdf_artifacts(text)
    text = re.sub(r'\n\n+', '\n\n', text)
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
    text = re.sub(r' +', ' ', text)
    if len(text) > max_length:
        truncated = text[:max_length]
        last_period = truncated.rfind('.')
        last_question = truncated.rfind('?')
        last_exclaim = truncated.rfind('!')
        break_point = max(last_period, last_question, last_exclaim)

        if break_point > max_length * 0.5:
            text = truncated[:break_point + 1]
        else:
            text = truncated.rsplit(' ', 1)[0] + '...'

    return text.strip()

# OCR dependencies (optional)
try:
    from pdf2image import convert_from_path
    import pytesseract
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
Settings.llm = OpenAI(model=Config.LLM_MODEL, api_key=Config.OPENAI_API_KEY)
Settings.embed_model = OpenAIEmbedding(
    model=Config.EMBEDDING_MODEL,
    api_key=Config.OPENAI_API_KEY,
    dimensions=Config.EMBEDDING_DIM,
)
Settings.chunk_size = Config.CHUNK_SIZE
Settings.chunk_overlap = Config.CHUNK_OVERLAP


def report_progress(job_id: int, pct: int) -> None:
    pct = max(0, min(100, pct))
    try:
        with get_db() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE jobs SET progress = %s WHERE id = %s",
                    (pct, job_id)
                )
            conn.commit()
    except Exception as e:
        logger.warning("Failed to report progress for job %d: %s", job_id, e)


def get_semantic_splitter():
    global _semantic_splitter
    if _semantic_splitter is None:
        try:
            from llama_index.core.node_parser import SemanticSplitterNodeParser
            _semantic_splitter = SemanticSplitterNodeParser(
                buffer_size=1,
                breakpoint_percentile_threshold=Config.SEMANTIC_BREAKPOINT_THRESHOLD,
                embed_model=Settings.embed_model
            )
        except ImportError as e:
            logger.warning("Semantic chunking unavailable: %s", e)
            return None
        except Exception as e:
            logger.warning("Failed to initialize semantic splitter: %s", e)
            return None
    return _semantic_splitter


def get_splitter(page_count=0):
    if Config.USE_SEMANTIC_CHUNKING:
        if page_count > Config.SEMANTIC_CHUNKING_PAGE_LIMIT:
            logger.info("Large doc (%d pages), using SentenceSplitter instead of semantic", page_count)
            return SentenceSplitter(
                chunk_size=Config.CHUNK_SIZE,
                chunk_overlap=Config.CHUNK_OVERLAP
            )
        semantic = get_semantic_splitter()
        if semantic is not None:
            return semantic
        logger.warning("Falling back to standard chunking")

    return SentenceSplitter(
        chunk_size=Config.CHUNK_SIZE,
        chunk_overlap=Config.CHUNK_OVERLAP
    )


def get_content_hash(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def get_cached_embedding(content_hash: str, conn) -> list | None:
    if not Config.USE_EMBEDDING_CACHE:
        return None
    try:
        with conn.cursor() as cur:
            cur.execute("SAVEPOINT cache_lookup")
            cur.execute(
                "SELECT embedding FROM embedding_cache WHERE content_hash = %s",
                (content_hash,)
            )
            row = cur.fetchone()
            cur.execute("RELEASE SAVEPOINT cache_lookup")
            if row and row[0]:
                emb = row[0]
                if isinstance(emb, str):
                    emb = [float(x) for x in emb.strip('[]').split(',')]
                return emb
    except Exception:
        with conn.cursor() as cur:
            cur.execute("ROLLBACK TO SAVEPOINT cache_lookup")
    return None


def store_cached_embedding(content_hash: str, embedding: list, conn):
    if not Config.USE_EMBEDDING_CACHE:
        return
    try:
        with conn.cursor() as cur:
            cur.execute("SAVEPOINT cache_insert")
            cur.execute(
                """INSERT INTO embedding_cache (content_hash, embedding, model)
                   VALUES (%s, %s, %s)
                   ON CONFLICT (content_hash) DO NOTHING""",
                (content_hash, embedding, Config.EMBEDDING_MODEL)
            )
            cur.execute("RELEASE SAVEPOINT cache_insert")
    except Exception:
        with conn.cursor() as cur:
            cur.execute("ROLLBACK TO SAVEPOINT cache_insert")
        pass


def ocr_pdf_to_text(pdf_path: str) -> list:
    """Run Tesseract OCR on an image-based PDF, return Document list."""
    if not OCR_AVAILABLE:
        raise RuntimeError("pdf2image and pytesseract are required for OCR")

    logger.info("PDF has no text layer, running OCR...")

    poppler_path = os.getenv("POPPLER_PATH")
    try:
        if poppler_path:
            images = convert_from_path(pdf_path, poppler_path=poppler_path)
        else:
            images = convert_from_path(pdf_path)
        logger.info("Converted PDF to %d images", len(images))
    except Exception as e:
        raise RuntimeError(f"Failed to convert PDF to images (is poppler installed?): {e}")
    documents = []
    for i, image in enumerate(images):
        text = pytesseract.image_to_string(image, lang=Config.OCR_LANGUAGES)

        if text.strip():
            doc = Document(
                text=text,
                metadata={
                    "page": i + 1,
                    "source": "ocr"
                }
            )
            documents.append(doc)
            logger.info("OCR page %d: %d chars", i + 1, len(text))
        else:
            logger.warning("OCR page %d: no text found", i + 1)

    return documents


# --- Multi-modal extraction ---

def extract_tables_from_pdf(pdf_path: str) -> list:
    try:
        import pdfplumber
    except ImportError:
        logger.warning("pdfplumber not installed, skipping table extraction")
        return []

    tables_found = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                page_tables = page.extract_tables()
                if not page_tables:
                    continue
                for tbl_idx, table in enumerate(page_tables):
                    if not table:
                        continue
                    md_lines = []
                    for row_idx, row in enumerate(table):
                        cells = [str(cell).strip() if cell is not None else "" for cell in row]
                        md_lines.append("| " + " | ".join(cells) + " |")
                        if row_idx == 0:
                            md_lines.append("| " + " | ".join(["---"] * len(cells)) + " |")
                    table_md = "\n".join(md_lines)
                    if table_md.strip():
                        tables_found.append({
                            "page": page_num,
                            "table_markdown": table_md,
                            "table_index": tbl_idx,
                        })
        if tables_found:
            logger.info("Extracted %d tables from PDF", len(tables_found))
        else:
            logger.info("No tables found in PDF")
    except Exception as e:
        logger.warning("Table extraction failed: %s", e)

    return tables_found


def describe_pdf_images(pdf_path: str, pages: list = None) -> list:
    """Extract images from PDF and describe them via vision API."""
    try:
        import fitz  # PyMuPDF
    except ImportError:
        logger.warning("PyMuPDF (fitz) not available - skipping image extraction")
        return []

    import base64

    descriptions = []
    try:
        doc = fitz.open(pdf_path)
        client = _get_context_client()  # reuse cached OpenAI client

        for page_idx in range(len(doc)):
            page_num = page_idx + 1
            if pages is not None and page_num not in pages:
                continue

            image_list = doc[page_idx].get_images(full=True)
            for img_idx, img_info in enumerate(image_list):
                xref = img_info[0]
                base_image = doc.extract_image(xref)
                if not base_image:
                    continue
                image_bytes = base_image["image"]

                if len(image_bytes) < 10_240:  # skip icons/logos
                    continue

                b64_image = base64.b64encode(image_bytes).decode("utf-8")
                mime = base_image.get("ext", "png")
                if mime == "jpg":
                    mime = "jpeg"
                data_uri = f"data:image/{mime};base64,{b64_image}"

                try:
                    response = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "text",
                                        "text": (
                                            "Describe this image from a document in detail. "
                                            "If it is a chart or graph, describe the data it "
                                            "shows, axes, trends, and key takeaways. "
                                            "If it is a diagram, describe its components and "
                                            "relationships. Keep the description concise but "
                                            "informative (max 200 words)."
                                        ),
                                    },
                                    {
                                        "type": "image_url",
                                        "image_url": {"url": data_uri},
                                    },
                                ],
                            }
                        ],
                        max_tokens=300,
                        temperature=0,
                    )
                    desc = response.choices[0].message.content.strip()
                    if desc:
                        descriptions.append({
                            "page": page_num,
                            "image_index": img_idx,
                            "description": desc,
                        })
                        logger.info("Described image on page %d (idx %d)", page_num, img_idx)
                except Exception as e:
                    logger.warning("Vision API call failed for page %d image %d: %s", page_num, img_idx, e)

        doc.close()

        if descriptions:
            logger.info("Described %d images from PDF", len(descriptions))
        else:
            logger.info("No significant images found in PDF")
    except Exception as e:
        logger.warning("Image extraction failed: %s", e)

    return descriptions


_context_client = None

def _get_context_client():
    global _context_client
    if _context_client is None:
        from openai import OpenAI as OpenAIClient
        _context_client = OpenAIClient(api_key=Config.OPENAI_API_KEY)
    return _context_client


def generate_chunk_context(chunk_text: str, document_text: str, filename: str) -> str:
    """Prepend a short LLM-generated context to a chunk for better retrieval."""
    if not Config.USE_CONTEXTUAL_CHUNKING:
        return chunk_text

    doc_preview = document_text[:6000]

    prompt = f"""<document>
{doc_preview}
</document>

Here is a chunk from the document "{filename}":
<chunk>
{chunk_text[:1500]}
</chunk>

Give a short succinct context (1-2 sentences) to situate this chunk within the overall document.
Focus on what section/topic this chunk covers and how it relates to the document's main subject.
Answer ONLY with the context, nothing else."""

    try:
        client = _get_context_client()
        response = client.chat.completions.create(
            model=Config.CONTEXT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150,
            temperature=0
        )
        context = response.choices[0].message.content.strip()
        return f"{context}\n\n{chunk_text}"
    except Exception as e:
        logger.warning("Contextual chunking failed for chunk: %s", e)
        return chunk_text


def generate_chunk_contexts_parallel(chunks, document_text, filename, max_workers=None):
    if not Config.USE_CONTEXTUAL_CHUNKING:
        for c in chunks:
            c["embedding_text"] = c["chunk_text"]
        return

    max_workers = max_workers or Config.CONTEXT_CONCURRENCY
    logger.info("Generating contextual prefixes for %d chunks (max_workers=%d)", len(chunks), max_workers)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(generate_chunk_context, c["chunk_text"], document_text, filename)
                   for c in chunks]
        for chunk, future in zip(chunks, futures):
            try:
                chunk["embedding_text"] = future.result()
            except Exception as e:
                logger.warning("Contextual chunking failed in parallel: %s", e)
                chunk["embedding_text"] = chunk["chunk_text"]


def process_document(document_id: int, job_id: int):
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT project_id, filename, file_path FROM documents WHERE id = %s",
                (document_id,)
            )
            row = cur.fetchone()
            if not row:
                raise ValueError(f"Document {document_id} not found")
            project_id, filename, file_path = row

    if not file_path or not os.path.exists(file_path):
        raise ValueError(f"File not found: {file_path}")

    report_progress(job_id, 5)

    if file_path.lower().endswith('.pdf'):
        reader = PyMuPDFReader()
        documents = reader.load(file_path=file_path)

        total_text = "".join(doc.text for doc in documents if doc.text)
        if not total_text.strip():
            if OCR_AVAILABLE:
                documents = ocr_pdf_to_text(file_path)
            else:
                raise ValueError("PDF has no text and OCR is not available")
    else:
        reader = SimpleDirectoryReader(input_files=[file_path])
        documents = reader.load_data()

    if not documents:
        raise ValueError("No content could be extracted from the file")

    report_progress(job_id, 10)

    is_pdf = file_path.lower().endswith('.pdf')

    table_documents = []
    if is_pdf and Config.EXTRACT_TABLES:
        logger.info("Extracting tables from PDF...")
        tables = extract_tables_from_pdf(file_path)
        for tbl in tables:
            table_doc = Document(
                text=tbl["table_markdown"],
                metadata={
                    "source": "table",
                    "page": tbl["page"],
                    "table_index": tbl["table_index"],
                    "content_type": "table",
                },
            )
            table_documents.append(table_doc)
        if table_documents:
            logger.info("Created %d table documents", len(table_documents))

    image_documents = []
    if is_pdf and Config.EXTRACT_IMAGES:
        logger.info("Extracting and describing images from PDF...")
        image_descs = describe_pdf_images(pdf_path=file_path)
        for desc in image_descs:
            image_doc = Document(
                text=desc["description"],
                metadata={
                    "source": "image",
                    "page": desc["page"],
                    "image_index": desc["image_index"],
                    "content_type": "image",
                },
            )
            image_documents.append(image_doc)
        if image_documents:
            logger.info("Created %d image-description documents", len(image_documents))

    for doc in documents:
        doc.metadata["filename"] = filename
        doc.metadata["doc_id"] = document_id
        if "content_type" not in doc.metadata:
            doc.metadata["content_type"] = "text"

    for extra_doc in table_documents + image_documents:
        extra_doc.metadata["filename"] = filename
        extra_doc.metadata["doc_id"] = document_id
    documents.extend(table_documents)
    documents.extend(image_documents)

    report_progress(job_id, 25)

    page_count = len(documents)
    splitter = get_splitter(page_count=page_count)
    chunking_type = "semantic" if (Config.USE_SEMANTIC_CHUNKING
                                   and page_count <= Config.SEMANTIC_CHUNKING_PAGE_LIMIT
                                   and get_semantic_splitter()) else "sentence"
    logger.info("Chunking with %s splitter (%d pages)...", chunking_type, page_count)
    nodes = splitter.get_nodes_from_documents(documents)

    if not nodes:
        raise ValueError("No text chunks could be created from the document")

    nodes = merge_short_chunks(nodes, Config.MIN_CHUNK_LENGTH)

    report_progress(job_id, 35)

    full_doc_text = "\n".join(doc.text for doc in documents if doc.text)

    logger.info("Pass 1: validating %d chunks", len(nodes))
    chunks = []
    for i, node in enumerate(nodes):
        if not node.text or not node.text.strip():
            logger.warning("Skipping empty chunk %d", i)
            continue

        chunk_text = sanitize_text(node.text)
        chunk_text = clean_pdf_artifacts(chunk_text)

        if not chunk_text.strip():
            logger.warning("Skipping chunk %d - empty after cleaning", i)
            continue

        stripped = chunk_text.strip()
        garbage_chars = sum(1 for c in stripped if c == '?' or (not c.isprintable() and c not in '\n\r\t'))
        if len(stripped) > 0 and garbage_chars / len(stripped) > 0.3:
            logger.warning("Skipping chunk %d - %.0f%% garbage characters", i, garbage_chars / len(stripped) * 100)
            continue

        page_num = None
        if hasattr(node, 'metadata') and node.metadata:
            page_num = node.metadata.get('page') or node.metadata.get('page_number') or node.metadata.get('page_label')
            if page_num is not None:
                try:
                    page_num = int(page_num)
                except (ValueError, TypeError):
                    page_num = None
        if page_num is None and hasattr(node, 'source_node') and node.source_node:
            source_meta = getattr(node.source_node, 'metadata', None)
            if source_meta:
                page_num = source_meta.get('page') or source_meta.get('page_number') or source_meta.get('page_label')
                if page_num is not None:
                    try:
                        page_num = int(page_num)
                    except (ValueError, TypeError):
                        page_num = None

        content_type = "text"
        if hasattr(node, 'metadata') and node.metadata:
            content_type = node.metadata.get('content_type', 'text')
        if content_type == "text" and hasattr(node, 'source_node') and node.source_node:
            source_meta = getattr(node.source_node, 'metadata', None)
            if source_meta:
                content_type = source_meta.get('content_type', 'text')

        chunks.append({
            "chunk_index": i,
            "chunk_text": chunk_text,
            "embedding_text": None,  # filled in Pass 2
            "embedding": None,       # filled in Pass 3/4
            "page_num": page_num,
            "content_type": content_type,
        })

    if not chunks:
        raise ValueError("No valid text chunks could be created from the document")

    logger.info("Pass 1: %d valid chunks from %d nodes", len(chunks), len(nodes))
    report_progress(job_id, 40)

    logger.info("Pass 2: contextual chunking (%d chunks)", len(chunks))
    generate_chunk_contexts_parallel(chunks, full_doc_text, filename)
    report_progress(job_id, 55)

    logger.info("Pass 3: cache lookup (%d chunks)", len(chunks))
    cache_hits = 0
    cache_misses = []
    with get_db() as conn:
        for c in chunks:
            content_hash = get_content_hash(c["embedding_text"])
            c["content_hash"] = content_hash
            cached = get_cached_embedding(content_hash, conn)
            if cached is not None:
                c["embedding"] = cached
                cache_hits += 1
            else:
                cache_misses.append(c)

    logger.info("Pass 3: %d hits, %d misses", cache_hits, len(cache_misses))
    report_progress(job_id, 60)

    if cache_misses:
        logger.info("Pass 4: embedding %d chunks", len(cache_misses))
        miss_texts = [c["embedding_text"] for c in cache_misses]
        try:
            batch_embeddings = Settings.embed_model.get_text_embedding_batch(miss_texts, show_progress=True)
        except Exception as e:
            logger.warning("Batch embedding failed, falling back to individual: %s", e)
            batch_embeddings = []
            for text in miss_texts:
                try:
                    batch_embeddings.append(Settings.embed_model.get_text_embedding(text))
                except Exception as e2:
                    logger.warning("Individual embedding failed: %s", e2)
                    batch_embeddings.append(None)

        with get_db() as conn:
            for c, emb in zip(cache_misses, batch_embeddings):
                if emb is not None:
                    c["embedding"] = emb
                    store_cached_embedding(c["content_hash"], emb, conn)
                else:
                    logger.warning("Chunk %d has no embedding, will be skipped", c["chunk_index"])
            conn.commit()

        logger.info("Pass 4: embedded %d chunks", sum(1 for e in batch_embeddings if e is not None))
    else:
        logger.info("Pass 4: all from cache")

    report_progress(job_id, 85)

    logger.info("Pass 5: inserting into DB")
    valid_chunk_count = 0
    with get_db() as conn:
        with conn.cursor() as cur:
            for c in chunks:
                if c["embedding"] is None:
                    logger.warning("Skipping chunk %d - no embedding available", c["chunk_index"])
                    continue

                cur.execute(
                    """INSERT INTO chunks (document_id, project_id, content, embedding, chunk_index, page_number, content_type)
                       VALUES (%s, %s, %s, %s, %s, %s, %s)
                       ON CONFLICT (document_id, chunk_index) DO NOTHING""",
                    (document_id, project_id, c["chunk_text"], c["embedding"],
                     c["chunk_index"], c["page_num"], c["content_type"])
                )
                valid_chunk_count += 1

            if valid_chunk_count == 0:
                raise ValueError("No valid text chunks could be embedded from the document")

            cur.execute(
                "UPDATE documents SET status = 'ready', chunk_count = %s WHERE id = %s",
                (valid_chunk_count, document_id)
            )

            cur.execute(
                """UPDATE jobs SET status = 'completed', progress = 100, completed_at = CURRENT_TIMESTAMP
                   WHERE id = %s""",
                (job_id,)
            )
        conn.commit()

    report_progress(job_id, 100)
    logger.info("Processed document %d: %d/%d chunks embedded (cache hits: %d)",
                document_id, valid_chunk_count, len(chunks), cache_hits)
