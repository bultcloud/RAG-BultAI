# RAG Template

A production-ready **Retrieval-Augmented Generation** chat application that lets you upload documents and ask questions about them. Built with FastAPI, LlamaIndex, PostgreSQL+pgvector, and a modern single-page UI.

Deploy on [bult.ai](https://bult.ai)

---

## Features

- **Multi-model LLM support** -- OpenAI (GPT-4o, GPT-4o-mini, o1, o3-mini), Anthropic (Claude Sonnet/Opus), Google (Gemini 2.0 Flash, 2.5 Pro)
- **Advanced RAG pipeline** -- hybrid search (vector + BM25), cross-encoder reranking, contextual chunking, multi-query retrieval
- **Optimized ingestion** -- batch embeddings, parallel contextual chunking, smart semantic/sentence splitting, table extraction from PDFs
- **Document support** -- PDF (with OCR), DOCX, PPTX, TXT, Markdown, code files
- **Streaming responses** -- real-time answers with inline citations
- **Faithfulness scoring** -- verify LLM responses are grounded in source documents
- **Authentication** -- JWT + optional Google OAuth
- **Analytics dashboard** -- usage metrics, cost tracking, query latency
- **Export conversations** -- Markdown, JSON, and PDF formats
- **Dark mode** -- clean single-page UI with theme toggle

---

## Supported LLM Providers

Configure via environment variables. All providers support streaming responses and inline citations.

| Provider | Models |
|----------|--------|
| **OpenAI** | gpt-4o, gpt-4o-mini, o1, o3-mini |
| **Anthropic** | claude-sonnet-4, claude-opus-4 |
| **Google** | gemini-2.0-flash, gemini-2.5-pro |

Set `LLM_PROVIDER` in `.env` to choose the default provider. Users can override per-conversation via the UI model selector.

---

## Deploy on bult.ai

[bult.ai](https://bult.ai) is a PaaS that deploys from GitHub with built-in database templates and Docker support.

### Prerequisites

- GitHub account
- OpenAI API key ([get one here](https://platform.openai.com/api-keys))
- [bult.ai](https://bult.ai) account

### Step 1: Fork this repository

Fork or clone this repo to your GitHub account.

### Step 2: Create the app service (GitHub)

1. On bult.ai, click **Create** > **GitHub**
2. Select your forked repository
3. Go to the **Git** tab and set build settings to **Dockerfile**
4. Set **Dockerfile Path** to `Dockerfile` and **Dockerfile Context** to `.`
5. Set the port to `8002`
6. Add environment variables from `.env.example`:

   | Variable | Value |
   |----------|-------|
   | `PG_CONN` | `postgresql://YOUR_USER:YOUR_PASSWORD@pgvector:5432/YOUR_DB` |
   | `OPENAI_API_KEY` | Your OpenAI API key |
   | `JWT_SECRET` | Random string (generate: `python -c "import secrets; print(secrets.token_urlsafe(32))"`) |

   **Note**: The `PG_CONN` hostname must match your pgvector service name (step 4).

### Step 3: Create the PostgreSQL database

1. Click **Create** > **Databases** > **Postgres**
2. Set environment variables:

   | Variable | Value |
   |----------|-------|
   | `POSTGRES_USER` | Your database username (e.g., `pgvector`) |
   | `POSTGRES_PASSWORD` | A strong password |
   | `POSTGRES_DB` | Your database name (e.g., `ragdb`) |

   These must match the values in `PG_CONN` from step 2.

### Step 4: Create the pgvector service (Docker)

1. Click **Create** > **Docker**
2. Docker image: `ankane/pgvector:latest`
3. Name this service to match the hostname in `PG_CONN` (e.g., `pgvector`)
4. Add a volume at `/var/lib/postgresql/data` for persistent storage
5. Add internal port `5432`
6. Deploy

### Step 5: Deploy and verify

1. All three services should show as running
2. Check app logs for database migrations and worker startup
3. Open the app's public URL, register a user, and start chatting

Google Sign-In

To enable Google OAuth:

1. Go to [Google Cloud Console](https://console.cloud.google.com/) and create a project
2. Navigate to **APIs & Services** > **OAuth Consent Screen** and configure it
3. Go to **Credentials** > **Create OAuth Client ID** > **Web Application**
4. Add your callback URL:
   ```
   https://<your-project>.<region>.bult.app/api/auth/google/callback
   ```
5. Add environment variables to your app service:

   | Variable | Value |
   |----------|-------|
   | `GOOGLE_CLIENT_ID` | Your OAuth client ID |
   | `GOOGLE_CLIENT_SECRET` | Your OAuth client secret |
   | `OAUTH_REDIRECT_URI` | Your callback URL |

6. Redeploy. The login page will show "Sign in with Google".

---

## Configuration

All settings are controlled via environment variables. Copy `.env.example` to `.env` and adjust as needed.

### Required

| Variable | Description |
|----------|-------------|
| `PG_CONN` | PostgreSQL connection string |
| `OPENAI_API_KEY` | OpenAI API key (used for embeddings and as default LLM) |
| `JWT_SECRET` | Random string for signing authentication tokens |

### LLM Provider

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `openai` | `openai`, `anthropic`, or `google` |
| `LLM_MODEL` | `gpt-4o-mini` | Model name (e.g., `gpt-4o`, `claude-sonnet-4-20250514`, `gemini-2.5-pro`) |
| `ANTHROPIC_API_KEY` | -- | Required if `LLM_PROVIDER=anthropic` |
| `GOOGLE_API_KEY` | -- | Required if `LLM_PROVIDER=google` |

### RAG Tuning

| Variable | Default | Description |
|----------|---------|-------------|
| `HYBRID_SEARCH_ALPHA` | `0.5` | 0 = keyword only, 1 = vector only |
| `USE_RERANKING` | `true` | Cross-encoder reranking for better precision |
| `USE_SEMANTIC_CHUNKING` | `true` | Chunk at topic boundaries instead of fixed token counts |
| `USE_CONTEXTUAL_CHUNKING` | `true` | Prepend document context to each chunk |
| `USE_MULTI_QUERY` | `true` | Generate query variations for broader recall |
| `USE_QUERY_DECOMPOSITION` | `false` | Break complex questions into sub-queries |
| `USE_HYDE` | `false` | Generate hypothetical answer for retrieval |
| `EXTRACT_TABLES` | `true` | Extract tables from PDFs via pdfplumber |
| `EXTRACT_IMAGES` | `false` | Extract and describe images via OpenAI vision API |

### Advanced

| Variable | Default | Description |
|----------|---------|-------------|
| `MIN_CHUNK_LENGTH` | `50` | Minimum chunk length (chars) |
| `CONTEXT_CONCURRENCY` | `15` | Parallel contextual chunking LLM calls |
| `SEMANTIC_CHUNKING_PAGE_LIMIT` | `50` | Max pages for semantic chunking (larger docs use sentence splitting) |
| `ENABLE_FAITHFULNESS_SCORING` | `false` | Verify LLM responses are grounded in sources (extra LLM call) |

See [.env.example](.env.example) for the full list.

---

## Project Structure

```
rag-template-multimodel/
├── app.py                 # FastAPI entry point, all endpoints, DB bootstrap
├── core/
│   ├── config.py          # Environment-driven configuration
│   ├── db.py              # PostgreSQL connection pool
│   ├── auth.py            # JWT authentication, user registration/login
│   ├── oauth.py           # Google OAuth2 integration
│   ├── tasks.py           # Document processing: load, chunk, embed, OCR
│   ├── retrieval.py       # RAG pipeline: hybrid search, reranking, HyDE
│   ├── export.py          # Conversation export (Markdown, JSON, PDF)
│   ├── worker.py          # Background worker logic
│   └── logging_config.py  # Structured logging setup
├── providers/
│   ├── base.py            # Abstract LLM/embedding provider interface
│   ├── registry.py        # Provider auto-detection and registration
│   ├── llm_openai.py      # OpenAI LLM provider
│   ├── llm_anthropic.py   # Anthropic LLM provider
│   └── embedder_openai.py # OpenAI embedding provider
├── static/
│   └── index.html         # Single-page frontend
├── Dockerfile             # Docker image with OCR dependencies
├── nixpacks.toml          # Nixpacks config (used by bult.ai)
├── requirements.txt       # Python dependencies
└── .env.example           # Environment variable template
```

---

## API Reference

All endpoints except `/api/health` and `/api/auth/*` require a JWT token in the `Authorization: Bearer <token>` header.

### Authentication

```
POST /api/auth/register    -- Create account (email + password)
POST /api/auth/login       -- Get JWT token
GET  /api/auth/me          -- Get current user info
GET  /api/auth/google/login -- Start Google OAuth flow (if configured)
```

### Projects & Documents

```
GET    /api/projects                    -- List user's projects
POST   /api/projects                    -- Create project
DELETE /api/projects/{id}               -- Delete project and all its data
POST   /api/upload                      -- Upload document to project
DELETE /api/documents/batch             -- Delete multiple documents
GET    /api/projects/{id}/documents     -- List documents in project
```

### Chat

```
GET  /api/projects/{id}/conversations          -- List conversations
POST /api/projects/{id}/conversations          -- Create conversation
POST /api/chat                                 -- Send message (SSE streaming response)
GET  /api/conversations/{id}/export?format=md  -- Export conversation (md|json|pdf)
```

### Analytics

```
GET /api/analytics                 -- Usage stats, query history, top documents
```

### System

```
GET /api/health                    -- Health check (no auth required)
GET /api/jobs/{id}/progress        -- Job processing progress
```

---

## How It Works

### Document Processing Pipeline

1. User uploads a file through the web UI
2. A background job is created (status: `queued`)
3. The worker picks up the job and processes the document:
   - Loads the file with PyMuPDF (PDFs) or LlamaIndex readers (other formats)
   - Detects if the PDF has a text layer -- if not, runs OCR (Tesseract)
   - Optionally extracts tables with pdfplumber
   - Splits text into chunks (semantic or sentence-based, configurable)
   - Generates embeddings via OpenAI API (batched for performance)
   - Stores chunks + embeddings in PostgreSQL with pgvector
4. Document status updates: `queued` > `processing` > `ready`

### Query Pipeline

1. User asks a question in a conversation
2. Optional query transformations: multi-query expansion, decomposition, HyDE
3. Hybrid search combines BM25 keyword matching + vector similarity
4. Cross-encoder reranks the top candidates for precision
5. Top chunks are sent to the LLM with a system prompt that enforces inline citations
6. Response streams back to the UI character by character

---

### Customize the System Prompt

Edit `SYSTEM_PROMPT` in `core/config.py` to change citation style, tone, or response structure.

### Change the UI Theme

Edit CSS variables in `static/index.html`. The UI supports light and dark mode.

---

## License

MIT -- see [LICENSE](LICENSE).
