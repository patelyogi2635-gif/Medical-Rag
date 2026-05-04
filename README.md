# 🧠 MediRAG — Multi-Document Medical Knowledge Assistant

> A production-grade Retrieval-Augmented Generation (RAG) system engineered for high-accuracy, low-hallucination medical Q&A across large document corpora.

![Python](https://img.shields.io/badge/Python-3.11+-blue?style=flat-square&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-async-009688?style=flat-square&logo=fastapi)
![Groq](https://img.shields.io/badge/Groq-Llama_3.1-orange?style=flat-square)
![FAISS](https://img.shields.io/badge/FAISS-vector_search-purple?style=flat-square)
![License](https://img.shields.io/badge/license-MIT-green?style=flat-square)

---

## 📌 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Quickstart](#quickstart)
- [API Reference](#api-reference)
- [Key Design Decisions](#key-design-decisions)
- [Performance](#performance)
- [Roadmap](#roadmap)
- [Disclaimer](#disclaimer)

---

## Overview

MediRAG is an end-to-end AI knowledge assistant that ingests multiple medical PDFs and answers clinical queries with verified, context-grounded responses. The system is built around three core engineering goals:

| Goal | Implementation |
|------|---------------|
| **Accuracy** | Hybrid BM25 + dense vector retrieval surfaces the most relevant context |
| **Hallucination Reduction** | A dedicated verification layer cross-checks every generated answer against retrieved chunks |
| **Performance** | Precomputed embeddings + FAISS indexing + Groq's low-latency inference keep P95 response times under 2 seconds |

> **Real-world applicability:** The same architecture generalizes to legal, financial, or enterprise document search — any domain where factual precision is non-negotiable.

---

## Architecture
┌─────────────────────────────────────────────────────────────────────┐
│                        MediRAG Pipeline                             │
│                                                                     │
│  ┌──────────┐  ┌──────────────┐  ┌──────────────────────────────┐  │
│  │   PDF    │─▶│  Semantic    │─▶│       Embedding Layer        │  │
│  │  Corpus  │  │  Chunker     │  │   (Sentence Transformers)    │  │
│  └──────────┘  │  (overlap)   │  │       + Disk Cache           │  │
│                └──────────────┘  └──────────────┬───────────────┘  │
│                                                 │                  │
│                                  ┌──────────────▼───────────────┐  │
│                                  │        FAISS Index           │  │
│                                  │        (Persisted)           │  │
│                                  └──────────────┬───────────────┘  │
│                                                 │                  │
│  ┌──────────┐  ┌──────────────────────────────────────────────┐    │
│  │  User   │─▶│              Hybrid Retrieval                │◀┘  │
│  │  Query  │  │   BM25 (sparse) + FAISS (dense)              │    │
│  └──────────┘  │   → Context Ranking via RRF Fusion          │    │
│                └──────────────────────┬───────────────────────┘   │
│                                       │                           │
│                       ┌──────────────▼───────────────┐           │
│                       │   LLM Layer (Llama 3.1/Groq) │           │
│                       │   + Medical Prompt Template  │           │
│                       └──────────────┬───────────────┘           │
│                                      │                           │
│                       ┌──────────────▼───────────────┐           │
│                       │      Verification Layer       │           │
│                       │     (Groundedness Check)      │           │
│                       └──────────────┬───────────────┘           │
│                                      │                           │
│                       ┌──────────────▼───────────────┐           │
│                       │    FastAPI ──▶ Chatbot UI     │           │
│                       └───────────────────────────────┘          │
└─────────────────────────────────────────────────────────────────────┘

### Component Breakdown

#### 📄 Document Ingestion (`ingest.py`)
- Parses multi-page PDFs using **PyMuPDF** / **pdfplumber**
- Applies semantic chunking with configurable overlap to preserve cross-boundary context
- Avoids naive fixed-size splitting that breaks mid-sentence or mid-concept

#### 🔢 Embedding Layer
- Uses `sentence-transformers` (`all-MiniLM-L6-v2` or domain-tuned variants)
- Embeddings are serialized to disk on first run — subsequent loads skip re-computation entirely
- Ensures **zero cold-start penalty** in production

#### 🗄️ Vector Database (FAISS)
- Flat L2 index for exact nearest-neighbor search on moderate corpora
- Index is persisted alongside the embedding cache for stateless restarts

#### 🔍 Hybrid Retrieval (`retriever.py`)
- **BM25** captures exact keyword matches (drug names, ICD codes, dosage values)
- **FAISS** dense search captures semantic intent ("side effects of blood thinners")
- Scores are fused via **Reciprocal Rank Fusion (RRF)** before context ranking
- _Rationale: neither method alone handles the full diversity of clinical queries_

#### 🤖 LLM Layer (`rag_pipeline.py`)
- `llama-3.1-8b-instant` via **Groq API** for sub-second inference
- Prompt template enforces:
  - Source-grounded responses only
  - Explicit uncertainty acknowledgment when context is insufficient
  - Safe refusal for out-of-scope clinical decisions

#### ✅ Verification Layer (`verifier.py`)
- A secondary LLM pass checks whether the generated answer is entailed by retrieved context
- Returns `verified: YES | NO | PARTIAL` alongside the answer
- `NO` or `PARTIAL` responses are flagged in the UI, preventing silent hallucinations

#### 🚀 Backend (`main.py`)
- FastAPI with async endpoints
- CORS-configured for local frontend development

#### 🖥️ Frontend (`frontend/`)
- Vanilla HTML/CSS/JS chatbot UI
- Displays verification status as a **trust indicator** on each response

---

## Tech Stack

| Layer | Technology | Why |
|-------|-----------|-----|
| Backend | FastAPI + Uvicorn | Async, OpenAPI auto-docs, production-proven |
| LLM | Llama 3.1 via Groq | Best-in-class open-weight model; Groq delivers ~500 tok/s |
| RAG Framework | LangChain | Composable chains, community integrations |
| Vector DB | FAISS | Zero-infra vector search; fast for <1M vectors |
| Embeddings | Sentence Transformers | Lightweight, offline-capable, high retrieval quality |
| Sparse Retrieval | BM25 (`rank_bm25`) | Handles exact medical terminology reliably |
| Frontend | HTML / CSS / JS | Zero-dependency, instantly deployable |

---

## Project Structure
medical-rag/
│
├── app/
│   ├── main.py           # FastAPI app, CORS config, /query endpoint
│   ├── ingest.py         # PDF parsing + semantic chunking
│   ├── retriever.py      # Hybrid BM25 + FAISS retrieval with RRF fusion
│   ├── rag_pipeline.py   # LLM prompt construction + Groq inference
│   └── verifier.py       # Groundedness verification layer
│
├── data/
│   └── pdfs/             # Drop medical PDFs here — auto-ingested on startup
│
├── embeddings/           # Auto-generated: persisted FAISS index + embedding cache
│
├── frontend/
│   ├── index.html        # Chatbot UI shell
│   ├── style.css         # Styling + verification status indicators
│   └── script.js         # Fetch API integration + response rendering
│
├── .env                  # API keys (never commit this)
├── requirements.txt
└── runtime.txt           # Python version pin (for deployment targets)

---

## Quickstart

### Prerequisites
- Python 3.11+
- A free [Groq API key](https://console.groq.com)

### 1. Clone

```bash
git clone https://github.com/your-username/medical-rag.git
cd medical-rag
```

### 2. Create Virtual Environment

```bash
# macOS / Linux
python -m venv venv && source venv/bin/activate

# Windows
python -m venv venv && venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment

```bash
echo "GROQ_API_KEY=your_groq_api_key_here" > .env
```

### 5. Add Your PDFs

Drop any medical PDFs into `data/pdfs/`. The system ingests and indexes them automatically on first startup.

### 6. Start the Backend

```bash
uvicorn app.main:app --reload
# API running at  http://localhost:8000
# Docs available at http://localhost:8000/docs
```

### 7. Start the Frontend

```bash
cd frontend
python -m http.server 5500
# Open http://localhost:5500 in your browser
```

---

## API Reference

### `POST /query`

Submits a natural language medical query and returns a verified, context-grounded answer.

**Request**

```json
{
  "query": "What are the contraindications of aspirin in pediatric patients?"
}
```

**Response**

```json
{
  "answer": "Aspirin is contraindicated in pediatric patients under 16 years with viral illnesses due to the risk of Reye's syndrome, a rare but serious condition causing liver and brain damage...",
  "verified": "YES",
  "sources": [
    "pharmacology_textbook.pdf — p.142",
    "clinical_guidelines_2023.pdf — p.87"
  ]
}
```

**Verification Status Values**

| Value | Meaning |
|-------|---------|
| `YES` | Answer is fully supported by retrieved context |
| `PARTIAL` | Answer is mostly grounded; minor inference beyond sources |
| `NO` | Answer could not be verified against context — treat with caution |

---

## Key Design Decisions

**Why hybrid retrieval over dense-only search?**

Medical documents contain precise identifiers — drug names, dosage figures, ICD codes — that dense embeddings routinely mis-rank. BM25 ensures these exact terms are never missed. RRF fusion then combines both ranked lists without requiring manual weight tuning.

**Why a separate verification layer?**

LLMs generate fluent text even when context is absent or contradictory. A second-pass groundedness check converts this silent failure into an explicit, observable signal — critical in any safety-adjacent domain.

**Why Groq over a self-hosted model?**

Groq's LPU hardware delivers ~500 tokens/second on Llama 3.1, making it the fastest available inference option for open-weight models. For a demo or MVP, this eliminates GPU infrastructure cost entirely.

**Why cache embeddings to disk?**

Re-embedding a large PDF corpus on every server restart is the single largest source of startup latency. Persisting the FAISS index and embedding vectors reduces cold start from minutes to under a second.

---

## Performance

| Metric | Value |
|--------|-------|
| Embedding generation (first run) | ~60s per 100 pages |
| Embedding load (cached) | < 1s |
| Hybrid retrieval latency | ~80–150ms |
| LLM inference (Groq) | ~300–800ms |
| **End-to-end P95 response time** | **< 2 seconds** |

---

## Roadmap

- [ ] **Page-level citations** — surface exact page numbers alongside answers
- [ ] **Streaming responses** — token-by-token output for perceived speed improvement
- [ ] **Cross-encoder reranking** — replace RRF with a fine-tuned reranker for higher precision
- [ ] **Multi-turn conversation** — maintain session history across queries
- [ ] **User authentication** — JWT-based auth + per-user query history
- [ ] **Docker Compose deployment** — one-command production setup
- [ ] **Evaluation suite** — RAGAS-based benchmarking for retrieval and answer quality

---

## Disclaimer

> ⚠️ This project is developed for **educational and research purposes only**. It is not a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider for clinical decisions.
