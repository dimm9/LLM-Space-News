# LLM-Space-News
IBM LLM final project - A Retrieval-Augmented Language Model for Astronomy Knowledge Access

# LLM Tool-Calling API with Mini-RAG and Guardrails

Python · FastAPI · OpenAI / Gemini · Local LLM · FAISS

## Overview

This project implements a lightweight LLM service with:

* REST API (`/ask`) built using FastAPI,
* support for API-based LLMs (OpenAI / Gemini / Groq) and a local LLM fallback,
* function/tool calling with strict allowlist and argument validation,
* a mini Retrieval-Augmented Generation (RAG) pipeline,
* basic guardrails (prompt-injection heuristics, path sanitization, timeouts),
* evaluation and automated tests.

The project was created for the **IBM SkillsBuild – UMCS: Large Language Models** course.

---

## Core Components

* **FastAPI backend** with `/ask` endpoint
* **Tool registry and dispatcher**

  * allowlisted tools only
  * Pydantic argument validation
  * execution timeouts
* **Mini-RAG**

  * sentence-transformer embeddings
  * FAISS vector index
  * hybrid retrieval with optional reranking
* **Guardrails**

  * prompt-injection detection and scrubbing
  * path traversal protection for file tools
* **Evaluation**

  * automated tests
  * latency and success-rate metrics

---

## Project Structure

```
main.py        # FastAPI entrypoint
router.py      # request orchestration and tool loop
llm_engine.py  # LLM backends
registry.py    # tool schemas and allowlist
handlers.py    # tool implementations
guardrails.py  # input sanitization
ingest.py      # RAG ingestion
retriever.py   # retrieval logic
evaluate.py    # evaluation and tests
```

---

## Environment Setup

Create a `.env` file based on `.env.example`:

```env
MODEL_MODE=gemini     
GOOGLE_API_KEY=your_key
OPENAI_API_KEY=your_key
GROQ_API_KEY=your_key
LOCAL_MODEL_NAME=Qwen/Qwen2.5-1.5B-Instruct
```

---

## Running the Project Locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the API server:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

---

## Access

* **REST API**:
  `http://localhost:8000/ask`

* **API Docs (Swagger UI)**:
  `http://localhost:8000/docs`

---

## Frontend (Simple Web UI)

A minimal frontend is served directly by FastAPI for manual testing.

After starting the server, open:

```
http://localhost:8000
```

The frontend sends requests to the `/ask` endpoint and displays raw responses.

---

## RAG Ingestion

To build the local vector index:

```bash
python ingest.py
```

---

## Evaluation and Tests

Run evaluation:

```bash
python evaluate.py
```

Run tests:

```bash
pytest
```

---

## Notes

* API-based models use native function/tool calling where available.
* The local model uses a stubbed function-calling mechanism.
* All tool executions are validated, allowlisted, and sandboxed.

---

Developed for **IBM SkillsBuild – UMCS: Large Language Models**.

