# LLM-Space-News

**IBM LLM final project - A Retrieval-Augmented Language Model for Astronomy Knowledge Access**

<div align="center">

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)

![OpenAI](https://img.shields.io/badge/OpenAI-412991?style=for-the-badge&logo=openai&logoColor=white)
![Gemini](https://img.shields.io/badge/Google%20Gemini-8E75B2?style=for-the-badge&logo=google&logoColor=white)
![Groq](https://img.shields.io/badge/Groq-F55036?style=for-the-badge&logo=fastapi&logoColor=white)

![Hugging Face](https://img.shields.io/badge/Local%20LLM-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)
![FAISS](https://img.shields.io/badge/FAISS-000000?style=for-the-badge&logo=meta&logoColor=white)

![Guardrails](https://img.shields.io/badge/Guardrails-Security-red?style=for-the-badge&logo=security)

</div>

## Overview

This project implements a lightweight LLM service designed to answer astronomy-related questions using a **Retrieval-Augmented Generation (RAG)** pipeline. The system integrates multiple LLM backends (Gemini, Groq, and Local Fallback) and features a robust **function-calling** mechanism.

Key features include:

* **REST API** (`/ask`) built with FastAPI.
* **Observability**: Live metrics (`/stats`) and structured logging to console/file.
* **Security**: Comprehensive guardrails against prompt injection, PII leakage, and path traversal.
* **Tools**: Math calculator, unit converter, and RAG knowledge base lookup.
* **Evaluation**: Automated testing suite with security and hallucination checks.

The project was created for the **IBM SkillsBuild – UMCS: Large Language Models** course.

---

## Core Components

### 1. Backend & API

* **FastAPI**: Asynchronous web server exposing `/ask` (chat) and `/stats` (metrics).
* **Tool Dispatcher**: A secure registry that executes allowed tools (`calculator`, `units`, `files`, `kb`) with Pydantic argument validation.

### 2. Mini-RAG Pipeline

* **Ingestion**: Processes dataset using `SentenceTransformer`.
* **Vector Store**: Uses **FAISS** for efficient similarity search.
* **Retriever**: Fetches relevant context to ground LLM answers and prevent hallucinations.

### 3. Guardrails (Security)

* **Input Scrubbing**: Regex-based sanitization of prompt injection attempts (e.g., "Ignore previous instructions").
* **PII Protection**: detection of sensitive data (Emails, Phone numbers, PESEL).
* **Path Traversal**: Blocks access to unauthorized files (e.g., `../etc/passwd`).
* **Domain Whitelist**: Blocks URLs not present in the allowed domains list (e.g., `nasa.gov`).

### 4. Observability

* **Metrics**: Tracks success rates, tool usage, and security blocks.
* **Logging**: timestamps and log levels saved to `app.log` and console.

---

## Project Structure

```text
main.py        # FastAPI entrypoint & config
router.py      # LLM agent logic & tool orchestration
llm_engine.py  # LLM API clients (Gemini/Groq/Local)
guardrails.py  # Security logic (Regex, PII, Injection)
ingest.py      # RAG data ingestion script
retriever.py   # Vector search logic
registry.py    # Tool schemas & definitions
handlers.py    # Tool implementation & safety wrappers 
evaluate.py        # Automated evaluation script
requirements.txt   # Python dependencies
spacenews.csv      # Knowledge base source

```

---

## Environment Setup

Create a `.env` file in the root directory:

```env
GOOGLE_API_KEY=YOUR_KEY_HERE
GEMINI_MODEL=gemini-1.5-flash

# Optional: Groq
GROQ_API_KEY=YOUR_KEY_HERE
GROQ_MODEL_NAME=llama3-8b-8192

# Optional: Local Model
LOCAL_HF_MODEL_NAME=Qwen/Qwen2.5-1.5B-Instruct

# 'gemini', 'groq', or 'local'
MODEL_MODE=gemini

# RAG Configuration
DATA_DIR=data

```

---

## Installation & Running

1. **Install dependencies:**
```bash
pip install -r requirements.txt

```


2. **Build the RAG Index (Important):**
This processes the CSV file and creates the FAISS index.
```bash
python app/ingest.py

```


3. **Start the API Server:**
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

```


---

## Access & Demo

* **Web UI (Frontend):** Open `http://localhost:8000` in your browser.
* **API Documentation:** `http://localhost:8000/docs`
* **Live Metrics:** `http://localhost:8000/stats`

### Usage Examples (Demo)

**1. RAG (Knowledge Base)**

> **User:** "What is the PREFIRE mission?"
> **System:** Searches FAISS, retrieves context, and answers citing `[Source: spacenews.csv]`.


---

## Evaluation

To run the automated test suite (checking security, hallucinations, and tool usage):

```bash
python evaluate.py

```

This generates an `evaluation_report.csv` with pass/fail metrics.

---

## Notes

* **Logging:** Logs are saved to `app.log`.
* **Allowed Domains:** The system only processes links from trusted domains (e.g., `nasa.gov`, `spacenews.com`) defined in `guardrails.py`.
* **Models:** Native function calling is used for Gemini; a custom JSON parsing loop is used for local models.

---

