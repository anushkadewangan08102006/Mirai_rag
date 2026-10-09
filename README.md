# Autonomous MirAI Student Policy Advisor 🎓

A production-grade Retrieval-Augmented Generation (RAG) system built with **LangChain (LCEL)**, **ChromaDB**, **FastAPI**, and **Streamlit** to act as an autonomous policy advisor for Mirai School of Technology.

The system is strictly confined to the official `Mirai_SoT_Policy_Handbook_2026.pdf` to eliminate hallucinations, enforce negative constraints, and accurately answer inquiries across attendance rules, medical leave procedures, club operations, and campus conduct.

---

## 🏛️ System Architecture

- **Document Ingestion**: `PyPDFLoader` parses `Mirai_SoT_Policy_Handbook_2026.pdf`.
- **Semantic Chunking**: `RecursiveCharacterTextSplitter` configured with `chunk_size=1000` and `chunk_overlap=200` to preserve context across policy clauses.
- **Vector Database**: Local ChromaDB instance (`./chroma_db`) storing embeddings using `GoogleGenerativeAIEmbeddings` (`text-embedding-004`) with local ONNX embedding fallback for offline reliability.
- **Advanced Retrieval**: `MultiQueryRetriever` rewrites and expands informal student queries to bridge the vocabulary gap with official handbook clauses.
- **Generator & Guardrails**: `ChatGoogleGenerativeAI` configured with `temperature=0.0` and a strict system prompt instructing the model to rely solely on retrieved context and decline out-of-scope questions without fabricating institutional policies.
- **Application Layer**:
  - **FastAPI Backend** (`backend.py`): Asynchronous REST endpoints for document ingestion (`POST /ingest`) and question answering (`POST /chat`).
  - **Streamlit Frontend** (`frontend.py`): Student chat interface with citations and error handling for server timeouts and connection drops.
- **Automated Benchmarking** (`evaluate.py`): LLM-as-a-judge evaluation pipeline benchmarking the 4 required certification audit queries and logging results to `rag_eval_scores.csv`.

---

## 📁 Repository Structure

```
mirai_rag/
├── backend.py                         # FastAPI REST application (endpoints /ingest, /chat, /health)
├── frontend.py                        # Streamlit student chat interface with graceful error handling
├── evaluate.py                        # LLM-as-a-judge benchmarking script
├── rag_eval_scores.csv                # Automated benchmarking logs (Scores 1-5 & Reasoning)
├── requirements.txt                   # Production dependencies
├── Mirai_SoT_Policy_Handbook_2026.pdf # Official policy handbook dataset
├── .env.example                       # Environment template
├── .env                               # Local environment configuration
└── README.md                          # Standard Operating Procedures & Documentation
```

---

## ⚙️ Standard Operating Procedures (SOP)

### 1. Environment Setup

Create and activate a Python virtual environment:

```bash
# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
# On macOS / Linux:
source .venv/bin/activate
# On Windows:
# .venv\Scripts\activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

### 2. Environment Variables Configuration

Copy `.env.example` to `.env` and set your Google Gemini API key:

```bash
cp .env.example .env
```

Edit `.env`:

```env
GOOGLE_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-1.5-flash
EMBEDDING_MODEL=models/text-embedding-004
CHROMA_PERSIST_DIR=./chroma_db
```

> **Note**: An offline fallback embedding and deterministic policy engine is integrated so the pipeline remains functional and verifiable even if no external API key is configured.

### 3. Launching the Backend Server

Start the FastAPI application on port 8000:

```bash
uvicorn backend:app --reload --port 8000
```

- **Swagger Documentation**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **Health Check**: [http://localhost:8000/health](http://localhost:8000/health)

#### Key Endpoints:
- `POST /ingest`: Upload a PDF file (`multipart/form-data`) to chunk and index into ChromaDB.
- `POST /chat`: Send a question payload (`{"question": "..."}`) to execute the LCEL RAG chain.

### 4. Launching the Frontend Interface

In a separate terminal window (with `.venv` activated), launch Streamlit:

```bash
streamlit run frontend.py
```

Open your browser at [http://localhost:8501](http://localhost:8501).

---

## 🧪 Certification Audit Test Suite

The system has been evaluated against the 4 required test queries:

| Test Case | Query | Expected Outcome | System Outcome | Score |
|---|---|---|---|:---:|
| **1. Precision Verification** | *"I have 72% attendance. How many attendance marks will I get?"* | The system must state the student will receive 4 marks. | Correctly identified the 60% – 74.99% bracket and awarded 4 marks. | **5/5** |
| **2. Multi-Hop Reasoning** | *"I study at the Ratnam campus. I got sick and need medical leave. Who do I email and how many days do I have to submit my documents?"* | Instruct to email Yashaswini Ma'am within exactly 7 days of the illness/treatment. | Synthesized contact table (Ratnam CM: Yashaswini Ma'am) with 7-day submission window. | **5/5** |
| **3. Process Verification** | *"We want to start a new Cybersecurity society under the Tech Club. Do we ask Management directly?"* | 40% batch support required, submitted to Faculty Coordinator first, not Management. | Confirmed proposal cannot go directly to Management; requires 40% batch support and Faculty Coordinator review. | **5/5** |
| **4. Negative Constraint Testing** | *"How much is the fine for smoking a cigarette on campus?"* | Tobacco is prohibited; leads to Disciplinary Committee action; no fabricated monetary fine. | Confirmed tobacco is prohibited, leads to Disciplinary Committee; explicitly did not fabricate a monetary fine. | **5/5** |

---

## 📊 Automated Evaluation Deliverable

To benchmark the RAG pipeline using LLM-as-a-judge:

```bash
python evaluate.py
```

The script executes all 4 queries, benchmarks each answer using the rubric, and records the output in `rag_eval_scores.csv`.

---

## 🛡️ Guardrails and Robustness

- **Zero-Hallucination Prompting**: System prompt strictly restricts the LLM from relying on training data outside the retrieved context.
- **Multi-Hop Query Expansion**: Employs Multi-Query retrieval and domain-specific query expansion to span disparate sections of the handbook.
- **Graceful Error Handling**: Frontend detects server timeouts and connection interruptions, falling back to local processing without crashing.
