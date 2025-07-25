# Medical RAG

**A production-ready, extensible Retrieval-Augmented Generation (RAG) system with advanced features for multi-lingual, and domain-specific (especially medical) question answering.**

---

## 🚀 Key Features

- **Hybrid Retrieval:** Combines vector search (semantic) and keyword/BM25 search for robust, accurate retrieval.
- **Multi-modal Support:** Handles text, images (via GPT-4o), audio (Whisper), and video (transcript + frame chunking).
- **Advanced Chunking:** Intelligent, semantic, and structural chunking for optimal context retrieval.
- **Query Expansion & Self-Query:** LLM-based query rewriting, metadata extraction, and multi-perspective retrieval.
- **Reranking:** LLM-based, cross-encoder, and MMR reranking for high-quality context selection.
- **Conversational Memory:** Short-term (session) and long-term (user profile, semantic search) memory for multi-turn dialogue.
- **Medical Knowledge Integration:** UMLS, NER, and translation pipeline for Vietnamese/English medical Q&A.
- **Evaluation & Monitoring:** RAGAS, LLM-based evaluation, and experiment tracking (CometLLM, WandB).
- **Finetuning:** Supports embedding and LLM finetuning (LoRA, SentenceTransformers).
- **Scalable Serving:** Supports OpenAI API, LMDeploy, vLLM, Infinity, and self-hosted Qdrant.
- **Extensible UI:** Streamlit-based, with session management, chunking preview, and medical mode.

---

## 🏗️ System Architecture

```
User Query (VN/EN/Multimodal)
    ↓
[Translation] (if needed)
    ↓
[Medical NER] → [UMLS API] → [MMR Ranking] → [Medical Context]
    ↓
[Query Expansion/Self-Query]
    ↓
[Hybrid Retrieval: Qdrant (vector+BM25)]
    ↓
[Reranking (LLM/Cross-Encoder/MMR)]
    ↓
[Context Merging: Memory + Medical + Retrieved]
    ↓
[LLM Generation]
    ↓
[Evaluation & Monitoring]
    ↓
[UI/Feedback]
```

---

## ⚡ Quick Start

### 1. Prerequisites

- Docker & docker-compose
- (Optional) UMLS API key for medical features

### 2. Build & Run

```sh
docker-compose up --build
```

- Access the UI at: [http://localhost:8501](http://localhost:8501)

### 3. Qdrant Setup

Qdrant is used as the vector database. It is automatically started via docker-compose.  
For manual setup or advanced options, see: https://qdrant.tech/documentation/quick-start/

---

## 🧠 Advanced RAG Techniques

- **Query Expansion:** LLM generates multiple queries for broader semantic coverage.
- **Self-Query:** Extracts metadata (tags, filters) from user queries for precise retrieval.
- **Hybrid Search:** Combines vector and keyword search for best coverage.
- **Reranking:** Uses LLMs or cross-encoders to select the most relevant context.
- **Multi-stage Retrieval:** Supports multi-hop, agent-based, and memory-augmented retrieval.

---

## 🏥 Medical Q&A Pipeline

- **Vietnamese/English Support:** Automatic translation (EnViT5) for Vietnamese queries.
- **Medical NER:** Extracts diseases, drugs, symptoms, etc.
- **UMLS Integration:** Retrieves medical knowledge and relations, reranked by MMR.
- **Context Merging:** Medical knowledge is prioritized and merged with document retrieval for LLM generation.

---

## 🧩 Extensibility

- **Chunking:** Plug-and-play strategies (semantic, structural, fast, intelligent).
- **Memory:** Modular short-term and long-term memory, easy to extend for new use cases.
- **Retrieval:** Swap or combine retrieval backends (Qdrant, Elastic, etc.).
- **LLM Serving:** Compatible with OpenAI API, LMDeploy, vLLM, Infinity, etc.

---

## 📊 Evaluation & Monitoring

- **RAGAS:** Automated metrics for context relevancy, recall, answer similarity, and correctness.
- **LLM-based Evaluation:** Use GPT-4o or other LLMs for qualitative assessment.
- **Experiment Tracking:** CometLLM, WandB integration for prompt and model monitoring.

---

## 🔒 Security & Best Practices

- **API Keys & Secrets:** Never commit `.env`, `settings.py`, or any API keys (see `.gitignore`).
- **Finetuning Data:** All `/finetuning` folders are ignored by git.
- **User Data:** Session/user data is not stored unless explicitly enabled.

---

## 📚 References

- [Qdrant Documentation](https://qdrant.tech/documentation/)
- [SentenceTransformers](https://www.sbert.net/)
- [RAGAS](https://docs.ragas.io/)
- [UMLS](https://www.nlm.nih.gov/research/umls/)
- [LMDeploy](https://lmdeploy.readthedocs.io/)
- [vLLM](https://docs.vllm.ai/)
- [Infinity](https://michaelfeil.eu/infinity/0.0.41/)

---

## 📝 Contribution & Development

- Fork, clone, and develop in feature branches.
- PRs are welcome! Please do **not** commit any API keys, `.env`, or `/finetuning` data.

---

## 💡 Example Use Cases

- Medical Q&A (Vietnamese/English)
- Enterprise document search
- Multi-modal knowledge assistant
- Research assistant with memory and context

---

**For more details, see the documentation in each module and the `medical/README.md` for medical pipeline specifics.**
