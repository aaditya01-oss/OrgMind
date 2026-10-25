# 🧠 OrgMind AI

### A Security-First Organizational AI Platform for Nepal

**OrgMind AI** is an AI-powered organizational intelligence platform designed to help Nepali SMEs, cooperatives, and institutions securely manage knowledge, process documents, automate workflows, and improve internal support systems.

Built with a focus on **security, privacy, scalability, and Nepali language support**, OrgMind AI aims to make enterprise-grade AI capabilities accessible to organizations across Nepal.

---

## 👨‍💻 Built By

**Aaditya Ghimire**
Computer Engineering Student — TU Affiliated University, Kathmandu, Nepal
Building OrgMind AI as a real-world AI platform for Nepal's organizations.

GitHub: [aaditya01-oss](https://github.com/aaditya01-oss)

---

## 📌 Table of Contents

* [Overview](#-overview)
* [Core Services](#-core-services)
* [Key Features](#-key-features)
* [Technology Stack](#-technology-stack)
* [System Architecture](#-system-architecture)
* [Security Architecture](#-security-architecture)
* [Getting Started](#-getting-started)
* [Environment Configuration](#-environment-configuration)
* [Project Structure](#-project-structure)
* [Development Roadmap](#-development-roadmap)
* [Contributing](#-contributing)
* [License](#-license)

---

## 🚀 Overview

Organizations generate valuable information through documents, policies, reports, customer interactions, and internal communications. However, accessing this information efficiently while maintaining data privacy and security remains a challenge.

**OrgMind AI transforms organizational data into secure, searchable, and actionable intelligence.**

The platform combines retrieval-augmented generation (RAG), intelligent document processing, workflow automation, and layered AI security mechanisms to help organizations retrieve trusted information and automate repetitive tasks without compromising sensitive data.

Designed with the needs of Nepali organizations in mind, OrgMind AI supports local languages, organizational workflows, and Nepal-specific regulatory requirements.

### 🎯 Target Users

* Small and medium-sized enterprises (SMEs)
* Cooperatives and financial institutions
* Educational institutions
* Healthcare and service organizations
* Government and public-sector institutions

---

## 🧩 Core Services

| Service | Description |
| --- | --- |
| **OrgBot** | Organizational knowledge assistant that answers questions using internal documents |
| **DocMind** | Intelligent document processing — extract, index and retrieve information from PDFs and DOCX |
| **SupportDesk** | AI-powered support with ticket management and automated responses |
| **SecureLayer** | Prompt injection detection, PII masking, audit logging and tenant isolation |
| **NepaliMind** | Nepali language support for multilingual organizational communication |
| **ComplianceBot** | Nepal-specific regulatory compliance assistant |
| **InterLink** | Cross-department workflow automation |

---

## ✨ Key Features

### 🧠 Organizational Intelligence
* RAG for knowledge-grounded responses
* Semantic search across organizational documents
* Context-aware question answering
* Document chunking, embedding generation and vector retrieval

### 📄 Intelligent Document Processing
* PDF and DOCX ingestion and text extraction
* Document chunking with overlap strategy
* Embedding generation using sentence-transformers
* pgvector storage and similarity search

### 🔐 Security and Privacy
* **3-layer prompt injection defense** — pattern matching, length anomaly detection, LLM-based detection
* **Nepal-specific PII detection and masking** — phone numbers, citizenship numbers, PAN, bank accounts, NPR salary
* **Per-tenant data isolation** — org_id based separation
* **JWT authentication** with BCrypt password hashing
* **Audit logging** — every event tracked in PostgreSQL with JSONB
* **Langfuse monitoring** — LLM observability and cost tracking

### 🌐 Nepal-Focused AI
* Nepali language support
* Nepal-specific PII patterns
* Nepal regulatory document support
* Architecture designed for local compliance requirements

---

## 🛠️ Technology Stack

| Layer | Technology | Purpose |
| --- | --- | --- |
| AI Engine | Python | AI logic and processing |
| AI Framework | LangChain | LLM integration and document processing |
| AI Orchestration | LangGraph | Stateful multi-agent workflows |
| Backend API | C# / ASP.NET Core | REST APIs and business logic |
| Database | PostgreSQL | Relational data storage |
| Vector Search | pgvector | Vector storage and similarity search |
| Frontend | React | User interface |
| Containerization | Docker | Development and deployment |
| Authentication | JWT | Token-based authentication |
| Password Security | BCrypt | Password hashing |
| LLM Observability | Langfuse | LLM tracing and monitoring |
| Embeddings | sentence-transformers | Text to vector conversion |

---

## 🏗️ System Architecture

```text
User Input (Nepali/English)
         ↓
   React Frontend (Port 3000)
         ↓ HTTP + JWT
   C# ASP.NET Core API (Port 5147)
         ↓
   SecureLayer FIRST
   - Prompt injection check
   - JWT authentication  
   - RBAC permission check
         ↓
   NepaliMind (language processing)
         ↓
   OrgBot Router (intent detection)
         ↓
    ┌────┬─────┬──────────┬────────┐
    ▼    ▼     ▼          ▼        ▼
   HR  Finance Operations Support ComplianceBot
  Agent Agent   Agent     Desk    
         ↓
   Knowledge Layer
   DocMind — RAG + pgvector
         ↓
   LLM (Groq API)
         ↓
   SecureLayer AGAIN (output check)
         ↓
   Audit Logger + Langfuse
         ↓
   Response to User
```

---

## 🔒 Security Architecture

Security is a core design principle of OrgMind AI — not an afterthought.

### Multi-Layer Prompt Injection Defense
* **Layer 1** — Pattern matching: blocks known attack phrases instantly
* **Layer 2** — Length anomaly: catches unusually long injection attempts  
* **Layer 3** — LLM detection: catches sophisticated attacks that bypass patterns

An attacker must bypass ALL THREE layers.

### Nepal-Specific PII Detection
Detects and masks:
* Nepali phone numbers (+977-98XXXXXXXX)
* Citizenship numbers
* PAN numbers
* Bank account numbers
* NPR salary information
* Email addresses

### Why not Pinecone or other cloud vector DBs?
OrgMind uses pgvector inside PostgreSQL running on YOUR infrastructure. Your organizational data never leaves your server. No third party access. No monthly cloud bill. That's the security promise.

---

## ⚙️ Getting Started

### Prerequisites
* Python 3.11+
* .NET 8 SDK
* Node.js and npm
* Docker Desktop
* Git

### 1. Clone the Repositories

```bash
# Python AI Engine
git clone https://github.com/aaditya01-oss/OrgMind.git

# C# Backend API  
git clone https://github.com/aaditya01-oss/OrgMind-API.git

# React Frontend
git clone https://github.com/aaditya01-oss/OrgMind-Frontend.git
```

### 2. Start Database

```bash
docker run -d --name orgmind-db \
  -e POSTGRES_PASSWORD=orgmind123 \
  -e POSTGRES_DB=orgmind \
  -p 5433:5432 \
  -v orgmind_data:/var/lib/postgresql/data \
  pgvector/pgvector:pg16
```

### 3. Setup Python AI Engine

```bash
cd OrgMind
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# Add your API keys to .env
python main.py
```

### 4. Setup C# Backend

```bash
cd OrgMind-API/OrgMind.API
dotnet restore
# Configure appsettings.Development.json
dotnet run
```

### 5. Setup React Frontend

```bash
cd OrgMind-Frontend
npm install
npm start
```

---

## 🔑 Environment Configuration

```env
# Groq API
GROQ_API_KEY=your_groq_api_key

# Database
DB_HOST=127.0.0.1
DB_PORT=5433
DB_NAME=orgmind
DB_USER=postgres
DB_PASSWORD=orgmind123

# Langfuse Monitoring
LANGFUSE_PUBLIC_KEY=your_public_key
LANGFUSE_SECRET_KEY=your_secret_key
LANGFUSE_BASE_URL=https://us.cloud.langfuse.com
```

**Never commit real API keys. Add .env to .gitignore.**

---

## 📂 Project Structure
OrgMind/ ← Python AI Engine
├── main.py ← Chat with RAG
├── agent.py ← ReAct agent with tools
├── multi_agent.py← Multi-agent LangGraph system
├── security.py ← Prompt injection detection
├── pii_detector.py← PII detection and masking
├── audit_logger.py← Audit logging
├── monitoring.py ← Langfuse integration
├── memory.py ← Persistent memory system
├── embeddings.py ← Embedding generation
├── search.py ← Semantic search
├── document_processor.py ← PDF ingestion
└── config.py ← Configuration

OrgMind-API/ ← C# ASP.NET Core Backend
├── Models/ ← ChatRequest, ChatResponse, User
├── Services/ ← IChatService, IAuthService
├── Data/ ← OrgMindDbContext (EF Core)
└── Program.cs ← API entry point

OrgMind-Frontend/ ← React Frontend
├── src/
│ ├── App.js ← Main chat interface
│ └── Login.js ← Authentication
└── package.json


---

## 🗺️ Development Roadmap

- [x] Phase 1 — LLM Integration (API, system prompts, streaming, structured outputs)
- [x] Phase 2 — RAG Systems (pgvector, embeddings, semantic search, document ingestion)
- [x] Phase 3 — C# .NET Backend (REST API, EF Core, JWT authentication)
- [x] Phase 4 — Agentic Systems (ReAct agents, tools, multi-agent, persistent memory)
- [x] Phase 5 — LLMOps + Security (prompt injection, PII, audit logs, Langfuse)
- [x] Phase 6 — Frontend (React, login, chat interface)
- [ ] File upload UI for document ingestion
- [ ] Real-time streaming in React (SSE)
- [ ] Admin dashboard
- [ ] Full NepaliMind implementation
- [ ] ComplianceBot with Nepal regulations
- [ ] Production deployment
- [ ] First pilot client in Nepal

---

## 📜 License

MIT License — see LICENSE file for details.

---

<p align="center">
  <strong>OrgMind AI — Intelligence for Organizations. Security by Design.</strong><br>
  Built in Nepal 🇳🇵
</p>