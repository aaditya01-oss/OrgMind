> Part of the **OrgMind AI** platform (AI engine, API, website). For the architecture, the security design and how to run all of it with one command, see [OrgMind-Stack](https://github.com/aaditya01-oss/OrgMind-Stack).
# OrgMind AI Engine

The Python service behind OrgMind AI, a security-first organizational AI assistant built for Nepal.
It answers questions from an organization's own documents, scans documents (OCR), and decides
when a conversation should become a support ticket.

It is one of three repositories:

| Repository | What it is |
|---|---|
| **OrgMind** (this one) | Python / FastAPI AI engine |
| OrgMind-API | .NET 8 API: sign-in, roles, tickets, forms, messaging, audit log |
| OrgMind-Frontend | React app: chat widget and staff console |

The browser never talks to this engine. Only the .NET API does, with a shared key (`INTERNAL_API_KEY`).
The API decides who a person is; the engine trusts it and applies that person's access to every answer.

## What works today

- **Document question answering (RAG)** with pgvector search, per organization. An answer only uses
  documents the asker may see. Each document carries a label: public, registered, internal,
  one department, or admins only.
- **Agents** (LangGraph): a document agent and a support agent, chosen per message.
- **Support tickets:** created from chat, with priority, department and customer sentiment.
  A code check (not only the prompt) stops duplicate tickets for the same problem.
- **Document scanning (OCR)** with Tesseract, including field extraction for form "scan to fill".
- **Personal-data detection** (`pii_detector.py`).
- **Conversation memory** and **audit events.** Question events keep the lengths and a short preview of the question, not the full text; this is on the list to tighten.
- **Checks you can run:** `sentiment_eval.py`, `security_eval.py`, `docmind_eval.py`.

## Known limits (honest list)

- Runs on Groq's free tier, so heavy use is rate-limited.
- Answers can be wrong. Retrieval quality depends on the documents; there is no human review step.
- No automated unit-test suite yet. The `*_eval.py` scripts are checks, not a test framework.
- Not packaged for production: no container image, one process, local settings file.
- Nepali-language support is planned, not done.

## Run it locally

You need: Python 3.13, PostgreSQL 16 with the **pgvector** extension, and the
[Tesseract](https://github.com/UB-Mannheim/tesseract/wiki) program (needed for scanning).

1. **Database.** Create it from `OrgMind-API/db/schema.sql`
   (`CREATE DATABASE orgmind;` then `psql -d orgmind -f schema.sql`).
2. **Settings.** Copy `.env.example` to `.env` and fill in your own values.
   `INTERNAL_API_KEY` must equal `InternalApiKey` in the API's settings.
3. **Install and start:**

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
python api_server.py
```

The engine listens on `http://127.0.0.1:8000`. The first start downloads the embedding model.

## Security notes

- `.env` is git-ignored. Never commit it; rotate any key that was ever committed.
- The engine is bound to `127.0.0.1` on purpose. Do not expose it directly to the internet.