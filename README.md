# AI Document Q&A Agent (Advanced RAG)
A RAG-based chatbot that answers questions from any PDF you give it — grounded strictly in the document, so it doesn't make things up. Currently demoed on the Indian Penal Code, but works with any document you upload.

I built this to actually understand how a production-grade RAG pipeline works, including the parts that don't show up in tutorials — chunking documents in a way that doesn't cut answers in half, catching cases where plain vector search quietly fails, and making follow-up questions actually work. Most of what's below came from debugging real failures on a real legal document, not from following a guide.

**Live demo:** [[https://ai-support-agent-ziwrcfrmdqskm8crjcxngl.streamlit.app]]

---

## What it does

Upload a PDF, click **Build Knowledge Base**, and ask it anything. It answers only from what's actually in the document — if the answer isn't there, it says so instead of guessing. You can bring your own Groq API key, or leave it blank and it falls back to a default key on the backend, so anyone can try it without signing up for anything first.

It also holds a real conversation. Ask "What does Section 302 deal with?", then follow up with "what's the punishment for it?" — it correctly resolves "it" back to Section 302 and answers accurately, instead of getting lost the moment a pronoun shows up.

---

## The three real bugs this project taught me

**1. Semantic search fails on exact numeric lookups.**
Testing this on the IPC, I found that asking "What does Section 302 deal with?" returned "I don't know" — even though the punishment-for-murder text was sitting right there in the document. The problem: embedding models capture *meaning*, not literal digits. A chunk starting with "302. Punishment for murder" doesn't reliably score as similar to a question containing the number 302, because "302" carries almost no semantic weight on its own.

**Fix — hybrid retrieval.** Alongside normal vector search, I added a regex check: if the question references a specific section number, directly force-include any chunk that literally starts with that number, before re-ranking. This guarantees numeric lookups can't be missed regardless of how the question is phrased.

**2. Chunk boundaries were cutting answers in half.**
The first version used 500-character chunks, which was fine for shorter FAQ-style text but kept slicing legal definitions apart mid-sentence — the section number would land in one chunk and its actual content in the next, with no overlap connecting them. Increased chunk size to 1000 with 200-character overlap, which keeps full section definitions intact as a single retrievable unit.

**3. Follow-up questions retrieved nothing relevant.**
Memory alone doesn't fix retrieval. The chat history was being passed to the LLM correctly, but retrieval happens *before* the LLM ever sees anything — so a follow-up like "what's the punishment for it?" was being embedded and searched literally, with no idea what "it" referred to.

**Fix — query contextualization.** Before retrieval runs, an LLM call rewrites the follow-up question into a standalone one using the conversation history ("what's the punishment for it?" → "what is the punishment for Section 302 murder under the IPC?"), and *that* rewritten version is what actually gets searched.

---

## How it works

```
upload PDF → click Build Knowledge Base
        ↓
PDF split into 1000-char chunks (200 overlap) → embedded → in-memory vector store
        ↓
question comes in → if it has chat history, rewritten into a standalone question
        ↓
top 25 chunks retrieved by similarity + any exact section-number match force-included
        ↓
cross-encoder re-ranks the full candidate pool → keeps top 3
        ↓
top 3 chunks + chat history + question → sent to the LLM
        ↓
answer shown + saved to session memory
```

**Stack:**
- Groq API (LLaMA 3.1 8B Instant) for generation — free tier
- LangChain for chaining retrieval → prompt → LLM, and for query contextualization
- ChromaDB (in-memory, session-isolated) as the vector store
- `all-MiniLM-L6-v2` for embeddings
- `ms-marco-MiniLM-L-6-v2` cross-encoder for re-ranking
- A regex-based hybrid keyword fallback for exact numeric/ID lookups
- Streamlit for the interface

---

## Running it locally

```bash
git clone [your-repo-url]
cd ai-document-qa-agent
pip install -r requirements.txt
streamlit run app.py
```

To enable the "no API key entered" fallback, add your own key to `.streamlit/secrets.toml`:
```toml
GROQ_API_KEY = "your_key_here"
```
Otherwise, visitors can paste their own free key from [console.groq.com](https://console.groq.com) directly in the sidebar.

---

## Project structure

```
├── app.py              # upload handling, hybrid retrieval, re-ranking, contextualization, memory, UI
├── requirements.txt
└── README.md
```

---

## What I'd still like to improve

- No formal benchmark yet on retrieval accuracy before vs. after the hybrid fix — verified manually on real test questions, want real numbers eventually
- Query contextualization adds an extra LLM call per follow-up question — fine on light usage, worth watching on Groq's free-tier rate limits under heavier traffic
- Only handles one document at a time, no multi-document knowledge bases
- Next on my list: a Self-RAG style grounding check, so the model verifies its own answer is actually backed by the retrieved text before responding

---

## About Me

[Dinesh kannaa K] — BSc Data Science student, Thiagarajar College of Arts and Science
[www.linkedin.com/in/dinesh-kannaa-2780a5378] ·[https://github.com/dineshkannaak/ai-support-agent]
