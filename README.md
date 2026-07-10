# AI Customer Support Agent

A RAG-based chatbot that answers questions from any PDF you give it — grounded strictly in the document, so it doesn't make things up. Upload a file, ask questions, and it retrieves the relevant part of the document and answers from that, with memory across the conversation.

I built this to actually understand how a full RAG pipeline works — not just call an API and hope for good answers, but wire up the retrieval, re-ranking, memory, and session handling myself and deal with the problems that come up along the way.

**Live demo:** [https://ai-support-agent-ziwrcfrmdqskm8crjcxngl.streamlit.app]

---

## What it does

Upload a PDF in the sidebar, hit **Build Knowledge Base**, and start asking questions. The answers come only from what's actually in the document — if it's not there, it says so instead of guessing.

It also remembers the conversation. Ask "What is a Label widget?" and then follow up with "what are its bg and fg options?" and it correctly resolves "its" back to the widget you just asked about, since each session keeps its own chat history.

You can bring your own Groq API key, or leave it blank and it'll fall back to a default key I've set up on the backend — so people can try it without needing to sign up for anything first.

---

## Why it's built this way

**Two-stage retrieval instead of one.** The first version just grabbed the top 3 chunks by similarity search and called it done. It worked, but on short or vague questions it would sometimes surface a chunk that was topically close but not actually the best match — plain cosine similarity compares the query and each chunk independently, so it doesn't always catch which one really answers the question. Now it pulls the top 10 candidates first, then a cross-encoder re-ranks all 10 by scoring the query and each chunk together, and only the best 3 of those go into the final prompt. Costs nothing extra — the re-ranker runs locally.

**A build button instead of automatic rebuilding.** Originally the pipeline tried to rebuild on every interaction, which is wasteful and, worse, caused a bug where swapping in a new document silently kept using the old one because Streamlit's caching didn't realize the file had changed. Now the knowledge base only builds when you explicitly click the button, and I hash the file's contents so a genuinely different upload forces a fresh build instead of quietly reusing stale data.

**In-memory vector storage per session.** Early on this used a persisted ChromaDB folder on disk, which is a problem the moment more than one person uses the app at the same time on a shared server — their documents could end up mixed together. Switching to in-memory storage means each session's data stays isolated.

---

## How it works

```
upload PDF → click Build Knowledge Base
        ↓
PDF split into chunks → embedded → stored in an in-memory vector store
        ↓
question comes in → top 10 chunks retrieved by similarity
        ↓
cross-encoder re-ranks those 10 → keeps top 3
        ↓
top 3 chunks + chat history + question → sent to the LLM
        ↓
answer shown + saved to session memory
```

**Stack:**
- Groq API (LLaMA 3.1 8B Instant) for generation — free tier
- LangChain for chaining retrieval → prompt → LLM
- ChromaDB (in-memory) as the vector store
- `all-MiniLM-L6-v2` for embeddings
- `ms-marco-MiniLM-L-6-v2` cross-encoder for re-ranking
- Streamlit for the interface

---

## Project structure

```
├── app.py              # upload handling, retrieval, re-ranking, memory, UI — all in one file
├── requirements.txt
└── README.md
```

---
Running it locally

bash
git clone [your-repo-url]
cd ai-support-agent
pip install -r requirements.txt
streamlit run app.py

If you want the "no API key entered" fallback to work, add your own key to .streamlit/secrets.toml:

toml
GROQ_API_KEY = "your_key_here"

Otherwise, visitors can just paste their own free key from console.groq.com directly in the sidebar.
---

## What I'd still like to improve

- No real benchmark yet on how much the re-ranking step actually helps — I've tested it manually and the difference is noticeable, but I want to run a proper before/after comparison on a labeled set of questions
- Only handles one PDF at a time; no multi-document knowledge bases yet
- Session memory and uploaded files don't persist between visits — intentional for privacy, but means a returning user has to re-upload each time
- Next thing on my list: either query rewriting for vague/short questions, or a self-check step where the model verifies its own answer is actually backed by the retrieved text before responding

---

## About Me

[Dinesh kannaa K] — BSc Data Science student, Thiagarajar College of Arts and Science
[www.linkedin.com/in/dinesh-kannaa-2780a5378] · [https://github.com/dineshkannaak/ai-support-agent]
