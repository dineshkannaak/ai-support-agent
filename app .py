import re
import json
import time
import hashlib
from collections import Counter
import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import ChatMessageHistory
from sentence_transformers import CrossEncoder
import os

# Use an authenticated Hugging Face token if one is configured in secrets —
# this gives a higher, more reliable rate limit for downloading the
# embedding and re-ranker model weights, instead of the shared unauthenticated
# quota, which is more prone to slow or stalled downloads.
if "HF_TOKEN" in st.secrets:
    os.environ["HF_TOKEN"] = st.secrets["HF_TOKEN"]

st.set_page_config(page_title="Support Agent", page_icon="🤖", initial_sidebar_state="expanded")
st.title("AI Document Q&A Agent")
st.caption("Upload a PDF and ask questions about it — answers are grounded strictly in the document.")

# ── Sidebar: user provides their own key (optional) and document ──
with st.sidebar:
    st.header("Setup")
    user_api_key = st.text_input(
        "Groq API Key (optional)",
        type="password",
        help="Leave blank to use the demo's default key. Get your own free key at console.groq.com"
    )
    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
    build_clicked = st.button("Build Knowledge Base", type="primary")
    st.divider()
    st.caption("Your key and document are only used for this session and are not stored.")


def _log(msg, t0):
    # Prints to terminal / Streamlit Cloud "Manage app" logs with elapsed
    # time, so a hang shows exactly which stage it's stuck on instead of
    # going completely silent with no way to diagnose it.
    print(f"[{time.perf_counter() - t0:6.1f}s] {msg}", flush=True)


def is_noise_chunk(text):
    stripped = text.strip()
    return bool(re.match(r'^\(Part\s+[IVX]+.*Arts?\.?\s*[\d\-–—,]+.*\)$', stripped))


def is_repetitive_chunk(text, repetition_threshold=0.5):
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    if len(lines) < 6:
        return False

    def normalize(line):
        return re.sub(r'\([a-z]+\)', '(X)', line.lower())

    normalized = [normalize(line) for line in lines]
    counts = Counter(normalized)
    most_common_count = counts.most_common(1)[0][1]

    return (most_common_count / len(lines)) > repetition_threshold


@st.cache_resource(show_spinner=False)
def load_models():
    # Loaded ONCE for the life of the app process, independent of which PDF
    # is uploaded. Previously these lived inside load_pipeline(), so every
    # new file_hash (i.e. any new PDF) forced a cache miss that reloaded —
    # and potentially re-downloaded — both models from scratch.
    t0 = time.perf_counter()
    _log("Loading embedding model (all-MiniLM-L6-v2)...", t0)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    _log("Embedding model loaded. Loading reranker (ms-marco-MiniLM-L-6-v2)...", t0)
    reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    _log("Reranker loaded.", t0)
    return embeddings, reranker


@st.cache_resource(show_spinner="Building knowledge base...")
def load_pipeline(api_key, pdf_path, file_hash):
    t0 = time.perf_counter()

    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=api_key,
        temperature=0.0
    )

    _log("Loading PDF...", t0)
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    _log(f"PDF loaded ({len(docs)} pages). Splitting into chunks...", t0)

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    chunks = [c for c in chunks if not is_noise_chunk(c.page_content)]
    chunks = [c for c in chunks if not is_repetitive_chunk(c.page_content)]
    _log(f"{len(chunks)} chunks after filtering. Fetching models...", t0)

    embeddings, reranker = load_models()
    _log("Models ready. Embedding chunks into Chroma (this is the slow step for large PDFs)...", t0)

    vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings)
    _log("Vectorstore built.", t0)

    retriever = vectorstore.as_retriever(search_kwargs={"k": 25})

    decompose_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Analyze this question. If it asks about TWO OR MORE distinct topics, "
         "articles, or sections that would each need a separate search to answer "
         "well — comparisons, relationships, multi-part questions — break it into "
         "separate standalone search queries, one per topic, MAXIMUM 3 topics. "
         "If it's a single-topic question, return it unchanged as the only item.\n\n"
         "Output ONLY a JSON list of strings, nothing else, no explanation. "
         'Example: ["Article 14 equality before law", "Article 15 discrimination"] '
         'or ["Article 21 right to life"] for a single-topic question.'),
        ("human", "{question}")
    ])
    decompose_chain = decompose_prompt | llm | StrOutputParser()

    def decompose_query(query):
        try:
            raw = decompose_chain.invoke({"question": query}).strip()
            if raw.startswith("```"):
                raw = raw.strip("`").replace("json", "", 1).strip()
            sub_queries = json.loads(raw)
            if isinstance(sub_queries, list) and sub_queries and all(isinstance(q, str) for q in sub_queries):
                return sub_queries[:3]
        except Exception:
            pass
        return [query]

    def retrieve_and_rerank(query, top_k=3):
        sub_queries = decompose_query(query)
        multi_topic = len(sub_queries) > 1

        all_candidates = []
        for sq in sub_queries:
            candidates = retriever.invoke(sq)

            section_match = re.search(r'\b(\d{1,3}[A-Za-z]?)\b', sq)
            if section_match:
                num = section_match.group(1)
                for chunk in chunks:
                    stripped = chunk.page_content.strip()
                    if stripped.startswith(f"{num}.") or f"\n{num}." in chunk.page_content:
                        if chunk not in candidates:
                            candidates.append(chunk)

            all_candidates.extend(candidates)

        seen = set()
        unique_candidates = []
        for c in all_candidates:
            if c.page_content not in seen:
                seen.add(c.page_content)
                unique_candidates.append(c)

        if not unique_candidates:
            return []

        pairs = [[query, doc.page_content] for doc in unique_candidates]
        scores = reranker.predict(pairs)
        ranked = sorted(zip(unique_candidates, scores), key=lambda x: x[1], reverse=True)

        final_k = max(top_k, 2 * len(sub_queries)) if multi_topic else top_k
        return [doc for doc, score in ranked[:final_k]]

    contextualize_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Given the conversation history and a follow-up question, rewrite the "
         "follow-up question as a standalone question that includes all necessary "
         "context from the history. If it's already standalone, return it unchanged. "
         "Output ONLY the rewritten question, nothing else."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}")
    ])
    contextualize_chain = contextualize_prompt | llm | StrOutputParser()

    def get_standalone_question(x):
        if not x.get("chat_history"):
            return x["question"]
        return contextualize_chain.invoke({
            "chat_history": x["chat_history"],
            "question": x["question"]
        })

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a helpful assistant. Answer ONLY using the context below. "
         "Never use any outside knowledge, even if you are confident it is correct. "
         "If the context is incomplete or only partially answers the question, "
         "explicitly say what is missing rather than filling the gap yourself. "
         "If the answer is not in the context, say: I don't know, that information "
         "is not in the document.\n\nContext: {context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}")
    ])

    store = {}
    def get_session_history(session_id):
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]

    inner_chain = (
        {
            "context": lambda x: retrieve_and_rerank(get_standalone_question(x)),
            "question": lambda x: x["question"],
            "chat_history": lambda x: x.get("chat_history", [])
        }
        | prompt | llm | StrOutputParser()
    )

    _log("Pipeline ready.", t0)

    return RunnableWithMessageHistory(
        inner_chain, get_session_history,
        input_messages_key="question",
        history_messages_key="chat_history"
    )


def resolve_api_key(user_provided_key):
    if user_provided_key:
        return user_provided_key, "your own key"
    try:
        return st.secrets["GROQ_API_KEY"], "the demo's default key"
    except Exception:
        return None, None


def invoke_with_retry(chain, inputs, config, max_retries=3):
    last_error = None
    for attempt in range(max_retries):
        try:
            return chain.invoke(inputs, config=config)
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
    return (
        "Sorry, this request hit a temporary error after a few attempts "
        f"(likely a rate limit or connection issue). Please try again in a moment. "
        f"Details: {last_error}"
    )


if build_clicked:
    if not uploaded_file:
        st.sidebar.error("Please upload a PDF before building.")
    else:
        api_key, key_source = resolve_api_key(user_api_key)
        if not api_key:
            st.sidebar.error(
                "No API key provided, and no default key is configured. "
                "Please enter a Groq API key."
            )
        else:
            file_bytes = uploaded_file.getvalue()
            file_hash = hashlib.md5(file_bytes).hexdigest()

            temp_pdf_path = "temp_uploaded.pdf"
            with open(temp_pdf_path, "wb") as f:
                f.write(file_bytes)

            try:
                st.session_state.chain = load_pipeline(api_key, temp_pdf_path, file_hash)
                st.session_state.doc_name = uploaded_file.name
                st.session_state.messages = []
                st.sidebar.success(f"Knowledge base built using {key_source}.")
            except Exception as e:
                st.sidebar.error(f"Something went wrong building the pipeline: {e}")


if "chain" in st.session_state:
    st.caption(f"Currently answering from: **{st.session_state.doc_name}**")

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    user_input = st.chat_input("Ask a question about the document...")
    if user_input:
        st.session_state.messages.append({"role": "human", "content": user_input})
        st.chat_message("human").write(user_input)

        with st.spinner("Thinking..."):
            answer = invoke_with_retry(
                st.session_state.chain,
                {"question": user_input},
                config={"configurable": {"session_id": "streamlit_session"}}
            )

        st.session_state.messages.append({"role": "ai", "content": answer})
        st.chat_message("ai").write(answer)
else:
    st.info("👈 Upload a PDF in the sidebar, then click **Build Knowledge Base** to get started.")
