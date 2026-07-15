import re
import os
import json
import time
import uuid
import hashlib
from collections import Counter
import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import ChatMessageHistory
from sentence_transformers import CrossEncoder

if "HF_TOKEN" in st.secrets:
    os.environ["HF_TOKEN"] = st.secrets["HF_TOKEN"]

st.set_page_config(page_title="Support Agent", page_icon="🤖", initial_sidebar_state="expanded")
st.title("AI Document Q&A Agent")
st.caption("Upload a PDF and ask questions about it — answers are grounded strictly in the document.")
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

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


def format_context(docs):
    if not docs:
        return "No relevant content was found in the document."
    parts = []
    for i, doc in enumerate(docs, start=1):
        parts.append(f"--- Source {i} ---\n{doc.page_content.strip()}")
    return "\n\n".join(parts)


# Dynamic conversational filler detection 
_QUESTION_INDICATORS = {
    "what", "why", "how", "who", "when", "where", "which", "does", "do",
    "did", "is", "are", "was", "were", "can", "could", "would", "should",
    "will", "explain", "tell", "describe", "define", "compare", "summarize",
    "summarise", "list", "give", "show"
}

_FILLER_TERMS = {
    "ok", "okay", "kk", "k", "thanks", "thank", "thankyou", "thx", "ty",
    "cool", "great", "nice", "perfect", "awesome", "sure", "alright",
    "got", "it", "understood", "noted", "appreciate", "appreciated",
    "welcome", "yep", "yeah", "yup", "no", "yes", "please", "good", "fine"
}


def is_conversational_filler(text):
    stripped = text.strip()
    if not stripped:
        return True
    if "?" in stripped:
        return False
    if re.search(r'\d', stripped):
        return False
    words = re.findall(r"[a-zA-Z']+", stripped.lower())
    if not words:
        return True
    if any(w in _QUESTION_INDICATORS for w in words):
        return False
    if len(words) <= 3 and all(w in _FILLER_TERMS for w in words):
        return True
    return False


_META_PATTERNS = [
    r'\babout this (pdf|document|file)\b',
    r'\bsummar(y|ize) (this|the) (pdf|document|file)\b',
    r'\bwhat is this (pdf|document|file)\b',
    r'\bwhat.?s (this|in) (the )?(pdf|document|file) about\b',
    r'\bsummar(y|ize) (the )?whole (ipc|document|pdf|file)\b',
]


def is_meta_question(text):
    lowered = text.strip().lower()
    return any(re.search(p, lowered) for p in _META_PATTERNS)


META_RESPONSE = (
    "I answer specific questions about the document's content rather than "
    "giving a general summary, since I only ever see small relevant excerpts "
    "at a time, not the whole document at once. Try asking about a specific "
    "section, article, or topic instead — for example, \"What does Section "
    "302 say?\" or \"What's the punishment for theft?\""
)

# Fixed, non-negotiable answer used whenever retrieval finds nothing
# relevant. See the grounding fix below for why this is enforced in code
# instead of only via the system prompt.
NO_CONTEXT_ANSWER = "I don't know, that information is not in the document."


@st.cache_resource(show_spinner=False)
def load_models():
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
        temperature=0.0,
        max_tokens=600
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

    def retrieve_and_rerank(query, top_k=3, max_forced=6):
        sub_queries = decompose_query(query)
        multi_topic = len(sub_queries) > 1

     .
        forced_docs = []
        forced_seen = set()
        semantic_pairs = []
        semantic_seen = set()

        for sq in sub_queries:
            candidates = retriever.invoke(sq)
            for c in candidates:
                if c.page_content not in semantic_seen:
                    semantic_seen.add(c.page_content)
                    semantic_pairs.append((c, sq))

            section_numbers = set(re.findall(r'\b(\d{1,3}[A-Za-z]?)\b', sq))
            for num in section_numbers:
                for chunk in chunks:
                    stripped = chunk.page_content.strip()
                    if stripped.startswith(f"{num}.") or f"\n{num}." in chunk.page_content:
                        if chunk.page_content not in forced_seen:
                            forced_seen.add(chunk.page_content)
                            forced_docs.append(chunk)

        forced_docs = forced_docs[:max_forced]
        semantic_pairs = [(d, sq) for (d, sq) in semantic_pairs if d.page_content not in forced_seen]

        if not forced_docs and not semantic_pairs:
            return []

        base_budget = max(top_k, 2 * len(sub_queries)) if multi_topic else top_k
        remaining_budget = max(0, base_budget - len(forced_docs))

        ranked_semantic = []
        if semantic_pairs and remaining_budget > 0:
            pairs = [[sq, doc.page_content] for doc, sq in semantic_pairs]
            scores = reranker.predict(pairs)
            ranked = sorted(
                zip([d for d, _ in semantic_pairs], scores),
                key=lambda x: x[1], reverse=True
            )
            ranked_semantic = [doc for doc, score in ranked[:remaining_budget]]

        return forced_docs + ranked_semantic

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

    def get_standalone_question(question, chat_history):
        if not chat_history:
            return question
        return contextualize_chain.invoke({
            "chat_history": chat_history,
            "question": question
        })

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a helpful assistant. Answer ONLY using the context below. "
         "The context is divided into separate labeled sources (--- Source 1 ---, "
         "--- Source 2 ---, etc.). Each source is an independent excerpt — content "
         "and section/article numbers from DIFFERENT sources must NEVER be mixed "
         "together. Only state that a fact belongs to a specific section or article "
         "number if that exact number and its content appear together WITHIN THE "
         "SAME source block. If you are not sure which source a section number "
         "belongs to, say so explicitly rather than guessing.\n\n"
         "IMPORTANT: The 'Source 1', 'Source 2' labels are for YOUR internal "
         "reference only — NEVER mention them in your answer. Always cite the "
         "real section or article number found written in the text itself, "
         "never the source label.\n\n"
         "Never use any outside knowledge, even if you are confident it is correct. "
         "If the context is incomplete or only partially answers the question, "
         "explicitly say what is missing rather than filling the gap yourself. "
         "If the answer is not in the context, say: I don't know, that information "
         "is not in the document.\n\nContext:\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}")
    ])

    answer_chain = prompt | llm | StrOutputParser()

    store = {}

    def get_session_history(session_id):
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]

    def answer_question(question, session_id):
        history = get_session_history(session_id)
        standalone_question = get_standalone_question(question, history.messages)
        docs = retrieve_and_rerank(standalone_question)

        if not docs:
            answer_text = NO_CONTEXT_ANSWER
        else:
            context = format_context(docs)
            answer_text = invoke_with_retry(
                answer_chain,
                {"context": context, "chat_history": history.messages, "question": question}
            )

        history.add_user_message(question)
        history.add_ai_message(answer_text)
        return answer_text

    _log("Pipeline ready.", t0)
    return answer_question


def resolve_api_key(user_provided_key):
    if user_provided_key:
        return user_provided_key, "your own key"
    try:
        return st.secrets["GROQ_API_KEY"], "the demo's default key"
    except Exception:
        return None, None


def invoke_with_retry(chain, inputs, max_retries=3):
    last_error = None
    for attempt in range(max_retries):
        try:
            return chain.invoke(inputs)
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
                st.session_state.answer_fn = load_pipeline(api_key, temp_pdf_path, file_hash)
                st.session_state.doc_name = uploaded_file.name
                st.session_state.messages = []
                st.sidebar.success(f"Knowledge base built using {key_source}.")
            except Exception as e:
                st.sidebar.error(f"Something went wrong building the pipeline: {e}")


if "answer_fn" in st.session_state:
    st.caption(f"Currently answering from: **{st.session_state.doc_name}**")

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    user_input = st.chat_input("Ask a question about the document...")
    if user_input:
        st.session_state.messages.append({"role": "human", "content": user_input})
        st.chat_message("human").write(user_input)

        if is_conversational_filler(user_input):
            answer = "You're welcome! Let me know if you have another question about the document."
        elif is_meta_question(user_input):
            answer = META_RESPONSE
        else:
            with st.spinner("Thinking..."):
                answer = st.session_state.answer_fn(user_input, st.session_state.session_id)

        st.session_state.messages.append({"role": "ai", "content": answer})
        st.chat_message("ai").write(answer)
else:
    st.info("👈 Upload a PDF in the sidebar, then click **Build Knowledge Base** to get started.")
