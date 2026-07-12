import re
import json
import hashlib
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

st.set_page_config(page_title="Support Agent", page_icon="🤖", initial_sidebar_state="expanded")
st.title("AI Document Q&A Agent")
st.caption("Upload a PDF and ask questions about it — answers are grounded strictly in the document.")

# Sidebar: user provides their own key if user has their own and document
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


def is_noise_chunk(text):
    # Filters out chunks that are mostly bracketed headings / cross-reference
    # index lines with little substantive content, which otherwise confuse
    # the model into narrating its own uncertainty.
    stripped = text.strip()
    return bool(re.match(r'^\(Part\s+[IVX]+.*Arts?\.?\s*[\d\-–—,]+.*\)$', stripped))


@st.cache_resource(show_spinner="Building knowledge base...")
def load_pipeline(api_key, pdf_path, file_hash):
    # file_hash forces a new cache entry whenever the uploaded file's content
    # changes, so swapping documents doesn't silently reuse a stale pipeline.
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=api_key,
        temperature=0.0
    )

    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    chunks = [c for c in chunks if not is_noise_chunk(c.page_content)]

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # In-memory only — keeps each session's vector store isolated so
    # different users on a shared server never mix documents.
    vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings)

    reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    retriever = vectorstore.as_retriever(search_kwargs={"k": 25})

    # Dynamic query decomposition: LLM decides if a question needs
    # multiple separate retrievals (comparisons, multi-part questions),
    # instead of relying on hardcoded regex patterns like "difference between".
    decompose_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Analyze this question. If it asks about TWO OR MORE distinct topics, "
         "articles, or sections that would each need a separate search to answer "
         "well — comparisons, relationships, multi-part questions — break it into "
         "separate standalone search queries, one per topic. If it's a single-topic "
         "question, return it unchanged as the only item.\n\n"
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
                return sub_queries
        except Exception:
            pass
        return [query]

    def retrieve_and_rerank(query, top_k=3):
        sub_queries = decompose_query(query)
        multi_topic = len(sub_queries) > 1

        all_candidates = []
        for sq in sub_queries:
            candidates = retriever.invoke(sq)

            # Hybrid fix: force-include any chunk starting with a referenced
            # section/article number, since plain semantic search often
            # misses exact numeric/ID lookups.
            section_match = re.search(r'\b(\d{1,3}[A-Za-z]?)\b', sq)
            if section_match:
                num = section_match.group(1)
                for chunk in chunks:
                    stripped = chunk.page_content.strip()
                    if stripped.startswith(f"{num}.") or f"\n{num}." in chunk.page_content:
                        if chunk not in candidates:
                            candidates.append(chunk)

            all_candidates.extend(candidates)

        # De-duplicate across sub-query results
        seen = set()
        unique_candidates = []
        for c in all_candidates:
            if c.page_content not in seen:
                seen.add(c.page_content)
                unique_candidates.append(c)

        if not unique_candidates:
            return []

        # Score everything against the ORIGINAL question, not the sub-queries
        # the final answer needs to address exactly what the user asked.
        pairs = [[query, doc.page_content] for doc in unique_candidates]
        scores = reranker.predict(pairs)
        ranked = sorted(zip(unique_candidates, scores), key=lambda x: x[1], reverse=True)

        final_k = max(top_k, 2 * len(sub_queries)) if multi_topic else top_k
        return [doc for doc, score in ranked[:final_k]]

    # Query contextualization: rewrite follow-up questions (with pronouns
    # like "it" or "that") into standalone questions before retrieval. 
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

    # Hardened prompt: explicitly forbids falling back to outside/general
    # knowledge, which was previously leaking into answers on partial matches.
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

    return RunnableWithMessageHistory(
        inner_chain, get_session_history,
        input_messages_key="question",
        history_messages_key="chat_history"
    )


# Resolve which API key to actually use
def resolve_api_key(user_provided_key):
    if user_provided_key:
        return user_provided_key, "your own key"
    try:
        return st.secrets["GROQ_API_KEY"], "the demo's default key"
    except Exception:
        return None, None


# Build step: only runs when the button is clicked 
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


# Main chat area
if "chain" in st.session_state:
    st.caption(f"Currently answering from: **{st.session_state.doc_name}**")

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    user_input = st.chat_input("Ask a question about the document...")
    if user_input:
        st.session_state.messages.append({"role": "human", "content": user_input})
        st.chat_message("human").write(user_input)

        with st.spinner("Thinking..."):
            answer = st.session_state.chain.invoke(
                {"question": user_input},
                config={"configurable": {"session_id": "streamlit_session"}}
            )

        st.session_state.messages.append({"role": "ai", "content": answer})
        st.chat_message("ai").write(answer)
else:
    st.info("👈 Upload a PDF in the sidebar, then click **Build Knowledge Base** to get started.")
