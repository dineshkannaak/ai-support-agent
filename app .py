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

st.set_page_config(
    page_title="Support Agent",
    page_icon="🤖",
    initial_sidebar_state="expanded"
)
st.title("AI Customer Support Agent")
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


@st.cache_resource(show_spinner="Building knowledge base...")
def load_pipeline(api_key, pdf_path, file_hash):
    # file_hash is unused inside the function body, but including it as an
    # argument forces Streamlit to treat a new/changed file as a NEW cache
    # entry instead of reusing a stale pipeline built from a previous file.
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=api_key,
        temperature=0.0
    )

    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # NOTE: no persist_directory here on purpose — this keeps each session's
    # vector store fully in-memory, so different users on a shared Streamlit
    # Cloud instance never end up reading each other's documents.
    vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings)

    reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

    def retrieve_and_rerank(query, top_k=3):
        candidates = retriever.invoke(query)
        if not candidates:
            return []
        pairs = [[query, doc.page_content] for doc in candidates]
        scores = reranker.predict(pairs)
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, score in ranked[:top_k]]

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a helpful assistant. Answer ONLY using the context below. "
         "If not in context say: I don't know.\n\nContext: {context}"),
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
            "context": lambda x: retrieve_and_rerank(x["question"]),
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


# ── Resolve which API key to actually use ──
def resolve_api_key(user_provided_key):
    if user_provided_key:
        return user_provided_key, "your own key"
    try:
        return st.secrets["GROQ_API_KEY"], "the demo's default key"
    except Exception:
        return None, None


# ── Build step: only runs when the button is clicked ──
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
            file_hash = hashlib.md5(file_bytes).hexdigest()  # changes whenever the file content changes

            temp_pdf_path = "temp_uploaded.pdf"
            with open(temp_pdf_path, "wb") as f:
                f.write(file_bytes)

            try:
                st.session_state.chain = load_pipeline(api_key, temp_pdf_path, file_hash)
                st.session_state.doc_name = uploaded_file.name
                st.session_state.messages = []  # fresh conversation for the new document
                st.sidebar.success(f"Knowledge base built using {key_source}.")
            except Exception as e:
                st.sidebar.error(f"Something went wrong building the pipeline: {e}")


# ── Main chat area ──
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
    st.info("👈 Upload a PDF in the sidebar, then click **Build Knowledge Base** to get started.") sidebar to get started.")
