
import subprocess
import sys

# Auto install all required packages
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
    "langchain", "langchain-groq", "langchain-community",
    "langchain-core", "langchain-text-splitters",
    "chromadb", "sentence-transformers", "pypdf", "python-dotenv"
], check=True)

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

st.set_page_config(page_title="Support Agent", page_icon="🤖")
st.title("AI Customer Support Agent")

@st.cache_resource
def load_pipeline():
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=st.secrets["GROQ_API_KEY"],
        temperature=0.0
    )

    loader = PyPDFLoader("PYTHON - tkinter.pdf")  # keep this PDF in your repo root
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(
        documents=chunks, embedding=embeddings, persist_directory="./chroma_db"
    )

    # NEW: re-ranker, loaded once via cache_resource
    reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

    # CHANGED: k=10 instead of k=3, since re-ranker narrows it down after
    retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

    # NEW: re-ranking function
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
            "context": lambda x: retrieve_and_rerank(x["question"]),  # CHANGED from retriever.invoke(...)
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

chain = load_pipeline()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

user_input = st.chat_input("Ask a question about the document...")
if user_input:
    st.session_state.messages.append({"role": "human", "content": user_input})
    st.chat_message("human").write(user_input)

    answer = chain.invoke(
        {"question": user_input},
        config={"configurable": {"session_id": "streamlit_session"}}
    )
    st.session_state.messages.append({"role": "ai", "content": answer})
    st.chat_message("ai").write(answer)
