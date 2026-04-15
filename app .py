
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
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import ChatMessageHistory
import os
import tempfile

st.set_page_config(
    page_title="AI Document Assistant",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 AI Document Assistant")
st.caption("Upload any PDF and chat with it using AI")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "chain" not in st.session_state:
    st.session_state.chain = None

if "store" not in st.session_state:
    st.session_state.store = {}

with st.sidebar:
    st.header("Setup")

    groq_key = st.text_input(
        "Groq API Key",
        type="password",
        placeholder="paste your gsk_ key here"
    )

    uploaded_file = st.file_uploader(
        "Upload your PDF",
        type="pdf"
    )

    if st.button("Build AI from PDF", type="primary"):
        if not groq_key:
            st.error("Please enter your Groq API key first")
        elif not uploaded_file:
            st.error("Please upload a PDF file first")
        else:
            with st.spinner("Reading PDF..."):
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=".pdf"
                ) as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name
                loader = PyPDFLoader(tmp_path)
                docs = loader.load()

            with st.spinner("Splitting into chunks..."):
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=500,
                    chunk_overlap=50
                )
                chunks = splitter.split_documents(docs)

            with st.spinner("Building AI brain... (1-2 mins first time)"):
                embeddings = HuggingFaceEmbeddings(
                    model_name="all-MiniLM-L6-v2"
                )
                vectorstore = Chroma.from_documents(
                    documents=chunks,
                    embedding=embeddings,
                    persist_directory="./chroma_db"
                )
                retriever = vectorstore.as_retriever(
                    search_kwargs={"k": 3}
                )

            with st.spinner("Setting up chat chain..."):
                llm = ChatGroq(
                    model="llama-3.1-8b-instant",
                    api_key=groq_key,
                    temperature=0.0
                )

                prompt = ChatPromptTemplate.from_messages([
                    ("system",
                     "You are a helpful assistant. "
                     "Answer ONLY using the context below. "
                     "If the answer is not in the context say: "
                     "I don't know, that information is not in the document.\n\n"
                     "Context: {context}"),
                    MessagesPlaceholder(variable_name="chat_history"),
                    ("human", "{question}")
                ])

                inner_chain = (
                    {
                        "context": lambda x: retriever.invoke(x["question"]),
                        "question": lambda x: x["question"],
                        "chat_history": lambda x: x.get("chat_history", [])
                    }
                    | prompt
                    | llm
                    | StrOutputParser()
                )

                def get_session_history(session_id):
                    if session_id not in st.session_state.store:
                        st.session_state.store[session_id] = ChatMessageHistory()
                    return st.session_state.store[session_id]

                st.session_state.chain = RunnableWithMessageHistory(
                    inner_chain,
                    get_session_history,
                    input_messages_key="question",
                    history_messages_key="chat_history"
                )

            st.success(f"Ready! Loaded {len(docs)} pages, {len(chunks)} chunks.")
            st.session_state.messages = []

if st.session_state.chain is None:
    st.info("Enter your Groq API key, upload a PDF, and click Build AI from PDF to start.")
else:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if question := st.chat_input("Ask anything about your PDF..."):
        st.session_state.messages.append({
            "role": "user",
            "content": question
        })
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = st.session_state.chain.invoke(
                    {"question": question},
                    config={"configurable": {"session_id": "main"}}
                )
            st.markdown(answer)

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer
        })
