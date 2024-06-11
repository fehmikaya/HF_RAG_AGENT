import streamlit as st

import time
import shutil
import os

from customllama3 import CustomLlama3
from ragagent import RAGAgent

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


icons = {"assistant": "robot.png", "user": "man-kddi.png"}
DATA_DIR = "data"
# Ensure data directory exists
os.makedirs(DATA_DIR, exist_ok=True)

if not hasattr(st, 'agent'):
  st.agent = "None"

def init_agent_with_docs():

    docs=[]

    if os.path.exists(DATA_DIR):
        try:
            pdf_loader = PyPDFDirectoryLoader(DATA_DIR)
            pdf_docs = pdf_loader.load()
            if pdf_docs:
                docs.append(pdf_docs)
                st.session_state["console_out"] += "Pdf's loaded\n"
        except Exception as e:
            st.error("PyPDFLoader Exception: " + e)
    return RAGAgent(docs)

def remove_old_files():
    st.session_state["console_out"] += "remove_old_files\n"
    shutil.rmtree(DATA_DIR)
    os.makedirs(DATA_DIR)

def streamer(text):
    for i in text:
        yield i
        time.sleep(0.02)

if "console_out" not in st.session_state:
    st.session_state["console_out"] = ""

# Streamlit app initialization
st.title("RAG AGENT")
st.markdown("RAG Agent with PDF and Web Search (Langchain & Langgraph)")
st.markdown("Finds most related content from given sources with Sanity/Hallucination checks")

if 'messages' not in st.session_state:
    st.session_state.messages = [{'role': 'assistant', "content": "Hello! Upload PDF's and ask me anything about the content."}]

for message in st.session_state.messages:
    with st.chat_message(message['role'], avatar=icons[message['role']]):
        st.write(message['content'])

with st.sidebar:
    uploaded_files = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", type="pdf", accept_multiple_files=True)

    if st.button("Submit & Process"):
        with st.spinner("Processing..."):
            st.session_state["console_out"] = ""
            if len(os.listdir(DATA_DIR)) !=0:
                remove_old_files()  
            for index, file in zip(range(0,uploaded_files),uploaded_files):
                filepath = DATA_DIR+"/saved_pdf_"+index+".pdf"
                with open(filepath, "wb") as f:
                    f.write(file.getbuffer())
            st.agent = init_agent_with_docs()
            st.success("Done")
    st.text_area("Console", st.session_state["console_out"])

user_prompt = st.chat_input("Ask me anything about the content of the PDF or Web Link:")

if user_prompt and (uploaded_file or web_url):
    st.session_state.messages.append({'role': 'user', "content": user_prompt})
    response = "Could not find an answer."
    with st.chat_message("user", avatar="man-kddi.png"):
        st.write(user_prompt)

    # Trigger assistant's response retrieval and update UI
    with st.spinner("Thinking..."):
        inputs = {"question": user_prompt}
        for output in st.agent.app.stream(inputs):
            for key, value in output.items():
                if "generation" in value:
                    response = value["generation"]
    with st.chat_message("user", avatar="robot.png"):
        st.write_stream(streamer(response))
    st.session_state.messages.append({'role': 'assistant', "content": response})

    st.rerun()