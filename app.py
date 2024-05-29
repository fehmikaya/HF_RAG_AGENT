import streamlit as st

import time
import shutil
import os

from customllama3 import CustomLlama3

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


# Get the API key from the environment variable

HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN is None:
    st.error("API key not found. Please set the HF_TOKEN secret in your Hugging Face Space.")
    st.stop()
icons = {"assistant": "robot.png", "user": "man-kddi.png"}
DATA_DIR = "data"
# Ensure data directory exists
os.makedirs(DATA_DIR, exist_ok=True)

if not hasattr(st, 'remote_llm'):
  st.remote_llm = CustomLlama3(bearer_token = HF_TOKEN)

if not hasattr(st, 'retriever'):
  st.retriever = "None"

def data_ingestion():

    docs=[]
    
    if os.path.exists(os.path.join(DATA_DIR, "saved_link.txt")):
        try:
            with open(os.path.join(DATA_DIR, "saved_link.txt"), 'r') as file:
                url = file.read()
                st.session_state["console_out"] += "URL: " + url + "\n"
                web_doc = WebBaseLoader(url).load()
                if web_doc:
                    docs.append(web_doc)
                    st.session_state["console_out"] += "Web link loaded: " + url + "\n"
        except Exception as e:
            st.error("WebBaseLoader Exception: " + e)

    if os.path.exists(os.path.join(DATA_DIR, "saved_pdf.pdf")):
        try:
            pdf_loader = PyPDFLoader(os.path.join(DATA_DIR, "saved_pdf.pdf"))
            pdf_doc = pdf_loader.load()
            if pdf_doc:
                docs.append(pdf_doc)
                st.session_state["console_out"] += "Pdf loaded\n"
        except Exception as e:
            st.error("PyPDFLoader Exception: " + e)
    
    docs_list = [item for sublist in docs for item in sublist]
    
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=250, chunk_overlap=0
    )
    doc_splits = text_splitter.split_documents(docs_list)
    
    embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Add to vectorDB
    vectorstore = Chroma.from_documents(
        documents=doc_splits,
        collection_name="rag-chroma",
        embedding=embedding_function,
    )
    return vectorstore.as_retriever()

def remove_old_files():
    shutil.rmtree(DATA_DIR)
    os.makedirs(DATA_DIR)

def retrieval_grader(question):

    # Prompt
    prompt = PromptTemplate(
        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are an assistant for question-answering tasks. 
        Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. 
        Use five sentences maximum and keep the answer concise <|eot_id|><|start_header_id|>user<|end_header_id|>
        Question: {question} 
        Context: {context} 
        Answer: <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
        input_variables=["question", "context"],
    )
    
    # Chain
    rag_chain = prompt | st.remote_llm | StrOutputParser()
    
    # Run
    docs = st.retriever.invoke(question)
    generation = rag_chain.invoke({"context": "\n\n".join(doc.page_content for doc in docs), "question": question})
    return generation


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
    st.session_state.messages = [{'role': 'assistant', "content": 'Hello! Upload a PDF/Web link and ask me anything about the content.'}]

for message in st.session_state.messages:
    with st.chat_message(message['role'], avatar=icons[message['role']]):
        st.write(message['content'])

with st.sidebar:
    uploaded_file = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button")
    web_url = st.text_input("Web Link: ")
    if st.button("Submit & Process"):
        with st.spinner("Processing..."):
            st.session_state["console_out"] = ""
            if len(os.listdir(DATA_DIR)) !=0:
                remove_old_files()  
            if uploaded_file:
                filepath = DATA_DIR+"/saved_pdf.pdf"
                with open(filepath, "wb") as f:
                    f.write(uploaded_file.getbuffer())
            if web_url:
                with open(DATA_DIR+"/saved_link.txt", "w") as file:
                    file.write(web_url)
            st.retriever = data_ingestion()
            st.success("Done")
    st.text_area("Console", st.session_state["console_out"])

user_prompt = st.chat_input("Ask me anything about the content of the PDF or Web Link:")

if user_prompt and (uploaded_file or web_url):
    st.session_state.messages.append({'role': 'user', "content": user_prompt})
    with st.chat_message("user", avatar="man-kddi.png"):
        st.write(user_prompt)

    # Trigger assistant's response retrieval and update UI
    with st.spinner("Thinking..."):
        response = retrieval_grader(user_prompt)
    with st.chat_message("user", avatar="robot.png"):
        st.write_stream(streamer(response))
    st.session_state.messages.append({'role': 'assistant', "content": response})

    st.rerun()