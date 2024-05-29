import streamlit as st

import time
import os

from customllama3 import CustomLlama3

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate

icons = {"assistant": "robot.png", "user": "man-kddi.png"}

# Get the API key from the environment variable
HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN is None:
    st.error("API key not found. Please set the HF_TOKEN secret in your Hugging Face Space.")
    st.stop()

remote_llm = CustomLlama3(bearer_token = HF_TOKEN)

def retrieval_grader(question):
    prompt = PromptTemplate(
        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a grader assessing relevance
        of a retrieved document to a user question. If the document contains keywords related to the user question,
        grade it as relevant. It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. \n
        Provide the binary score as a JSON with a single key 'score' and no premable or explanation.
         <|eot_id|><|start_header_id|>user<|end_header_id|>
        Here is the retrieved document: \n\n {document} \n\n
        Here is the user question: {question} \n <|eot_id|><|start_header_id|>assistant<|end_header_id|>
        """,
        input_variables=["question", "document"]
    )
    
    retrieval_grader = prompt | remote_llm | JsonOutputParser()
    
    # Example usage
    document = "Apples are rich in vitamins and fiber."
    
    result = retrieval_grader.invoke({
        "question": question,
        "document": document
    })
    
    return result["score"]


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
            print("Processing files")
                
            if uploaded_file:
                print("File uploaded")
        
            if web_url:
                print("Link uploaded:"+web_url)

            st.success("Done")
    st.text_area("Console", st.session_state["console_out"])

user_prompt = st.chat_input("Ask me anything about the content of the PDF or Web Link:")

if user_prompt: # and (uploaded_file or video_url)
    st.session_state.messages.append({'role': 'user', "content": user_prompt})
    with st.chat_message("user", avatar="man-kddi.png"):
        st.write(user_prompt)

    # Trigger assistant's response retrieval and update UI
    with st.spinner("Thinking..."):
        response = retrieval_grader(user_prompt)
        st.session_state["console_out"] += "retrieval_grader" + user_prompt + "\n"
    with st.chat_message("user", avatar="robot.png"):
        st.write_stream(streamer(response))
    st.session_state.messages.append({'role': 'assistant', "content": response})

    st.experimental_rerun()