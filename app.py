import streamlit as st

import time

icons = {"assistant": "robot.png", "user": "man-kddi.png"}

def streamer(text):
    for i in text:
        yield i
        time.sleep(0.005)

st.session_state.console_out=""

# Streamlit app initialization
st.title("RAG AGENT")
st.markdown("RAG Agent with PDF and Web Search")

if 'messages' not in st.session_state:
    st.session_state.messages = [{'role': 'assistant', "content": 'Hello! Upload a PDF/Youtube Video link and ask me anything about the content.'}]

for message in st.session_state.messages:
    with st.chat_message(message['role'], avatar=icons[message['role']]):
        st.write(message['content'])

with st.sidebar:
    st.title("Menu:")
    uploaded_file = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button")
    video_url = st.text_input("Enter Youtube Video Link: ")
    if st.button("Submit & Process"):
        with st.spinner("Processing..."):
            print("Processing files")
                
            if uploaded_file:
                print("File uploaded")
        
            if video_url:
                print("Link uploaded:"+video_url)

            st.success("Done")
    st.markdown("Console:")
    with st.container(height=200):
        st.text_area("Console",value=st.session_state.console_out, height=2000)

user_prompt = st.chat_input("Ask me anything about the content of the PDF or Web Link:")

if user_prompt and (uploaded_file or video_url):
    st.session_state.messages.append({'role': 'user', "content": user_prompt})
    with st.chat_message("user", avatar="man-kddi.png"):
        st.write(user_prompt)

    # Trigger assistant's response retrieval and update UI
    with st.spinner("Thinking..."):
        response = "I have an answer coming soon..."
        st.session_state.console_out=st.session_state.console_out+response+"\n"
    with st.chat_message("user", avatar="robot.png"):
        st.write_stream(streamer(response))
    st.session_state.messages.append({'role': 'assistant', "content": response})