import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from google import genai
from google.genai import types
import time 
from backbone_func import load_fitzpdf,chunker,create_vdb,query_with_llm,webloader
if "messages" not in st.session_state:
    st.session_state.messages = []
if 'ready' not in st.session_state:
    st.session_state.ready = False
if 'button' not in st.session_state:
    st.session_state.button = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

def gpt_alike(prompt,real_vector_db,seman_chunker):
    response = query_with_llm(prompt,real_vector_db,seman_chunker,st.session_state.chat_history)
    for words in response.split(" "):
        yield words + " "
        time.sleep(0.06)

st.title('Chat with pdf/Webpage')

with st.sidebar:
    st.title("Upload pdfs or Link")
    uploadedpdf = st.file_uploader(label="Upload Your Pdfs for Extraction",accept_multiple_files=True,type="pdf")

    if uploadedpdf:
        assimple_text = load_fitzpdf(uploaded=uploadedpdf)
    st.markdown(body="OR",width="stretch")
    web_link = st.text_input(label="Paste web page Link")

    if web_link:
         assimple_text = webloader(web_link).page_content

    submit_button = st.button(label="Click to run extraction",type="primary")

    if submit_button:
        if not uploadedpdf and not web_link:
            st.error('Upload any pdf or link to start extraction')    
        else:        
            if st.session_state.button==False:
                st.session_state.button=True


if st.session_state.button==True:
    with st.spinner(text="Processing...",show_time=1):
        if "embeddings" not in st.session_state:
            st.session_state.embeddings=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        # embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        if "seman_chunker" not in st.session_state:
            st.session_state.seman_chunker = chunker(assimple_text,st.session_state.embeddings)
        # seman_chunker = chunker(assimple_text,embeddings)
        if "real_vector_db" not in st.session_state:
            st.session_state.real_vector_db=create_vdb(chunk_list=st.session_state.seman_chunker,embeddings=st.session_state.embeddings)
        # real_vector_db=create_vdb(chunk_list=seman_chunker,embeddings=embeddings)
    if st.session_state.ready == False:
        st.session_state.ready = True
    if st.session_state.ready==True:
        for message in st.session_state.messages:
            with st.chat_message(message['role']):
                st.markdown(message['content'])

        if prompt:= st.chat_input("Chat with RAG"):
            st.chat_message("user").markdown(prompt)
            st.session_state.messages.append({'role':"user","content":prompt})

            with st.chat_message("bot"):
                # response = st.write_stream(query_with_llm(prompt,real_vector_db,seman_chunker))
                response = st.write_stream(gpt_alike(prompt=prompt,real_vector_db=st.session_state.real_vector_db,seman_chunker=st.session_state.seman_chunker))
                st.session_state.messages.append({"role":"bot","content":response})
