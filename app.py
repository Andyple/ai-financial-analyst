import streamlit as st

ollama_ip = "http://localhost:11434"

with st.sidebar:
    ollama_ip = st.text_area(
        "Ollama IP",
        "http://localhost:11434")
    st.file_uploader(
        "Upload 10-K File"
    )

st.text(ollama_ip)
