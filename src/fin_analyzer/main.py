import streamlit as st
import os
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import OllamaEmbeddings, HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from .ollama_helper import get_chat_models, get_embedding_models, pull_model

# --- App Configuration ---
st.set_page_config(page_title="AI Financial Analyst", page_icon="📈", layout="wide")

# --- Helper Functions ---

@st.cache_resource(show_spinner="Processing document...")
def get_vectorstore_from_file(file, _embedding_function):
    if file is None or _embedding_function is None:
        return None
    try:
        temp_dir = "temp_files"
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        
        file_path = os.path.join(temp_dir, file.name)
        with open(file_path, "wb") as f:
            f.write(file.getvalue())

        file_extension = os.path.splitext(file.name)[1].lower()
        if file_extension == ".pdf":
            loader = PyPDFLoader(file_path)
        elif file_extension == ".txt":
            loader = TextLoader(file_path)
        else:
            st.error(f"Unsupported file type: {file_extension}.")
            return None
        
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        document_chunks = text_splitter.split_documents(documents)
        vector_store = Chroma.from_documents(document_chunks, _embedding_function)
        return vector_store
    except Exception as e:
        st.error(f"An error occurred during document processing: {e}")
        return None
    finally:
        if 'file_path' in locals() and os.path.exists(file_path):
            os.remove(file_path)

def get_rag_chain(vector_store, llm):
    if vector_store is None or llm is None:
        return None
    retriever = vector_store.as_retriever()
    prompt_template = """
    You are an expert financial analyst. Answer the user's question based on the provided context.
    If the context does not contain the answer, state that you cannot find the information in the document.
    Provide a concise and accurate answer.
    Context: {context}
    Question: {input}
    Answer:
    """
    prompt = ChatPromptTemplate.from_template(prompt_template)
    document_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, document_chain)

# --- Session State Initialization ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "document_processed" not in st.session_state:
    st.session_state.document_processed = False
if "llm" not in st.session_state:
    st.session_state.llm = None
if "embedding_function" not in st.session_state:
    st.session_state.embedding_function = None

# --- UI Rendering ---

# --- Sidebar (Control Panel) ---
with st.sidebar:
    st.title("⚙️ Control Panel")

    if st.button("Start New Chat"):
        st.session_state.chat_history = []
        st.rerun()

    with st.expander("📚 Document Processing", expanded=True):
        uploaded_file = st.file_uploader("Upload a 10-K Report", type=["pdf", "txt"])
        if st.button("Process Document"):
            if uploaded_file is None:
                st.error("Please upload a document.")
            elif st.session_state.llm is None or st.session_state.embedding_function is None:
                st.error("Please configure your models before processing.")
            else:
                vector_store = get_vectorstore_from_file(uploaded_file, st.session_state.embedding_function)
                if vector_store:
                    st.session_state.rag_chain = get_rag_chain(vector_store, st.session_state.llm)
                    st.session_state.document_processed = True
                    st.session_state.chat_history = []
                    st.success("Document processed successfully!")
                else:
                    st.error("Document processing failed.")
        
        if st.session_state.document_processed:
            st.success("✅ Ready to chat")
        else:
            st.info("Awaiting document processing.")

    with st.expander("🤖 Model Configuration"):
        st.subheader("Chat Model")
        chat_provider = st.selectbox("Provider", ["Ollama (Local)", "OpenAI (Cloud API)"], key="chat_provider")
        
        if chat_provider == "Ollama (Local)":
            chat_models = get_chat_models()
            if chat_models:
                ollama_model = st.selectbox("Select Model", chat_models, index=chat_models.index("llama3") if "llama3" in chat_models else 0)
                try:
                    st.session_state.llm = ChatOllama(model=ollama_model)
                    st.success(f"Using Ollama: {ollama_model}")
                except Exception as e:
                    st.error(f"Failed to connect to Ollama. Is it running? Error: {e}")
            else:
                st.warning("No local Ollama chat models found. Please pull a model.")

        elif chat_provider == "OpenAI (Cloud API)":
            openai_api_key = st.text_input("OpenAI API Key", type="password")
            if openai_api_key:
                openai_model = st.selectbox("Select Model", ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"])
                try:
                    st.session_state.llm = ChatOpenAI(api_key=openai_api_key, model_name=openai_model)
                    st.success(f"Using OpenAI: {openai_model}")
                except Exception as e:
                    st.error(f"Connection to OpenAI failed. Check API key. Error: {e}")

        st.subheader("Embedding Model")
        embedding_provider = st.selectbox("Provider", ["Ollama (Local)", "OpenAI (Cloud API)", "HuggingFace (Local)"], key="embedding_provider")

        if embedding_provider == "Ollama (Local)":
            embedding_models = get_embedding_models()
            if embedding_models:
                ollama_embed_model = st.selectbox("Select Model", embedding_models, index=embedding_models.index("nomic-embed-text") if "nomic-embed-text" in embedding_models else 0, key="ollama_embed_select")
                try:
                    st.session_state.embedding_function = OllamaEmbeddings(model=ollama_embed_model)
                    st.success(f"Using Ollama embedding: {ollama_embed_model}")
                except Exception as e:
                    st.error(f"Failed to use Ollama embedding. Is it running? Error: {e}")
            else:
                st.warning("No local Ollama embedding models found.")

        elif embedding_provider == "OpenAI (Cloud API)":
            openai_api_key_embed = st.text_input("OpenAI API Key", type="password", key="openai_embed_key")
            if openai_api_key_embed:
                openai_embed_model = st.selectbox("Select Model", ["text-embedding-3-large", "text-embedding-3-small", "text-embedding-ada-002"])
                st.session_state.embedding_function = OpenAIEmbeddings(api_key=openai_api_key_embed, model=openai_embed_model)
                st.success(f"Using OpenAI embedding: {openai_embed_model}")
        
        elif embedding_provider == "HuggingFace (Local)":
            hf_embed_model = st.selectbox("Select Model", ["sentence-transformers/all-MiniLM-L6-v2", "sentence-transformers/all-mpnet-base-v2"])
            st.session_state.embedding_function = HuggingFaceEmbeddings(model_name=hf_embed_model)
            st.success(f"Using HuggingFace embedding: {hf_embed_model}")

    with st.expander("📥 Pull Ollama Models"):
        model_to_pull = st.text_input("Enter model name to pull (e.g., 'gemma3:4b')")
        if st.button("Pull Model"):
            if pull_model(model_to_pull):
                st.rerun() # Rerun to update the model lists

# --- Main Page (Chat Interface) ---
st.title("📈 AI Financial Report Analyst")
st.markdown("Welcome! Configure your models and upload a 10-K report in the sidebar to begin.")

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            with st.expander("View Sources"):
                for source in message["sources"]:
                    st.info(f"Source: Page {source.metadata.get('page', 'N/A')}\n\n{source.page_content}")

# Example Prompts
if not st.session_state.chat_history and st.session_state.document_processed:
    st.markdown("---")
    st.write("**Example Prompts:**")
    example_prompts = [
        "What were the total revenues for the last fiscal year?",
        "Summarize the main risk factors for the company.",
        "What is the company's net income?",
        "Who are the key executives of the company?",
        "What are the primary business segments?"
    ]
    
    cols = st.columns(len(example_prompts))
    for i, prompt in enumerate(example_prompts):
        if cols[i].button(prompt, use_container_width=True):
            st.session_state.user_query = prompt
            st.rerun()

# Chat input
user_query = st.chat_input("Ask a question...", disabled=not st.session_state.document_processed, key="main_chat_input")
if "user_query" in st.session_state and st.session_state.user_query:
    user_query = st.session_state.user_query
    st.session_state.user_query = None

if user_query:
    st.session_state.chat_history.append({"role": "user", "content": user_query})
    
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            if st.session_state.rag_chain:
                response = st.session_state.rag_chain.invoke({"input": user_query})
                answer = response.get("answer", "No answer found.")
                sources = response.get("context", [])
                
                st.markdown(answer)
                
                if sources:
                    with st.expander("View Sources"):
                        for source in sources:
                            st.info(f"Source: Page {source.metadata.get('page', 'N/A')}\n\n{source.page_content}")
                
                st.session_state.chat_history.append({
                    "role": "assistant", 
                    "content": answer, 
                    "sources": sources
                })
            else:
                st.error("RAG chain not initialized. Please process a document.")