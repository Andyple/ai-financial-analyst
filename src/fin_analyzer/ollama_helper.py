import streamlit as st
import ollama

def get_chat_models():
    """
    Gets a list of local Ollama models intended for chat, filtering out embedding models.

    Returns:
        A list of model names. Returns an empty list if connection fails.
    """
    try:
        models = ollama.list()["models"]
        # Filter out models that are likely for embeddings based on name
        return [m["name"] for m in models if 'embed' not in m["name"].lower()]
    except Exception as e:
        st.error(f"Could not connect to Ollama to get local models. Is Ollama running? Error: {e}")
        return []

def get_embedding_models():
    """
    Gets a list of local Ollama models intended for embedding, based on naming convention.

    Returns:
        A list of model names. Returns an empty list if connection fails.
    """
    try:
        models = ollama.list()["models"]
        # Filter for models that are likely for embeddings based on name
        return [m["name"] for m in models if 'embed' in m["name"].lower()]
    except Exception as e:
        st.error(f"Could not connect to Ollama to get local models. Is Ollama running? Error: {e}")
        return []

@st.cache_data(show_spinner="Pulling Ollama model...")
def pull_model(model_name):
    """
    Pulls a model from the Ollama registry.

    Args:
        model_name (str): The name of the model to pull (e.g., 'llama3:latest').

    Returns:
        bool: True if successful, False otherwise.
    """
    if not model_name:
        st.warning("Please enter a model name to pull.")
        return False
    try:
        # This is a generator, so we need to iterate through it to get the status.
        for _ in ollama.pull(model_name, stream=True):
            pass
        st.success(f"Successfully pulled model: {model_name}")
        return True
    except ollama.ResponseError as e:
        st.error(f"Model not found or error pulling model '{model_name}'. Please check the model name. Error: {e.error}")
        return False
    except Exception as e:
        st.error(f"An unexpected error occurred while pulling the model. Error: {e}")
        return False
