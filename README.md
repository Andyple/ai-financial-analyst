# AI Financial Report Analyst

This project is an interactive web application built with Streamlit that allows users to analyze financial documents (like 10-K reports) using the power of Large Language Models (LLMs).

The application leverages the Retrieval-Augmented Generation (RAG) technique to provide accurate, context-aware answers based on the content of the uploaded document. Users can interact with a local LLM via Ollama or connect to the OpenAI API.

## Features

- **Interactive Chat Interface:** Ask questions about your financial document in a natural, conversational way.
- **Flexible Model Selection:**
    - **Chat Models:** Choose between local models served by Ollama (e.g., Llama 3, Mistral) or cloud models from OpenAI (e.g., GPT-4o).
    - **Embedding Models:** Select from various local Ollama or HuggingFace sentence-transformer models, or use OpenAI's embedding APIs.
- **Dynamic Ollama Integration:**
    - Automatically detects and lists your locally installed Ollama models.
    - Pull new models from the Ollama registry directly through the UI.
- **Document Support:** Upload and analyze both PDF (`.pdf`) and Text (`.txt`) files.
- **Source Verification:** Every answer is linked back to the specific text chunks from the source document, allowing for easy verification.
- **User-Friendly UI:** A clean, organized control panel allows for easy configuration, and example prompts help users get started quickly.
- **Chat History Management:** Start a new chat at any time to clear the conversation.

## Setup and Installation

Follow these steps to set up and run the project locally.

### Prerequisites

- **Python 3.9+**
- **Ollama** installed and running on your local machine. You can download it from [ollama.com](https://ollama.com/).

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd Financial-Report-Analyst
```

### 2. Set Up Your Python Environment

You can use either Conda or a standard Python virtual environment (`venv`). Conda is recommended for easier dependency management.

<details>
<summary><strong>Option 1: Using Conda (Recommended)</strong></summary>

```bash
# Create a new Conda environment
conda create --name financial-analyst python=3.11

# Activate the environment
conda activate financial-analyst
```
</details>

<details>
<summary><strong>Option 2: Using venv</strong></summary>

```bash
# Create a new virtual environment
python -m venv venv

# Activate the environment
# On Windows:
# .\venv\Scripts\activate
# On macOS/Linux:
# source venv/bin/activate
```
</details>

### 3. Install Required Packages

Once your environment is activated, install the dependencies from `requirements.txt`.

```bash
pip install -r requirements.txt
```

### 4. Pull Ollama Models

For the application to work with local models, you need to have at least one chat model and one embedding model pulled in Ollama.

You can do this via the command line:

```bash
# Pull a recommended chat model
ollama pull llama3

# Pull a recommended embedding model
ollama pull nomic-embed-text
```

Alternatively, you can use the "Pull Ollama Models" feature in the application's UI after launching it.

## How to Run the Application

1.  Make sure your Python environment (`conda` or `venv`) is activated.
2.  Run the Streamlit app from your terminal:
    ```bash
    streamlit run app.py
    ```
The application should now be open and accessible in your web browser.

## How to Use the Application

1.  **Configure Your Models:**
    - In the sidebar, open the "Model Configuration" section.
    - Select your desired **Chat Model** and **Embedding Model** from the available providers and dropdowns.
    - If using OpenAI, you will need to enter your API key.
2.  **Process a Document:**
    - In the "Document Processing" section, upload a PDF or TXT file.
    - Click the **"Process Document"** button. The app will create and cache a vector store from the document's content.
3.  **Start Chatting:**
    - Once the document is processed, the chat input area will become active.
    - You can click on one of the **example prompts** to get started or type your own question into the chat box.
4.  **Start a New Chat:**
    - To clear the conversation and start fresh with the same document, click the **"Start New Chat"** button at the top of the sidebar.