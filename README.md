# BDL ChatBot (Internship Project)

A completely offline, secure, multi-user Retrieval-Augmented Generation (RAG) chatbot designed for the Bharat Dynamics Limited (BDL) office environment. It allows users to query uploaded documents locally without sending sensitive data to external servers.

## Features

- **Multi-User Local Server**: Supports up to 5 concurrent users and 1 admin session simultaneously.
- **Role-Based Access**: 
  - **Users** can chat with the AI and ask questions about the indexed documents.
  - **Admins** have access to a dashboard to upload documents, manage indexed files, and view system-wide chat history.
- **Privacy-First (100% Offline)**: Uses Ollama to run Large Language Models (LLMs) entirely locally on your hardware.
- **Fast & Accurate Document Retrieval**: Built with PyMuPDF for high-speed PDF extraction and LanceDB for hybrid (semantic + keyword) vector search.
- **Web Interface**: Accessible from any browser on the local network. No client installation required.

## Prerequisites

1. **Python 3.10+**
2. **Ollama**: Must be installed and running.
   ```bash
   # Pull the required models before starting
   ollama pull gemma4:e2b
   ollama pull nomic-embed-text
   ```

## Setup & Installation

1. Clone or download the project.
2. Install the required Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. (Optional) Configure the environment variables in the `.env` file:
   - `MODEL_NAME`: The Ollama model to use (default: `gemma4:e2b`).
   - `SERVER_PORT`: Port for the web server (default: `8550`).
   - `MAX_USERS`, `MAX_ADMINS`: Adjust concurrent connection limits.

## How to Run

**Option 1: Using the batch script (Windows)**
Simply double-click the `start_server.bat` file. It will check if Ollama is running and start the server automatically.

**Option 2: Using the command line**
```bash
# Start the web server (accessible on local network)
python scripts/frontend.py

# Alternatively, run as a local desktop app (single user)
python scripts/frontend.py --desktop
```

Once running, the terminal will display the local and network URLs (e.g., `http://192.168.x.x:8550`). Share the network URL with colleagues so they can connect via their web browsers.

## Usage

- **Default Admin Password**: `Admin321!` (can be changed in `scripts/frontend.py`)
- Users can log in, select specific documents to focus on, and ask questions.
- Admins can upload PDFs, TXTs, or MD files, which the system will automatically chunk, embed, and index for querying.
