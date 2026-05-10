# Internship Project for BDL

Our team's repo: https://github.com/Souravhmmm/BDL-CHATBOT (main repo is private, this is a public version with confidential things removed)

## Setup Instructions

### 1. Clone the Repository

```bash
git clone <repository-url>
cd BDL-RAGbot
```

### 2. Configure Environment Variables

```bash
# Copy the example file
cp .env.example .env

# Edit .env with your actual configuration
# IMPORTANT: Never commit .env to git (it's in .gitignore)
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Application

```bash
python scripts/frontend.py
```

## Configuration

Edit the `.env` file to customize:

- **MODEL_NAME**: Ollama model to use (default: gemma4:e2b)
- **NUM_GPU**: Number of GPUs (0 for CPU-only)
- **NUM_THREAD**: Number of processing threads
- **ADMIN_PASSWORD**: Secure admin access password

See `.env.example` for all available options.

## Security Notice

⚠️ **Important Security Practices:**

- **Never commit `.env` file** - it's automatically ignored by git
- **Never hardcode credentials** in source code
- **Always use `.env.example`** to document required variables
- **Regenerate ADMIN_PASSWORD** in production environments
- **Keep chat history private** - not committed to version control
- **Database files are excluded** from git for privacy
