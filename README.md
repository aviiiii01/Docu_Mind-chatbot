# DocuMind — AI-Powered PDF Chatbot

**DocuMind** is an AI-powered chatbot that transforms your PDFs into a conversational knowledge base using Google Gemini, LangChain, and FAISS vector search.

## Features
- 📄 Multi-PDF upload with drag & drop
- 🧠 Semantic search via FAISS + HuggingFace embeddings
- 💬 Conversational memory across questions
- ✨ Premium dark glassmorphism UI
- ⚡ Powered by Google Gemini 2.5 Flash Lite

## Setup

### 1. Clone & navigate
```bash
git clone https://github.com/aviiiii01/Docu_Mind-chatbot.git
cd Docu_Mind-chatbot
```

### 2. Create virtual environment
```bash
python -m venv myenv
source myenv/bin/activate   # Linux/macOS
myenv\Scripts\activate      # Windows
```

### 3. Install dependencies
```bash
pip install -r requirement.txt
```

### 4. Add your Gemini API key
Create a `.env` file in the project root:
```
GEMINI_API_KEY=your_api_key_here
```
Get a free key at: https://aistudio.google.com/apikey

### 5. Run the app
```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```
Open **http://localhost:8000** in your browser.

## Legacy Streamlit UI
The original Streamlit version is still available:
```bash
streamlit run main.py
```

## Tech Stack
| Layer | Technology |
|-------|-----------|
| LLM | Google Gemini 2.5 Flash Lite |
| Orchestration | LangChain |
| Vector Store | FAISS |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` |
| Backend | FastAPI + Uvicorn |
| Frontend | Vanilla HTML/CSS/JS |

## License
MIT
