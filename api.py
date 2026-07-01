from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import os, io, logging, warnings
import uvicorn

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

load_dotenv()

app = FastAPI(title="DocuMind API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Pre-load embedding model at startup (avoids crash on first upload) ─────────
logger.info("Loading embedding model — this may take a moment on first run...")
_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
logger.info("Embedding model loaded ✅")

# ── In-memory session state ───────────────────────────────────────────────────
_state = {
    "vectorstore": None,
    "memory": None,
    "uploaded_files": [],
    "processing": False,
}

def _reset_memory():
    _state["memory"] = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        chat_memory=ChatMessageHistory(),
    )

_reset_memory()


# ── Helpers ───────────────────────────────────────────────────────────────────
def extract_text_from_bytes(file_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(file_bytes))
    return "".join(page.extract_text() or "" for page in reader.pages)


def build_vectorstore(raw_text: str):
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
    )
    chunks = splitter.split_text(raw_text)
    if not chunks:
        raise ValueError("No text chunks generated from documents.")
    return FAISS.from_texts(texts=chunks, embedding=_embeddings)


# ── Schemas ───────────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    question: str

class ChatResponse(BaseModel):
    answer: str
    sources: List[str] = []

class StatusResponse(BaseModel):
    ready: bool
    file_count: int
    files: List[str]
    processing: bool


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/api/status", response_model=StatusResponse)
async def get_status():
    return StatusResponse(
        ready=_state["vectorstore"] is not None,
        file_count=len(_state["uploaded_files"]),
        files=_state["uploaded_files"],
        processing=_state["processing"],
    )


@app.post("/api/upload")
async def upload_documents(files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="No files provided.")

    _state["processing"] = True
    _state["vectorstore"] = None
    _state["uploaded_files"] = []
    _reset_memory()

    combined_text = ""
    uploaded_names = []

    try:
        for f in files:
            if not f.filename.lower().endswith(".pdf"):
                raise HTTPException(status_code=400, detail=f"Only PDF files accepted. Got: {f.filename}")
            raw = await f.read()
            text = extract_text_from_bytes(raw)
            if not text.strip():
                raise HTTPException(status_code=422, detail=f"No extractable text in '{f.filename}'. It may be a scanned/image PDF.")
            combined_text += text
            uploaded_names.append(f.filename)

        _state["vectorstore"] = build_vectorstore(combined_text)
        _state["uploaded_files"] = uploaded_names
        _state["processing"] = False
        logger.info(f"Processed {len(uploaded_names)} file(s) successfully.")

        return JSONResponse(content={
            "success": True,
            "message": f"Processed {len(uploaded_names)} document(s) successfully.",
            "files": uploaded_names,
        })

    except HTTPException:
        _state["processing"] = False
        raise
    except Exception as e:
        _state["processing"] = False
        logger.error(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    if _state["vectorstore"] is None:
        raise HTTPException(status_code=400, detail="No documents loaded. Please upload and process PDFs first.")
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY not found in .env file.")

    try:
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", api_key=api_key, temperature=0.7)
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=_state["vectorstore"].as_retriever(search_kwargs={"k": 4}),
            chain_type="stuff",
            return_source_documents=True,
        )
        chat_history = [(m.content, r.content) for m, r in zip(
            _state["memory"].chat_memory.messages[::2],
            _state["memory"].chat_memory.messages[1::2]
        )] if _state["memory"].chat_memory.messages else []

        response = qa_chain({"question": req.question, "chat_history": chat_history})
        answer = response["answer"]

        from langchain_core.messages import HumanMessage, AIMessage
        _state["memory"].chat_memory.add_message(HumanMessage(content=req.question))
        _state["memory"].chat_memory.add_message(AIMessage(content=answer))

        sources = []
        if "source_documents" in response:
            for doc in response["source_documents"][:3]:
                snippet = doc.page_content[:200].strip()
                if snippet:
                    sources.append(snippet)
        return ChatResponse(answer=answer, sources=sources)

    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")


@app.post("/api/reset")
async def reset_session():
    _state["vectorstore"] = None
    _state["uploaded_files"] = []
    _state["processing"] = False
    _reset_memory()
    return {"success": True, "message": "Session reset successfully."}


# ── Static files ──────────────────────────────────────────────────────────────
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def serve_ui():
    return FileResponse("static/index.html")


if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)
