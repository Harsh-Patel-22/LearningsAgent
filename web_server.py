import sys
import os
import shutil
import asyncio
import logging
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
import uvicorn
from dotenv import load_dotenv

# Apply Windows event loop fixes
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    os.environ["PYTHONIOENCODING"] = "utf-8"

load_dotenv()

from orchestrator import CMOSApplication, CMOSConfig
from memory_repo import (
    SimpleMemoryRepo, SimplePatternRepo,
    SimpleContradictionRepo, SimplePsychModelStore
)
from pdf_parser import parse_pdf

# Initialize Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("cmos_web")

app = FastAPI(title="Cognitive Memory OS Web Portal")

# Global Application Reference
cmos_app: Optional[CMOSApplication] = None
config = CMOSConfig(
    gemini_api_key=os.getenv("GEMINI_API_KEY"),
    qdrant_host=os.getenv("QDRANT_HOST", "localhost"),
    qdrant_port=int(os.getenv("QDRANT_PORT", "6333")),
)

@app.on_event("startup")
async def startup_event():
    global cmos_app
    cmos_app = CMOSApplication(config)
    await cmos_app.initialize(
        memory_repo=SimpleMemoryRepo(),
        pattern_repo=SimplePatternRepo(),
        contradiction_repo=SimpleContradictionRepo(),
        psych_model_store=SimplePsychModelStore(),
    )
    logger.info("CMOS Application initialized.")

# Request models
class ChatRequest(BaseModel):
    message: str

class TextIngestRequest(BaseModel):
    text: str
    date: Optional[str] = None

@app.get("/", response_class=HTMLResponse)
async def serve_home():
    template_path = Path("templates/index.html")
    if not template_path.exists():
        raise HTTPException(status_code=404, detail="Frontend templates/index.html not found.")
    return template_path.read_text(encoding="utf-8")

@app.post("/api/chat")
async def chat_endpoint(req: ChatRequest):
    if not cmos_app:
        return JSONResponse(status_code=500, content={"error": "CMOS not initialized"})
    try:
        response_text = await cmos_app.ask(req.message)
        return {"response": response_text}
    except Exception as e:
        logger.error(f"Chat error: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/api/ingest/text")
async def ingest_text_endpoint(req: TextIngestRequest):
    if not cmos_app:
        return JSONResponse(status_code=500, content={"error": "CMOS not initialized"})
    try:
        date_val = req.date if req.date and req.date.strip() else None
        result = await cmos_app.ingest_note(
            text=req.text,
            date_str=date_val,
            hint_life_phase="current"
        )
        if result.success:
            # Trigger background analysis in parallel to update patterns
            asyncio.create_task(cmos_app.run_background_analysis())
            return {"success": True, "memories_created": result.memories_created}
        else:
            return JSONResponse(status_code=400, content={"error": result.error})
    except Exception as e:
        logger.error(f"Ingestion error: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/api/ingest/file")
async def ingest_file_endpoint(
    file: UploadFile = File(...),
    date: Optional[str] = Form(None)
):
    if not cmos_app:
        return JSONResponse(status_code=500, content={"error": "CMOS not initialized"})
    
    file_bytes = await file.read()
    filename = file.filename or "uploaded_file"
    date_val = date if date and date.strip() else None
    
    try:
        if filename.lower().endswith(".pdf"):
            logger.info(f"Processing PDF file: {filename}")
            pages = parse_pdf(file_bytes)
            pages_processed = 0
            memories_created = 0
            
            for page in pages:
                logger.info(f"Processing PDF page {page.number}/{len(pages)}")
                # If page contains selectable text, ingest directly as text
                if len(page.text.split()) > 20:
                    result = await cmos_app.ingest_note(
                        text=page.text,
                        date_str=date_val,
                        hint_life_phase="current"
                    )
                else:
                    # Otherwise treat it as a scanned image and run OCR pipeline
                    result = await cmos_app.ingest_diary_image(
                        image_bytes=page.image_bytes,
                        filename=f"{filename}_page_{page.number}.png",
                        hint_date=date_val,
                        hint_life_phase="current"
                    )
                
                if result.success:
                    pages_processed += 1
                    memories_created += result.memories_created
                    
            # Trigger background analysis
            asyncio.create_task(cmos_app.run_background_analysis())
            return {
                "success": True,
                "type": "pdf",
                "pages_processed": pages_processed,
                "memories_created": memories_created
            }
            
        elif filename.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
            logger.info(f"Processing image file: {filename}")
            result = await cmos_app.ingest_diary_image(
                image_bytes=file_bytes,
                filename=filename,
                hint_date=date_val,
                hint_life_phase="current"
            )
            if result.success:
                asyncio.create_task(cmos_app.run_background_analysis())
                return {
                    "success": True,
                    "type": "image",
                    "memories_created": result.memories_created
                }
            else:
                return JSONResponse(status_code=400, content={"error": result.error})
                
        else:
            return JSONResponse(status_code=400, content={"error": "Unsupported file format. Please upload PDF or image files."})
            
    except Exception as e:
        logger.error(f"File ingestion error: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/api/memories")
async def get_memories_endpoint():
    repo = SimpleMemoryRepo()
    try:
        memories = await repo.get_all(limit=50)
        # Convert to JSON serializable structures
        serialized = []
        for m in memories:
            serialized.append({
                "id": m.id,
                "memory_type": m.memory_type,
                "title": m.title,
                "content": m.content,
                "date": str(m.date_info.occurred or "unknown"),
                "emotion": str(m.emotion_primary or "unknown")
            })
        return {"memories": serialized}
    except Exception as e:
        logger.error(f"Error loading memories: {e}")
        return {"memories": []}

@app.post("/api/reset")
async def reset_endpoint():
    if not cmos_app:
        return JSONResponse(status_code=500, content={"error": "CMOS not initialized"})
    try:
        logger.info("Wiping local databases & Qdrant collections...")
        # 1. Clear JSON files
        data_path = Path("./data")
        for sub in ["source_docs", "memories", "patterns", "contradictions"]:
            p = data_path / sub
            if p.exists():
                for f in p.glob("*.json"):
                    f.unlink()
        
        psych_model_file = data_path / "psych_model.json"
        if psych_model_file.exists():
            psych_model_file.unlink()

        # 2. Reset Qdrant
        from qdrant_client import QdrantClient
        qclient = QdrantClient(host=config.qdrant_host, port=config.qdrant_port)
        for col in ["semantic", "emotional"]:
            try:
                qclient.delete_collection(col)
            except Exception:
                pass
        
        # Reinitialize Qdrant
        from memory_repo import QdrantMemoryStore
        qstore = QdrantMemoryStore(host=config.qdrant_host, port=config.qdrant_port)
        qstore.initialize_collections()

        # 3. Reinitialize Application
        await cmos_app.initialize(
            memory_repo=SimpleMemoryRepo(),
            pattern_repo=SimplePatternRepo(),
            contradiction_repo=SimpleContradictionRepo(),
            psych_model_store=SimplePsychModelStore(),
        )
        return {"success": True}
    except Exception as e:
        logger.error(f"Reset error: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    uvicorn.run("web_server:app", host="0.0.0.0", port=8000, reload=True)
