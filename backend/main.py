"""
Legal RAG Backend - FastAPI Application Entry Point.
"""
import sys
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.config import settings
from backend.utils.logger import logger
from backend.utils.exceptions import LegalRAGException
from backend.db.database import init_db, close_db

# Import routers
from backend.api.status import router as status_router
from backend.api.search import router as search_router
from backend.api.chat import router as chat_router
from backend.api.upload import router as upload_router
from backend.api.llm_settings import router as llm_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: startup and shutdown events."""
    # Startup
    logger.info("🚀 Starting Legal RAG Backend...")
    await init_db()
    logger.info("✓ Database initialized")
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down Legal RAG Backend...")
    await close_db()
    logger.info("✓ Cleanup complete")


# Create FastAPI app
app = FastAPI(
    title="Legal RAG API",
    description="Vietnamese Legal Document RAG System with Hybrid Search",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================== Exception Handlers ====================

@app.exception_handler(LegalRAGException)
async def legal_rag_exception_handler(request: Request, exc: LegalRAGException):
    """Handle custom Legal RAG exceptions."""
    logger.error(f"LegalRAGException: {exc.message}")
    return JSONResponse(
        status_code=500,
        content=exc.to_dict()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions."""
    logger.exception(f"Unexpected error: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "InternalServerError",
            "message": "An unexpected error occurred",
            "details": {"type": type(exc).__name__}
        }
    )


# ==================== Include Routers ====================

app.include_router(status_router, prefix="/api", tags=["Status"])
app.include_router(search_router, prefix="/api", tags=["Search"])
app.include_router(chat_router, prefix="/api", tags=["Chat"])
app.include_router(upload_router, prefix="/api", tags=["Documents"])
app.include_router(llm_router, prefix="/api", tags=["LLM Settings"])


# ==================== Root Endpoints ====================

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint."""
    return {
        "name": "Legal RAG API",
        "version": "1.0.0",
        "docs": "/docs",
        "status": "/api/status"
    }


@app.get("/health", tags=["Root"])
async def health():
    """Simple health check."""
    return {"status": "healthy"}


# ==================== Run with Uvicorn ====================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG
    )
