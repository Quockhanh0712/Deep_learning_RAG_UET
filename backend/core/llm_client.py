"""
Ollama LLM Client for Legal RAG.
Wraps Ollama API for text generation.
"""
import os
from typing import Optional, Dict, Any, AsyncGenerator
import asyncio

from backend.config import settings
from backend.utils.logger import get_logger
from backend.utils.exceptions import OllamaConnectionError, LLMError

logger = get_logger("llm_client")


class OllamaClient:
    """
    Ollama LLM Client for text generation.
    Uses qwen2.5:3b by default with GPU acceleration.
    """
    
    _instance: Optional["OllamaClient"] = None
    
    def __new__(cls) -> "OllamaClient":
        """Singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self.model = settings.OLLAMA_MODEL
        self.host = settings.OLLAMA_HOST
        self.num_gpu = settings.OLLAMA_NUM_GPU
        self._client = None
        self._initialized = True
        
        logger.info(f"OllamaClient initialized (model={self.model})")
    
    def _get_client(self):
        """Get Ollama client (lazy initialization)."""
        if self._client is None:
            try:
                import ollama
                # Set host if different from default
                if self.host != "http://localhost:11434":
                    ollama.Client(host=self.host)
                self._client = ollama
                logger.info("✓ Ollama client initialized")
            except Exception as e:
                raise OllamaConnectionError(self.host, str(e))
        return self._client
    
    def check_available(self) -> bool:
        """Check if Ollama is available and model is loaded."""
        try:
            client = self._get_client()
            models = client.list()
            model_names = [m.get("name", "") for m in models.get("models", [])]
            
            # Check if our model is available
            for name in model_names:
                if self.model in name:
                    return True
            
            logger.warning(f"Model {self.model} not found in Ollama")
            return False
            
        except Exception as e:
            logger.error(f"Ollama not available: {e}")
            return False
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 2048,
        stop: Optional[list] = None
    ) -> str:
        """
        Generate text synchronously.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system message
            temperature: Sampling temperature (lower = more focused)
            max_tokens: Maximum tokens to generate
            stop: Stop sequences
            
        Returns:
            Generated text
        """
        try:
            client = self._get_client()
            
            options = {
                "num_gpu": self.num_gpu,
                "temperature": temperature,
                "num_predict": max_tokens,
            }
            
            if stop:
                options["stop"] = stop
            
            response = client.generate(
                model=self.model,
                prompt=prompt,
                system=system_prompt or "",
                options=options
            )
            
            return response.get("response", "")
            
        except Exception as e:
            raise LLMError(self.model, str(e))
    
    async def generate_async(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 2048
    ) -> str:
        """
        Generate text asynchronously.
        Runs sync generation in thread pool.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.generate(prompt, system_prompt, temperature, max_tokens)
        )
    
    def chat(
        self,
        messages: list,
        temperature: float = 0.1,
        max_tokens: int = 2048
    ) -> str:
        """
        Chat completion with conversation history.
        
        Args:
            messages: List of {"role": "user/assistant/system", "content": "..."}
            temperature: Sampling temperature
            max_tokens: Maximum tokens
            
        Returns:
            Assistant response
        """
        try:
            client = self._get_client()
            
            response = client.chat(
                model=self.model,
                messages=messages,
                options={
                    "num_gpu": self.num_gpu,
                    "temperature": temperature,
                    "num_predict": max_tokens
                }
            )
            
            return response.get("message", {}).get("content", "")
            
        except Exception as e:
            raise LLMError(self.model, str(e))
    
    def get_info(self) -> dict:
        """Get client information."""
        return {
            "model": self.model,
            "host": self.host,
            "num_gpu": self.num_gpu,
            "available": self.check_available()
        }


def get_llm_client() -> OllamaClient:
    """Get singleton LLM client instance."""
    return OllamaClient()


# ==================== Prompt Templates ====================

LEGAL_RAG_SYSTEM_PROMPT = """
Bạn là Trợ lý Luật sư AI chuyên nghiệp, trung thực và chính xác.
Nhiệm vụ của bạn là trả lời câu hỏi dựa trên các đoạn văn bản pháp luật được cung cấp dưới đây.

YÊU CẦU BẮT BUỘC:
1. CHỈ sử dụng thông tin trong phần CONTEXT. KHÔNG tự bịa ra thông tin.
2. Nếu context không có thông tin để trả lời, HÃY NÓI THẲNG: "Xin lỗi, tôi không tìm thấy thông tin trong tài liệu pháp luật được cung cấp."
3. Mọi câu trả lời PHẢI có trích dẫn cụ thể (Ví dụ: [Điều 123, Bộ luật Hình sự]).
4. Trình bày ngắn gọn, súc tích, đi thẳng vào vấn đề.
5. Giữ giọng văn khách quan, chuyên nghiệp.
"""


def build_rag_prompt(question: str, context: str) -> str:
    """
    Build RAG prompt with question and context.
    
    Args:
        question: User's question
        context: Retrieved legal context
        
    Returns:
        Formatted prompt
    """
    return f"""Ngữ cảnh pháp lý:
{context}

---

Câu hỏi: {question}

Dựa trên ngữ cảnh pháp lý được cung cấp ở trên, hãy trả lời câu hỏi một cách chính xác và đầy đủ. Trích dẫn số Điều, Khoản cụ thể khi cần thiết."""
