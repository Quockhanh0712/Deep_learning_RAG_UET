# src/llm_client.py
import os
from google import genai
import types
import logging

logger = logging.getLogger(__name__)

# Configuration
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")

# Initialize Gemini client
if GEMINI_API_KEY:
    _gemini_client = genai.Client(api_key=GEMINI_API_KEY)
else:
    # Dummy client for testing without API key
    class DummyClient:
        class DummyModels:
            def generate_content(self, model, contents):
                return types.SimpleNamespace(text="DUMMY_ANSWER_NO_KEY")
        @property
        def models(self):
            return self.DummyModels()
    _gemini_client = DummyClient()

def _generate_with_ollama(prompt: str) -> str:
    """Generate answer using Ollama local LLM"""
    try:
        import ollama
        logger.info(f"[OLLAMA] Generating with model: {OLLAMA_MODEL}")
        
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[{
                'role': 'user',
                'content': prompt
            }]
        )
        
        answer = response['message']['content']
        logger.info(f"[OLLAMA] Generated {len(answer)} characters")
        return answer
        
    except ImportError:
        logger.error("[OLLAMA] ollama package not installed. Run: pip install ollama")
        return "Error: Ollama Python package not installed. Please run: pip install ollama"
    except Exception as e:
        logger.error(f"[OLLAMA] Error: {e}")
        return f"Error calling Ollama: {str(e)}"

def _generate_with_gemini(prompt: str) -> str:
    """Generate answer using Google Gemini API"""
    logger.info(f"[GEMINI] Generating with model: {GEMINI_MODEL}")
    resp = _gemini_client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
    )
    logger.info(f"[GEMINI] Generated {len(resp.text)} characters")
    return resp.text

def generate_answer(prompt: str) -> str:
    """
    Generate answer using configured LLM provider.
    
    Provider is selected based on LLM_PROVIDER environment variable:
    - "ollama": Uses local Ollama (GPU, free, offline)
    - "gemini": Uses Google Gemini API (cloud, requires API key)
    """
    logger.info(f"[LLM] Using provider: {LLM_PROVIDER}")
    
    if LLM_PROVIDER == "ollama":
        return _generate_with_ollama(prompt)
    else:  # default to gemini
        return _generate_with_gemini(prompt)

