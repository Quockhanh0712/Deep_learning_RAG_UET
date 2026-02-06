"""
LLM Settings API - Manage LLM providers, models, and API keys.
"""
from typing import Optional, List
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.core.llm_factory import (
    get_llm_factory, 
    LLMProvider, 
    PROVIDER_CONFIGS,
    LLMResponse
)
from backend.utils.logger import get_logger

logger = get_logger("api.llm_settings")
router = APIRouter()


class SetProviderRequest(BaseModel):
    """Request to set active LLM provider."""
    provider: str  # e.g., "fpt_cloud", "local_ollama"
    model: Optional[str] = None
    api_key: Optional[str] = None


class ProviderInfo(BaseModel):
    """Information about a provider."""
    id: str
    name: str
    models: List[str]
    default_model: str
    has_api_key: bool
    cost_per_1m_input: float
    cost_per_1m_output: float


class ActiveProviderResponse(BaseModel):
    """Response with active provider info."""
    provider: str
    provider_name: str
    model: str
    has_api_key: bool


class TestChatRequest(BaseModel):
    """Request to test chat with current provider."""
    message: str = "Xin chào, hãy giới thiệu ngắn gọn."


class TestChatResponse(BaseModel):
    """Response from test chat."""
    success: bool
    content: Optional[str] = None
    provider: Optional[str] = None
    model: Optional[str] = None
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    error: Optional[str] = None


@router.get("/llm/providers")
async def list_providers():
    """List all available LLM providers."""
    factory = get_llm_factory()
    
    providers = []
    for provider_enum, config in PROVIDER_CONFIGS.items():
        has_key = provider_enum in factory._api_keys and len(factory._api_keys[provider_enum]) > 0
        providers.append({
            "id": provider_enum.value,
            "name": config.name,
            "models": config.models,
            "default_model": config.default_model,
            "has_api_key": has_key,
            "cost_per_1m_input": config.cost_per_1m_input,
            "cost_per_1m_output": config.cost_per_1m_output,
        })
    
    return {"providers": providers}


@router.get("/llm/active")
async def get_active_provider():
    """Get currently active LLM provider."""
    factory = get_llm_factory()
    config = factory.get_active_config()
    return config


@router.post("/llm/set-provider")
async def set_provider(request: SetProviderRequest):
    """Set active LLM provider and optionally add API key."""
    factory = get_llm_factory()
    
    try:
        provider = LLMProvider(request.provider)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Unknown provider: {request.provider}")
    
    # Add API key if provided
    if request.api_key:
        factory.set_api_key(provider, request.api_key)
    
    # Set active provider
    factory.set_provider(provider, request.model)
    
    logger.info(f"LLM provider set to: {provider.value} / {request.model}")
    
    return {
        "status": "success",
        "provider": provider.value,
        "model": request.model or PROVIDER_CONFIGS[provider].default_model,
        "message": f"Switched to {PROVIDER_CONFIGS[provider].name}"
    }


@router.post("/llm/add-key")
async def add_api_key(provider: str, api_key: str):
    """Add an API key for a provider."""
    factory = get_llm_factory()
    
    try:
        provider_enum = LLMProvider(provider)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Unknown provider: {provider}")
    
    factory.set_api_key(provider_enum, api_key)
    
    return {
        "status": "success",
        "provider": provider,
        "message": f"API key added for {PROVIDER_CONFIGS[provider_enum].name}"
    }


@router.post("/llm/test")
async def test_llm(request: TestChatRequest):
    """Test current LLM provider with a simple message."""
    factory = get_llm_factory()
    
    try:
        messages = [
            {"role": "system", "content": "You are a helpful assistant. Respond briefly."},
            {"role": "user", "content": request.message}
        ]
        
        response: LLMResponse = await factory.chat(messages, max_tokens=256)
        
        return TestChatResponse(
            success=True,
            content=response.content,
            provider=response.provider,
            model=response.model,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            cost_usd=response.cost_usd,
        )
        
    except Exception as e:
        logger.error(f"LLM test failed: {e}")
        return TestChatResponse(
            success=False,
            error=str(e)
        )
