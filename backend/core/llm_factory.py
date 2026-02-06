"""
Multi-LLM Provider Factory - Enterprise Grade LLM Management.

Supports:
- Local Ollama (qwen2.5:3b)
- FPT Cloud (Qwen3-32B, Qwen3-14B)
- OpenAI (gpt-4o-mini, gpt-4o)
- Anthropic (claude-3.5-sonnet)
- Groq (llama-3.1-70b)
"""
import os
import json
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum

from backend.utils.logger import get_logger

logger = get_logger("core.llm_factory")


class LLMProvider(str, Enum):
    """Supported LLM Providers."""
    LOCAL_OLLAMA = "local_ollama"
    FPT_CLOUD = "fpt_cloud"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GROQ = "groq"


@dataclass
class ProviderConfig:
    """Configuration for an LLM provider."""
    name: str
    base_url: str
    default_model: str
    models: List[str]
    cost_per_1m_input: float  # USD per 1M input tokens
    cost_per_1m_output: float  # USD per 1M output tokens


# Provider configurations
PROVIDER_CONFIGS: Dict[LLMProvider, ProviderConfig] = {
    LLMProvider.LOCAL_OLLAMA: ProviderConfig(
        name="Local Ollama",
        base_url="http://localhost:11434/v1",
        default_model="qwen2.5:3b",
        models=["qwen2.5:3b", "qwen2.5:7b", "llama3.1:8b"],
        cost_per_1m_input=0.0,
        cost_per_1m_output=0.0,
    ),
    LLMProvider.FPT_CLOUD: ProviderConfig(
        name="FPT Cloud",
        base_url="https://mkp-api.fptcloud.com/v1",
        default_model="Qwen3-32B",
        models=["Qwen3-32B", "Qwen3-14B", "Qwen3-8B"],
        cost_per_1m_input=0.06,
        cost_per_1m_output=0.08,
    ),
    LLMProvider.OPENAI: ProviderConfig(
        name="OpenAI",
        base_url="https://api.openai.com/v1",
        default_model="gpt-4o-mini",
        models=["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
        cost_per_1m_input=0.15,
        cost_per_1m_output=0.60,
    ),
    LLMProvider.ANTHROPIC: ProviderConfig(
        name="Anthropic",
        base_url="https://api.anthropic.com",
        default_model="claude-3.5-sonnet",
        models=["claude-3.5-sonnet", "claude-3-haiku"],
        cost_per_1m_input=3.0,
        cost_per_1m_output=15.0,
    ),
    LLMProvider.GROQ: ProviderConfig(
        name="Groq",
        base_url="https://api.groq.com/openai/v1",
        default_model="llama-3.1-70b-versatile",
        models=["llama-3.1-70b-versatile", "llama-3.1-8b-instant"],
        cost_per_1m_input=0.59,
        cost_per_1m_output=0.79,
    ),
}


@dataclass
class LLMResponse:
    """Response from LLM with cost tracking."""
    content: str
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    raw_response: Optional[Any] = None


class LLMFactory:
    """Multi-provider LLM factory with failover support."""
    
    def __init__(self):
        self._api_keys: Dict[str, List[str]] = {}
        self._active_provider: LLMProvider = LLMProvider.LOCAL_OLLAMA
        self._active_model: Optional[str] = None
        self._load_api_keys()
    
    def _load_api_keys(self):
        """Load API keys from environment or config file."""
        # Try environment variables first
        fpt_key = os.getenv("FPT_CLOUD_API_KEY")
        if fpt_key:
            self._api_keys[LLMProvider.FPT_CLOUD] = [fpt_key]
        
        openai_key = os.getenv("OPENAI_API_KEY")
        if openai_key:
            self._api_keys[LLMProvider.OPENAI] = [openai_key]
        
        groq_key = os.getenv("GROQ_API_KEY")
        if groq_key:
            self._api_keys[LLMProvider.GROQ] = [groq_key]
        
        # Local Ollama doesn't need API key
        self._api_keys[LLMProvider.LOCAL_OLLAMA] = ["local"]
        
        logger.info(f"Loaded API keys for providers: {list(self._api_keys.keys())}")
    
    def set_api_key(self, provider: LLMProvider, api_key: str):
        """Set or add an API key for a provider."""
        if provider not in self._api_keys:
            self._api_keys[provider] = []
        if api_key not in self._api_keys[provider]:
            self._api_keys[provider].append(api_key)
        logger.info(f"Added API key for {provider.value}")
    
    def set_provider(self, provider: LLMProvider, model: Optional[str] = None):
        """Switch active provider and model."""
        self._active_provider = provider
        if model:
            self._active_model = model
        else:
            self._active_model = PROVIDER_CONFIGS[provider].default_model
        logger.info(f"Switched to {provider.value} / {self._active_model}")
    
    def get_active_config(self) -> Dict[str, Any]:
        """Get current active provider configuration."""
        config = PROVIDER_CONFIGS[self._active_provider]
        return {
            "provider": self._active_provider.value,
            "provider_name": config.name,
            "model": self._active_model or config.default_model,
            "base_url": config.base_url,
            "has_api_key": self._active_provider in self._api_keys,
        }
    
    async def chat(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> LLMResponse:
        """Send chat completion request with automatic failover."""
        provider = self._active_provider
        model = model or self._active_model or PROVIDER_CONFIGS[provider].default_model
        config = PROVIDER_CONFIGS[provider]
        
        # Get API keys for this provider
        api_keys = self._api_keys.get(provider, [])
        if not api_keys and provider != LLMProvider.LOCAL_OLLAMA:
            raise ValueError(f"No API key configured for {provider.value}")
        
        # Try each API key with failover
        last_error = None
        for api_key in api_keys:
            try:
                return await self._call_provider(
                    provider=provider,
                    config=config,
                    api_key=api_key,
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            except Exception as e:
                logger.warning(f"API call failed with key ending ...{api_key[-4:]}: {e}")
                last_error = e
                continue
        
        raise Exception(f"All API keys failed for {provider.value}: {last_error}")
    
    async def _call_provider(
        self,
        provider: LLMProvider,
        config: ProviderConfig,
        api_key: str,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> LLMResponse:
        """Make actual API call to provider."""
        
        if provider == LLMProvider.LOCAL_OLLAMA:
            return await self._call_ollama(model, messages, temperature, max_tokens)
        else:
            return await self._call_openai_compatible(
                config, api_key, model, messages, temperature, max_tokens
            )
    
    async def _call_ollama(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> LLMResponse:
        """Call local Ollama instance."""
        import ollama
        
        response = ollama.chat(
            model=model,
            messages=messages,
            options={"temperature": temperature, "num_predict": max_tokens}
        )
        
        content = response.get("message", {}).get("content", "")
        
        return LLMResponse(
            content=content,
            provider="local_ollama",
            model=model,
            input_tokens=response.get("prompt_eval_count", 0),
            output_tokens=response.get("eval_count", 0),
            cost_usd=0.0,
            raw_response=response,
        )
    
    async def _call_openai_compatible(
        self,
        config: ProviderConfig,
        api_key: str,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> LLMResponse:
        """Call OpenAI-compatible API (FPT Cloud, OpenAI, Groq)."""
        from openai import OpenAI
        
        client = OpenAI(api_key=api_key, base_url=config.base_url)
        
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        
        content = response.choices[0].message.content or ""
        input_tokens = response.usage.prompt_tokens if response.usage else 0
        output_tokens = response.usage.completion_tokens if response.usage else 0
        
        # Calculate cost
        cost = (
            (input_tokens / 1_000_000) * config.cost_per_1m_input +
            (output_tokens / 1_000_000) * config.cost_per_1m_output
        )
        
        return LLMResponse(
            content=content,
            provider=config.name,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
            raw_response=response,
        )


# Global factory instance
_llm_factory: Optional[LLMFactory] = None


def get_llm_factory() -> LLMFactory:
    """Get or create global LLM factory instance."""
    global _llm_factory
    if _llm_factory is None:
        _llm_factory = LLMFactory()
    return _llm_factory
