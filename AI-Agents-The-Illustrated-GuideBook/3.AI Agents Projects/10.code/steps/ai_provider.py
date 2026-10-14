import os
from typing import Literal
from urllib.parse import urlparse

from openai import AsyncOpenAI

AIProvider = Literal['openai', 'azure', 'ollama']

def get_ai_provider() -> AIProvider:
    provider = os.getenv('AI_PROVIDER', 'openai').strip().lower()
    if provider not in ('openai', 'azure', 'ollama'):
        raise ValueError('AI_PROVIDER must be openai, azure or ollama')
    return provider


def create_ai_client() -> tuple[AsyncOpenAI, str, AIProvider]:
    provider = get_ai_provider()
    if provider == 'azure':
        client, model = create_azure_openai_client()
    elif provider == 'ollama':
        client, model = create_ollama_client()
    else:
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError('OPENAI_API_KEY is required for OpenAI content generation')
        client = AsyncOpenAI(api_key=api_key)
        model = 'gpt-4o'
    
    print(f"Using {provider} for LinkedIn content generation")

    return client, model

def create_ollama_client() -> tuple[AsyncOpenAI, str]:
    base_url = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434').strip()
    model = os.getenv('OLLAMA_MODEL', 'ollama/llama3.2:3b').strip()
    endpoint = urlparse(base_url)
    if endpoint.scheme not in ('http', 'https') or not endpoint.hostname:
        raise ValueError('OLLAMA_BASE_URL must be a valid HTTP or HTTPS OpenAI-compatible endpoint')
    if not model:
        raise ValueError('OLLAMA_MODEL must not be empty')
    client = AsyncOpenAI(base_url=base_url, api_key='ollama')
    return client, model


def create_azure_openai_client() -> tuple[AsyncOpenAI, str]:
    required = (
        'AZURE_OPENAI_API_KEY',
        'AZURE_OPENAI_ENDPOINT_LLM',
        'AZURE_OPENAI_API_VERSION',
        'AZURE_OPENAI_DEPLOYMENT_NAME_LLM',
    )
    settings = {key: os.getenv(key, '').strip() for key in required}
    missing = [key for key, value in settings.items() if not value]
    if missing:
        raise ValueError(f"Azure OpenAI requires: {', '.join(missing)}")

    endpoint = urlparse(settings['AZURE_OPENAI_ENDPOINT_LLM'])
    if endpoint.scheme != 'https' or not endpoint.hostname:
        raise ValueError('AZURE_OPENAI_ENDPOINT_LLM must be a valid HTTPS resource endpoint')

    client = AsyncOpenAI(
        api_key=settings['AZURE_OPENAI_API_KEY'],
        base_url=settings['AZURE_OPENAI_ENDPOINT_LLM']
    )
    return client, settings['AZURE_OPENAI_DEPLOYMENT_NAME_LLM']