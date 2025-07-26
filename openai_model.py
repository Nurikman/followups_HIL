"""
OpenAI model configuration for agent systems.
"""
from openai import OpenAI
import os
from typing import Optional

def get_openai_model(model_name: str = "gpt-4o") -> object:
    """
    Get an OpenAI model instance for use with pydantic_ai.
    
    Args:
        model_name: The name of the OpenAI model to use (e.g., "gpt-4o", "gpt-4", "gpt-3.5-turbo")
    
    Returns:
        OpenAI model instance configured for pydantic_ai
    """
    # Get API key from environment or use a default
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        # Try to get from common environment variables
        api_key = os.getenv("OPENAI_KEY") or os.getenv("OPENAI_TOKEN")
    
    if not api_key:
        raise ValueError(
            "OpenAI API key not found. Please set OPENAI_API_KEY environment variable "
            "or configure it in your Streamlit app."
        )
    
    # Return the model name for pydantic_ai
    # pydantic_ai handles the actual OpenAI client creation internally
    return model_name

def get_openai_client(api_key: Optional[str] = None) -> OpenAI:
    """
    Get a direct OpenAI client instance.
    
    Args:
        api_key: Optional API key. If not provided, will try to get from environment.
    
    Returns:
        OpenAI client instance
    """
    if not api_key:
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_KEY") or os.getenv("OPENAI_TOKEN")
    
    if not api_key:
        raise ValueError(
            "OpenAI API key not found. Please set OPENAI_API_KEY environment variable."
        )
    
    return OpenAI(api_key=api_key) 