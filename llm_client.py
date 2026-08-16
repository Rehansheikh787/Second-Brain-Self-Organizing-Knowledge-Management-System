"""Groq LLM client with retry logic and JSON parsing."""

import json
import re
import time
import random
import logging
from groq import Groq
from config import GROQ_API_KEY, GROQ_MODEL

logger = logging.getLogger(__name__)

FALLBACK_MODELS = [GROQ_MODEL, "llama-3.1-8b-instant", "gemma2-9b-it"]


def call_groq(
    system_prompt: str,
    user_content: str,
    temperature: float = 0.1,
    max_tokens: int = 200,
    max_retries: int = 4
) -> dict:
    """
    Send a request to Groq and parse the JSON response.
    Features multi-model fallback (llama-3.3-70b -> llama-3.1-8b-instant)
    and jittered exponential backoff for 429 rate limit resilience.
    
    Returns: parsed JSON dict
    Raises: ValueError if all retries and fallback models fail.
    """
    import os
    import config
    api_key = config.GROQ_API_KEY or os.getenv("GROQ_API_KEY", "")
    if not api_key:
        raise ValueError("GROQ_API_KEY is not set. Please add it to your .env file or Streamlit secrets.")
    client = Groq(api_key=api_key)
    last_error = None

    for model_name in FALLBACK_MODELS:
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content}
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    response_format={"type": "json_object"}
                )
                
                raw_text = response.choices[0].message.content.strip()
                
                # Fallback 1: Markdown code block stripping
                if raw_text.startswith("```"):
                    raw_text = raw_text.split("```")[1]
                    if raw_text.startswith("json"):
                        raw_text = raw_text[4:]
                    raw_text = raw_text.strip()
                
                # Try primary parse
                try:
                    return json.loads(raw_text)
                except json.JSONDecodeError:
                    # Fallback 2: Extract first JSON object via regex
                    match = re.search(r"\{.*\}", raw_text, re.DOTALL)
                    if match:
                        return json.loads(match.group(0))
                    raise
                
            except json.JSONDecodeError as err:
                last_error = err
                logger.warning(f"Attempt {attempt + 1} ({model_name}): Invalid JSON from LLM, retrying...")
                if attempt < max_retries - 1:
                    time.sleep(1)
                    continue
                break  # Try next model
                
            except Exception as e:
                last_error = e
                err_msg = str(e).lower()
                if "429" in err_msg or "rate" in err_msg or "quota" in err_msg:
                    wait = min(12.0, (1.5 ** attempt) + random.uniform(0.5, 1.2))
                    logger.warning(f"Rate limited on {model_name} (attempt {attempt + 1}). Waiting {wait:.1f}s...")
                    time.sleep(wait)
                    continue
                else:
                    logger.warning(f"Groq API call error on {model_name}: {e}")
                    break  # Try next model fallback
        
    raise ValueError(f"All LLM retries and fallback models failed: {last_error}")
