"""Groq LLM client with retry logic and JSON parsing."""

import json
import time
import logging
from groq import Groq
from config import GROQ_API_KEY, GROQ_MODEL

logger = logging.getLogger(__name__)


def call_groq(
    system_prompt: str,
    user_content: str,
    temperature: float = 0.1,
    max_tokens: int = 200,
    max_retries: int = 2
) -> dict:
    """
    Send a request to Groq and parse the JSON response.
    Retries on invalid JSON or rate limits.
    
    Returns: parsed JSON dict
    Raises: ValueError if all retries exhausted
    """
    client = Groq(api_key=GROQ_API_KEY)
    
    for attempt in range(max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            raw_text = response.choices[0].message.content.strip()
            
            # Try to parse as JSON
            # Handle case where LLM wraps JSON in markdown code block
            if raw_text.startswith("```"):
                raw_text = raw_text.split("```")[1]
                if raw_text.startswith("json"):
                    raw_text = raw_text[4:]
                raw_text = raw_text.strip()
            
            return json.loads(raw_text)
            
        except json.JSONDecodeError:
            logger.warning(f"Attempt {attempt + 1}: Invalid JSON from LLM, retrying...")
            if attempt < max_retries:
                time.sleep(1)
                continue
            raise ValueError(f"LLM returned invalid JSON after {max_retries + 1} attempts")
            
        except Exception as e:
            if "429" in str(e) or "rate" in str(e).lower():
                wait = 2 ** attempt
                logger.warning(f"Rate limited. Waiting {wait}s...")
                time.sleep(wait)
                continue
            raise
    
    raise ValueError("All retries exhausted")
