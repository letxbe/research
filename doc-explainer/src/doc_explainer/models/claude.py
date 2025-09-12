import base64
import json
import os
import random
import re
import time
from io import BytesIO
from typing import Optional

import anthropic
from PIL import Image

from .constants import SYSTEM_MESSAGE


def pil_to_base64(image: Image.Image) -> str:
    """Convert a PIL Image to base64 string."""
    buffered = BytesIO()
    image.save(buffered, format="JPEG")  # or PNG depending on input
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def safe_json_parse(text: str) -> Optional[dict]:
    # Remove leading/trailing spaces
    text = text.strip()

    # Extract JSON-like part
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        candidate = match.group(0)
    else:
        return None

    # Fix common trailing comma issues
    candidate = re.sub(r",\s*}", "}", candidate)
    candidate = re.sub(r",\s*\]", "]", candidate)

    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return None


def call_with_retries(client, **kwargs):
    max_retries = 5
    backoff = 3
    for attempt in range(max_retries):
        try:
            return client.beta.messages.create(**kwargs)
        except anthropic.InternalServerError as e:
            print(
                f"[Attempt {attempt + 1}] Server error (500): {e}. Retrying in {backoff} seconds..."
            )
            time.sleep(backoff + random.random())
            backoff *= 2
        except anthropic.RateLimitError as e:
            print(
                f"[Attempt {attempt + 1}] Rate limited: {e}. Retrying in {backoff} seconds..."
            )
            time.sleep(backoff + random.random())
            backoff *= 2
        except Exception as e:
            print(f"Unexpected error: {e}")
            raise
    raise RuntimeError("Failed after retries due to repeated errors")


def generate_prediction_claude(prompt: str, image: Image.Image) -> Optional[dict]:
    width, height = image.size

    if width > 8000 or height > 8000:
        print(
            f"Image dimensions are too large for processing. Width: {width}, Height: {height}"
        )
        return None

    client = anthropic.Anthropic()

    try:
        time.sleep(1)  # small delay between calls

        message = call_with_retries(
            client,
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            system=SYSTEM_MESSAGE,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": pil_to_base64(image),
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        )
    except Exception as e:
        print(f"Error generating prediction: {e}")
        return None

    text_content = "".join(block.text for block in message.content)
    return safe_json_parse(text_content)
