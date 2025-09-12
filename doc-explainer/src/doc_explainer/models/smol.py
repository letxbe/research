import json
import re

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from .constants import SYSTEM_MESSAGE


def safe_json_parse(text):
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


def get_model_and_processor_smol():
    """
    Get the SMOL model and processor.:
    Returns:
        model: The loaded model instance.
        processor: The processor instance for the model.
    """
    model = AutoModelForImageTextToText.from_pretrained(
        "HuggingFaceTB/SmolVLM2-2.2B-Instruct",
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2",
    )
    processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM2-2.2B-Instruct")
    return model, processor


def generate_prediction_smol(
    prompt: str,
    image: Image,
    model: AutoModelForImageTextToText,
    processor: AutoProcessor,
):
    messages = [
        {"role": "system", "content": [{"type": "system", "text": SYSTEM_MESSAGE}]},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        },
    ]

    # Output generation for SmolVLM2
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to("cuda", dtype=torch.bfloat16)

    input_length = inputs["input_ids"].shape[1]
    generated_ids = model.generate(**inputs, do_sample=False, max_new_tokens=2056)

    output_ids = generated_ids[:, input_length:]
    generated_texts = processor.batch_decode(
        output_ids,
        skip_special_tokens=True,
    )
    decoded_output = generated_texts[0].replace("Assistant:", "", 1).strip()

    try:
        json_response = safe_json_parse(decoded_output)
        return json_response
    except json.JSONDecodeError:
        return None
