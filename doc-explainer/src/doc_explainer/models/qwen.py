import json
import re

from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from .constants import SYSTEM_MESSAGE


def get_model_and_processor_qwen():
    """
    Get the Qwen2.5-VL model and processor.

    Returns:
        model: The loaded model instance.
        processor: The processor instance for the model.
    """
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct",
        torch_dtype="auto",
        device_map="auto",
        attn_implementation="flash_attention_2",
    )

    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

    return model, processor


def generate_prediction_qwen(
    prompt: str,
    image: Image,
    model: Qwen2_5_VLForConditionalGeneration,
    processor: AutoProcessor,
):
    """
    Generate prediction using Qwen2.5-VL model.

    Args:
        prompt (str): The text prompt for the model
        image (Image): PIL Image object
        model: The Qwen2.5-VL model instance
        processor: The processor instance

    Returns:
        dict or None: JSON response if valid, None if parsing fails
    """
    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_MESSAGE}]},
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,
                },
                {"type": "text", "text": prompt},
            ],
        },
    ]

    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=256)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids, strict=False)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    decoded_output = output_text[0].strip()

    # Extract JSON using regex to handle various markdown formats
    json_pattern = r"```(?:json)?\s*\n?(.*?)\n?```"
    match = re.search(json_pattern, decoded_output, re.DOTALL)

    if match:
        json_str = match.group(1).strip()
    else:
        # No markdown formatting found, use the full output
        json_str = decoded_output

    try:
        json_response = json.loads(json_str)
        return json_response
    except json.JSONDecodeError:
        return None
