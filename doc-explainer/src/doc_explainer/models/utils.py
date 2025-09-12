from enum import Enum
from typing import Any, Optional

from PIL.Image import Image

from .claude import generate_prediction_claude
from .qwen import generate_prediction_qwen, get_model_and_processor_qwen
from .smol import generate_prediction_smol, get_model_and_processor_smol


class VLMModel(str, Enum):
    SMOLVLM = "smolvlm"
    QWEN = "qwen2.5-vl-7b"
    CLAUDE = "claude-sonnet-4"


def get_model_and_processor(model_name: str):
    """
    Get the model and processor based on the model name.

    Args:
        model_name (str): The name of the model to load.

    Returns:
        model: The loaded model instance.
        processor: The processor instance for the model.
    """
    if model_name == VLMModel.SMOLVLM:
        model, processor = get_model_and_processor_smol()
    elif model_name == VLMModel.QWEN:
        model, processor = get_model_and_processor_qwen()
    elif model_name == VLMModel.CLAUDE:
        return None, None
    else:
        raise ValueError(f"Unsupported model name: {model_name}")
    return model, processor


def generate_prediction(
    prompt: str, image: Image, model_name: str, model: Any, processor: Any
) -> Optional[dict]:
    """
    Forward the prompt and image to the model to get a prediction.

    Args:
        prompt (str): The text prompt to be processed by the model.
        image (PIL.Image): The image to be processed by the model.
        model_name (str): The name of the model being used.
        model: The model instance.
        processor: The processor instance.
    """
    if model_name == VLMModel.SMOLVLM:
        prediction = generate_prediction_smol(prompt, image, model, processor)
    elif model_name == VLMModel.QWEN:
        prediction = generate_prediction_qwen(prompt, image, model, processor)
    elif model_name == VLMModel.CLAUDE:
        prediction = generate_prediction_claude(prompt, image)
    else:
        raise ValueError(f"Unsupported model name: {model_name}")
    return prediction
