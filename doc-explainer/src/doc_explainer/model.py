from typing import List, Optional

import torch.nn as nn
from PIL.Image import Image
from transformers import AutoModel

from .models.utils import generate_prediction, get_model_and_processor
from .type import ExplainableAnswer

VLM_PROMPT = """Based only on the document image, answer the following question:

Question: {QUESTION}

Provide ONLY a JSON response in the following format:
{{
  "content": "answer",
}}
"""


class DocExplainer(nn.Module):
    """
    DocExplainer is a Vision Language Model that can answer questions about document images
    and provide explanations by highlighting relevant regions in the document.

    This model combines a Vision Language Model (VLM) for question answering with an
    explainer model that identifies the relevant bounding box regions in the document
    that support the answer.

    Attributes:
        vlm_model_name (str): Name of the VLM model to use for question answering
        device (str): Device to run the models on (e.g., 'cuda', 'cpu')
        explainer: The explainer model for identifying relevant regions
        vlm: The Vision Language Model instance
        processor: The processor for the VLM model
    """

    def __init__(self, vlm_model_name: str = "smolvlm", device: str = "cuda"):
        super().__init__()

        self.vlm_model_name = vlm_model_name
        self.device = device
        self.explainer = AutoModel.from_pretrained(
            "letxbe/DocExplainer", trust_remote_code=True
        )

        self.vlm, self.processor = get_model_and_processor(vlm_model_name)

    def forward(
        self, document: List[Image], question: str
    ) -> Optional[ExplainableAnswer]:
        for page_idx, page in enumerate(document):
            prompt = VLM_PROMPT.format(QUESTION=question)

            prediction = generate_prediction(
                prompt=prompt,
                model_name=self.vlm_model_name,
                image=page,
                model=self.vlm,
                processor=self.processor,
            )

            if prediction and isinstance(prediction, dict):
                answer = prediction.get("content", None)

                if answer:
                    bbox = self.explainer.predict(
                        page, f"Question: {question} Answer: {answer}"
                    )
                    return ExplainableAnswer(answer=answer, page=page_idx, bbox=bbox)

        return None
