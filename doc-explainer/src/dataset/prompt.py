from typing import Tuple

ZERO_SHOT_PROMPT = """Based only on the document image, answer the following question:

Question: {QUESTION}

Provide ONLY a JSON response in the following format:
{{
  "content": "answer",
  "position": [x, y, w, h]
}}

Each position value MUST be in the range [0, 1000]."""

COT_ONE_SHOT_PROMPT = """
Based only on this document image, answer the question step-by-step.

Question: {QUESTION}

Think through the steps needed to find the answer from the document.

Then provide ONLY a JSON response with format:
{{"content": "answer", "position": [x, y, w, h]}}.

Position MUST be in the range [0, 1000].
"""

CLAUDE_PROMPT = """
Examine the document image provided and answer the following question:

Question: {QUESTION}

Analyze the image carefully to find the answer to the question. Once you have determined the answer, locate its position within the image.

To specify the position, provide a bounding box (bbox) in the format [x, y, w, h], where:
- x: the x-coordinate of the top-left corner of the bounding box
- y: the y-coordinate of the top-left corner of the bounding box
- w: the width of the bounding box
- h: the height of the bounding box

Important: All position values (x, y, w, h) must be scaled to fit within the range [0, 1000]. This means that (0, 0) represents the top-left corner of the image, and (1000, 1000) represents the bottom-right corner.

Format your response as a JSON object with the following structure:

{{
  "content": "answer value",
  "position": [x, y, w, h]
}}

Remember:
1. The "content" field should contain the text answer to the question.
2. The "position" field should be an array of four integers representing the bounding box coordinates, all within the range [0, 1000].

Provide only the JSON response without any additional text or formatting.
"""

def build_prompt_with_anchors(question: str, words_ocr: list, image_size: Tuple[int, int]) -> str:
    width, height = image_size
    anchors = []

    sampled_words = words_ocr[:3] if len(words_ocr) >= 3 else words_ocr

    for word in sampled_words:
        text = word['text']
        x, y, w, h = word['bbox']
        anchor = f'The word "{text}" is at [{x}, {y}, {w}, {h}]'
        anchors.append(anchor)

    anchors_str = " | ".join(anchors)

    prompt = (
        f"Document dimensions: {width} x {height}.\n"
        f"Anchors:\n{anchors_str}\n\n"
        f"Question: {question}\n\n"
        "Provide ONLY a JSON response in the following format:\n"
        "{\n"
        '  "content": "answer",\n'
        '  "position": [x, y, w, h]\n'
        "}\n\n"
        "Each position value MUST be in the range [0, 1000]."
    )

    return prompt