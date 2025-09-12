from typing import List

from PIL import ImageDraw
from PIL.Image import Image
from pydantic import BaseModel


class ExplainableAnswer(BaseModel):
    """
    A model representing an explainable answer extracted from a document.

    This class contains the answer text, page number, and bounding box coordinates
    for a piece of information extracted from a document image. It provides
    functionality to visualize the answer by drawing bounding boxes on the image.

    Attributes:
        answer (str): The extracted text answer from the document.
        page (int): The page number where the answer was found.
        bbox (List[float]): Normalized bounding box coordinates [x0, y0, x1, y1]
                           where values are between 0 and 1.
    """

    answer: str
    page: int
    bbox: List[float]

    def __str__(self) -> str:
        return f"Answer: {self.answer} | Page: {self.page} | BBox: {self.bbox}"

    def explain(
        self,
        image: Image,
        color: str = "red",
        width: int = 2,
        save_path: str = "example.png",
    ) -> None:
        """
        Draws the bounding box over the image and saves it.
        """
        draw = ImageDraw.Draw(image)
        img_w, img_h = image.size

        x0 = self.bbox[0] * img_w
        y0 = self.bbox[1] * img_h
        x1 = self.bbox[2] * img_w
        y1 = self.bbox[3] * img_h

        draw.rectangle([x0, y0, x1, y1], outline=color, width=width)
        image.show()
        image.save(save_path)
        print(f"Image saved to {save_path}")
