from datasets import load_dataset
from tqdm import tqdm
from collections import defaultdict
from anls_star import anls_score
from typing import List, Dict, Optional
from difflib import SequenceMatcher

import argparse
import json 
import sys 
import os 

# Add project root to path to enable local imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.models.utils import get_model_and_processor, generate_prediction
from src.docexplainer.metrics import compute_iou, compute_normalized_center_distance, compute_iou_with_threshold
from src.boundingDocs.utils import compute_mean_metrics
from src.boundingDocs.utils import METRIC_KEYS, union_boxes
from src.boundingDocs.OCR_Processor import OCRProcessor

PROMPT = """Based only on the document image, answer the following question:

Question: {QUESTION}

Provide ONLY a JSON response in the following format:
{{
  "content": "answer",
}}
"""


def parse_args(): 
    parser = argparse.ArgumentParser() 
    parser.add_argument('--vlm-model', type=str, default="smolvlm", help="Model Name")
    args = parser.parse_args()
    return args


def fuzzy_match(a: str, b: str, threshold: float = 0.6) -> bool:
    """Return True if two strings are similar enough."""
    return SequenceMatcher(None, a.lower().strip(), b.lower().strip()).ratio() >= threshold


def find_best_word_bbox(pred_answer: str | int, words_ocr: List[Dict], page_idx: int) -> Optional[List[int]]:
    """
    Try fuzzy match for the full answer first. If not found, fall back to first word.
    """
    def match_word(target: str | int) -> Optional[List[int]]:
        """Helper to match a single word/phrase against OCR words."""
        target = str(target).strip()
        best_match = None
        best_score = 0

        for w in words_ocr:
            if w.get('page') - 1 != page_idx:
                continue

            ocr_word = str(w.get("text", "")).strip()
           
            score = SequenceMatcher(None, target.lower(), ocr_word.lower()).ratio()

            if score > best_score and score >= 0.6:
                best_score = score
                bbox = w["bbox"]
                x, y, w_, h_ = bbox 
                best_match = [x, y, x + w_, y + h_]
                
        return best_match

    # 1. Try full answer fuzzy match
    bbox = match_word(pred_answer)
    if bbox:
        return bbox

    # 2. Fall back to first word
    if isinstance(pred_answer, str):
        first_word = pred_answer.split()[0] 
        return match_word(first_word)


if __name__ == '__main__':
    args = parse_args()
    dataset = load_dataset("letxbe/BoundingDocs", revision="v2.0", split='test')
   
    model_name = args.vlm_model 
    model, processor = get_model_and_processor(model_name)
    
    metrics_per_source = defaultdict(lambda: {key: [] for key in METRIC_KEYS})
    questions_per_source = defaultdict(int)
    processed_per_source = defaultdict(int)


    for i, document in tqdm(enumerate(dataset), total=len(dataset), desc="Processing Documents"):
    
        source = document.get('source') 
        doc_id = document.get('doc_id')
        images = document.get('doc_images', [])

        qa_data = json.loads(document.get('Q&A'))
        
        blocks = OCRProcessor.extract_blocks_from_ocr(document)
        words_ocr = OCRProcessor.extract_words_and_bboxes(blocks)

        for index, (q_key, data) in tqdm(enumerate(qa_data.items(), start=1),
                                         desc=f'Processing:{source}-{doc_id}',
                                         total=len(qa_data)):
            questions_per_source[source] += 1


            question = data.get('rephrased_question', data.get('question', ''))
            page_idx = data['answers'][0]['page'] - 1 
            image = images[page_idx]

            locations = data['answers'][0].get('location', [])
            bbox_gt = union_boxes(locations)
            
            prompt = PROMPT.format(QUESTION=question)
            prediction = generate_prediction(prompt, image, model_name, model, processor)

            if not isinstance(prediction, dict):
                continue

            pred_answer = prediction.get('content', None)
            
            if not pred_answer:
                continue

            bbox_pred = find_best_word_bbox(pred_answer, words_ocr, page_idx)
            if not bbox_pred:
                continue
            
            if not (isinstance(bbox_pred, list) and len(bbox_pred) == 4):
                continue

            processed_per_source[source] += 1
            
            iou = compute_iou(bbox_pred, bbox_gt)
            center_distance = compute_normalized_center_distance(bbox_pred, bbox_gt)
            iou_05 = compute_iou_with_threshold(bbox_pred, bbox_gt, threshold=0.5)
            iou_075 = compute_iou_with_threshold(bbox_pred, bbox_gt, threshold=0.75)

            gt_answer = data['answers'][0]['value']
            anls = anls_score(pred_answer, gt_answer)
            
            metrics_per_source[source]['iou'].append(iou)
            metrics_per_source[source]['center_distance'].append(center_distance)
            metrics_per_source[source]['iou_05'].append(iou_05)
            metrics_per_source[source]['iou_075'].append(iou_075)
            metrics_per_source[source]['anls'].append(anls)


    results = compute_mean_metrics(metrics_per_source, questions_per_source, processed_per_source)
    
    with open(f'{model_name}_first_word_ocr_result.json', 'w') as f:
        json.dump(results, f, indent=2)
