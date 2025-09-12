from datasets import load_dataset
from tqdm import tqdm
from collections import defaultdict
from anls_star import anls_score

import argparse
import json 
import sys 
import os 

# Add project root to path to enable local imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.models.utils import get_model_and_processor, generate_prediction
from src.docexplainer.metrics import compute_iou, compute_normalized_center_distance, compute_iou_with_threshold
from src.boundingDocs.utils import compute_mean_metrics
from src.boundingDocs.utils import METRIC_KEYS, union_boxes, save_bbox
from src.boundingDocs.prompt import ZERO_SHOT_PROMPT, COT_ONE_SHOT_PROMPT, CLAUDE_PROMPT,  build_prompt_with_anchors
from src.boundingDocs.OCR_Processor import OCRProcessor


def parse_args(): 
    parser = argparse.ArgumentParser() 
    parser.add_argument('--vlm-model', type=str, default="smolvlm", help="Model Name")
    parser.add_argument('--mode', type=str, choices=['cot', 'zero_shot', 'anchors'], default='zero_shot', help="Evaluation Mode")
    parser.add_argument('--draw-bbox', action='store_true', help="Save BBox images")
    return parser.parse_args()


def run_evaluation(args):
    dataset = load_dataset("letxbe/BoundingDocs", revision="v2.0", split="test")
   
    model_name = args.vlm_model 
    mode = args.mode
    draw_bbox = args.draw_bbox
    
    model, processor = get_model_and_processor(model_name)
    
    metrics_per_source = defaultdict(lambda: {key: [] for key in METRIC_KEYS})
    questions_per_source = defaultdict(int)
    processed_per_source = defaultdict(int)
    
    total_docs = len(dataset)

    for i, document in tqdm(enumerate(dataset), total=total_docs, desc="Processing Documents"):
        
        source = document.get('source')
        doc_id = document.get('doc_id')
        images = document.get('doc_images', [])

        qa_data = json.loads(document.get('Q&A'))
        previous_qa_chain = [] 
        
        words_ocr = None 
        if mode == 'anchors':
            # Retrieve OCR words and bounding boxes 
            blocks = OCRProcessor.extract_blocks_from_ocr(document)
            words_ocr = OCRProcessor.extract_words_and_bboxes(blocks)
            
    
        qa_bar = tqdm(enumerate(qa_data.items(), start=1), total=len(qa_data), leave=False)
        iou, anls = 0.0, 0.0

        for index, (q_key, data) in qa_bar :
            qa_bar.set_description(f"Processing {source}-{doc_id}, IoU={iou:.3f}, ANLS={anls:.3f}")
            questions_per_source[source] += 1 
            
            # Retain only the last 3 Q&A pairs for context
            if len(previous_qa_chain) > 3:
                previous_qa_chain = previous_qa_chain[-3:]
            
            question = data.get('rephrased_question', data.get('question', ''))
            page_idx = data['answers'][0]['page'] - 1 
            image = images[page_idx]

            locations = data['answers'][0].get('location', [])
            bbox_gt = union_boxes(locations)
            
            # Default zero shot prompt 
            prompt = ZERO_SHOT_PROMPT.format(QUESTION=question)
            if model_name == 'claude-sonnet-4':
                prompt = CLAUDE_PROMPT.format(QUESTION=question)
            
            if mode == 'cot':
                if len(previous_qa_chain) > 0:
                    cot_context = "Chain of Thought:\n"
                    for prev in previous_qa_chain:
                        cot_context += (
                            f"Q: {prev['question']}\n"
                            f"A: {{\"value\": \"{prev['answer_value']}\", \"position\": {prev['position']}}}\n"
                        )
                    question_for_model = cot_context + f"\nCurrent Question:\n{question}\n"
                else: 
                    question_for_model = question
                prompt = COT_ONE_SHOT_PROMPT.format(QUESTION=question_for_model)

            elif mode == 'anchors' and words_ocr:
                prompt = build_prompt_with_anchors(question, words_ocr, image.size)
    
            prediction = generate_prediction(prompt, image, model_name, model, processor)
            
            if isinstance(prediction, dict):
                pred_answer = prediction.get('content', None)
                bbox_pred = prediction.get('position', None)
                
                if isinstance(bbox_pred, list) and len(bbox_pred) == 4 and pred_answer:
                    processed_per_source[source] += 1
                    
                    x,y,w,h = bbox_pred
                    bbox_pred = [x, y, x+w, y+h]
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
                    
                    if draw_bbox: 
                        safe_source = source.replace("/", "_").replace("\\", "_").replace(":", "_")
                        safe_doc_id = doc_id.replace("/", "_").replace("\\", "_").replace(":", "_")
                        save_path = f"annotated_results/{safe_source}_{safe_doc_id}_q{index}_debug.png"
                        save_bbox(image, locations, question, gt_answer,  save_path )

            
                    if mode == 'cot':
                        previous_qa_chain.append({
                            "question": question,
                            "answer_value": gt_answer,
                            "position": bbox_gt
                        })
           
    
    results = compute_mean_metrics(metrics_per_source, questions_per_source, processed_per_source)
    
    result_file = f'{model_name}_{mode}_result.json'
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✅ Final Results saved to {result_file}")


if __name__ == '__main__':
    args = parse_args()
    run_evaluation(args)
