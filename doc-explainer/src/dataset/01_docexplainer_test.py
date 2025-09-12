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

from src.docexplainer.docexplainer import DocExplainer
from src.docexplainer.metrics import compute_iou, compute_normalized_center_distance, compute_iou_with_threshold
from src.boundingDocs.utils import compute_mean_metrics
from src.boundingDocs.utils import METRIC_KEYS, union_boxes, scaledown_bbox



def parse_args(): 
    parser = argparse.ArgumentParser() 
    parser.add_argument('--vlm-model', type=str, default="smolvlm", help="Model Name")
    args = parser.parse_args()
    
    return args


if __name__ == '__main__':
    args = parse_args()
    dataset = load_dataset("letxbe/BoundingDocs", revision="v2.0", split='test')
       
    metrics_per_source = defaultdict(lambda: {key: [] for key in METRIC_KEYS})
    questions_per_source = defaultdict(int)
    processed_per_source = defaultdict(int)
    
    model_name = args.vlm_model 
    explainer = DocExplainer(
         vlm_model_name = model_name
    )
    
    for i, document in tqdm(enumerate(dataset), total=len(dataset), desc="Processing Documents"):
        
        source = document.get('source')
        doc_id = document.get('doc_id')
        images = document.get('doc_images', [])

        qa_data = json.loads(document.get('Q&A'))
              
        
        for index, (q_key, data) in tqdm(enumerate(qa_data.items(), start=1), desc=f'Processing:{source}-{doc_id}', total=len(qa_data)):
            questions_per_source[source] += 1 
        
            question = data.get('rephrased_question', data.get('question', ''))
            page_idx = data['answers'][0]['page'] - 1 
            image = images[page_idx]

            result = explainer([image], question)
            
            pred_bbox = None
            pred_answer = None
            
            if result: 
                pred_bbox = result.bbox
                pred_answer = result.answer
                
            if pred_answer is not None and pred_bbox is not None:
                processed_per_source[source] += 1
                
                
                gt_answer = data['answers'][0]['value']
                # Predicted bbox is in range(0,1) and gt bbox are in range[0,1000]
                locations = data['answers'][0].get('location', [])
                bbox_gt = scaledown_bbox(union_boxes(locations))
                
                iou = compute_iou(pred_bbox, bbox_gt)
                center_distance = compute_normalized_center_distance(pred_bbox, bbox_gt)
                iou_05 = compute_iou_with_threshold(pred_bbox, bbox_gt, threshold=0.5)
                iou_075 = compute_iou_with_threshold(pred_bbox, bbox_gt, threshold=0.75)
                anls = anls_score(pred_answer, gt_answer)
                
                metrics_per_source[source]['iou'].append(iou)
                metrics_per_source[source]['center_distance'].append(center_distance)
                metrics_per_source[source]['iou_05'].append(iou_05)
                metrics_per_source[source]['iou_075'].append(iou_075)
                metrics_per_source[source]['anls'].append(anls)

    results = compute_mean_metrics(metrics_per_source, questions_per_source, processed_per_source)
    
        
    with open(f'{model_name}_docexplainer_result.json', 'w') as f:
        json.dump(results, f, indent=2)