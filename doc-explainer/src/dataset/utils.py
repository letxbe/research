from typing import List
from collections import defaultdict
from typing import Dict, List 
from statistics import mean
from PIL import Image, ImageDraw, ImageFont

def convert_to_xyxy(box: List[int]) -> List[int]:
    """Convert from [left, top, width, height] to [x1, y1, x2, y2]"""
    if not box or len(box) != 4:
        return [0, 0, 0, 0] 
    w, h, x1, y1 = box
    return [x1, y1, x1 + w, y1 + h]

def union_boxes(boxes: List[List[int]]) -> List[int]:
    """Returns the union of all [left, top, width, height] boxes as [x1, y1, x2, y2]"""
    if not boxes:
        return [0, 0, 0, 0]
    
    try:
        boxes_xyxy = [convert_to_xyxy(b) for b in boxes]
        x1 = min(b[0] for b in boxes_xyxy)
        y1 = min(b[1] for b in boxes_xyxy)
        x2 = max(b[2] for b in boxes_xyxy)
        y2 = max(b[3] for b in boxes_xyxy)
    except Exception:
        return [0, 0, 0, 0]  # Fallback for corrupted box data

    return [x1, y1, x2, y2]

def scaledown_bbox(bbox: List[int], scale_factor: float = 1000.0):
    if len(bbox) != 4 or scale_factor == 0:
        return [0.0, 0.0, 0.0, 0.0]

    x1, y1, x2, y2 = bbox
    
    return [
        x1 / scale_factor, 
        y1 / scale_factor, 
        x2 / scale_factor, 
        y2 / scale_factor
    ]


def scale_bbox_to_image(
    bbox: List[int],
    img_size: tuple[int, int],
    scale_factor: float = 1000.0
) -> List[int]:
    
    if len(bbox) != 4 or len(img_size) != 2 or scale_factor == 0:
        return [0, 0, 0, 0]
    
    x1, y1, x2, y2 = bbox
    img_w, img_h = img_size

    if img_w <= 0 or img_h <= 0:
        return [0, 0, 0, 0]

    x1_scaled = int((x1 / scale_factor) * img_w)
    y1_scaled = int((y1 / scale_factor) * img_h)
    x2_scaled = int((x2 / scale_factor) * img_w)
    y2_scaled = int((y2 / scale_factor) * img_h)

    return [x1_scaled, y1_scaled, x2_scaled, y2_scaled]

    

def save_bbox(image: Image, locations: list[List[int]], question: str, gt_answer: str, save_path: str): 
    image_with_boxes = image.copy()
    draw = ImageDraw.Draw(image_with_boxes)

    # Draw individual GT boxes (blue)
    for box in locations:
        # Convert each word box from [left, top, w, h] to [x1,y1,x2,y2]
        box = convert_to_xyxy(box)
        box = scale_bbox_to_image(box, image.size)
        draw.rectangle(box, outline="blue", width=2)

    # Draw union GT box (green)
    box_gt = union_boxes(locations)
    box_gt = scale_bbox_to_image(box_gt, image.size)
    draw.rectangle(box_gt, outline="green", width=3)

    try:
        font = ImageFont.truetype("arial.ttf", size=24)
    except IOError:
        font = ImageFont.load_default()

    question_text = f"Q: {question}"
    answer_text = f"GT Answer: {gt_answer}"

    text_padding = 10

    # Calculate text height (using font.getbbox)
    q_bbox = font.getbbox(question_text)
    a_bbox = font.getbbox(answer_text)

    text_height = (q_bbox[3] - q_bbox[1]) + (a_bbox[3] - a_bbox[1]) + 3 * text_padding

    # Create new image with extra space on top
    new_width = image_with_boxes.width
    new_height = image_with_boxes.height + text_height

    image_with_text = Image.new("RGB", (new_width, new_height), "white")
    image_with_text.paste(image_with_boxes, (0, text_height))

    # Draw text on the extra space at the top
    draw_text = ImageDraw.Draw(image_with_text)
    draw_text.text((text_padding, text_padding), question_text, fill="black", font=font)
    draw_text.text((text_padding, q_bbox[3] - q_bbox[1] + 2 * text_padding), answer_text, fill="black", font=font)
    
    if save_path:
        image_with_text.save(save_path)



METRIC_KEYS = [
    'iou', 
    'center_distance', 
    'iou_05', 
    'iou_075', 
    'anls'
]

def compute_mean_metrics(metrics_per_source: Dict[str, Dict[str, List[float]]], questions_per_source: Dict[str, int], processed_per_source: Dict[str,int]) -> Dict:
    """
    Compute per-source, macro-average, and micro-average metrics.
    
    Args:
        metrics_per_source: source -> {metric_name -> list of values}
        
    Returns:
        Dictionary with:
            - 'per_source_metrics'
            - 'macro_averages' (avg over per-source means)
            - 'micro_averages' (avg over all individual values)
    """
    
    

    mean_metrics_per_source = {}
    all_metrics_flat = defaultdict(list)
    per_metric_means = defaultdict(list)

    for source, metric_lists in metrics_per_source.items(): 
        source_metrics = {
            'elements': len(next(iter(metric_lists.values()), []))  # Use the length of any metric
        }

        for key, values in metric_lists.items():
            if values:
                key_mean = mean(values)
                source_metrics[f'mean_{key}'] = key_mean
                
                per_metric_means[key].append(key_mean)
                all_metrics_flat[key].extend(values)        

        mean_metrics_per_source[source] = source_metrics
     
    # Compute macro averages (equal weight per source)   
    macro_averages = {'num_sources': len(mean_metrics_per_source)}
    for key, values in per_metric_means.items():
        macro_averages[f'overall_macro_mean_{key}'] = mean(values)
    
    # Compute micro averages (equal weight per data point)
    micro_averages = {}
    for key, values in all_metrics_flat.items():
        micro_averages[f'overall_micro_mean_{key}'] = mean(values)
        micro_averages[f'total_{key}_values'] = len(values)
    
    results = {
        'per_source_metrics': mean_metrics_per_source,
        'macro_averages': macro_averages,
        'micro_averages': micro_averages
    }
    
    results['success_rate_per_source'] = {}
    
    for source in questions_per_source:
        total = questions_per_source[source]
        processed = processed_per_source.get(source, 0)
        
        results['success_rate_per_source'][source] = {
            'total': total,
            'processed': processed,
            'skipped': total - processed,
            'success_rate': round(processed / total, 4) if total > 0 else 0.0
        }

    total_questions = sum(questions_per_source.values())
    processed_predictions = sum(processed_per_source.values())
    
    results['total_questions'] = total_questions
    results['processed_predictions'] = processed_predictions
    results['skipped_predictions'] = total_questions - processed_predictions
    results['overall_success_rate'] = round(processed_predictions / total_questions, 4) if total_questions > 0 else 0.0

    
    return results