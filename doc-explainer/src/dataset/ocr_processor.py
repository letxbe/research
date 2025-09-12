from typing import Dict, Any, List, Optional, Union
import json 

class OCRProcessor:
    @staticmethod
    def scale_to_1000(value: float) -> int:
        scaled = int(value * 1000)
        return max(0, min(1000, scaled))
    
    @classmethod
    def extract_blocks_from_ocr(cls, sample: Dict[str, Any]) -> List[Dict[str, Any]]:
        try:
            ocr_raw = sample.get('doc_ocr', ['{}'])[0]
            ocr_json = json.loads(ocr_raw)
            
            if type(ocr_json) is list: 
                return ocr_json[0].get("Blocks", ocr_json)
            return ocr_json.get('Blocks', ocr_json)
        except (json.JSONDecodeError, IndexError) as e:
            print(f"Error extracting OCR blocks: {e}")
            return []
    
    @classmethod
    def process_block(cls, block: Dict[str, Any]) -> Optional[str]:
        if not all(key in block for key in ['Geometry', 'Text']) or 'BoundingBox' not in block.get('Geometry', {}):
            return None
        if block.get('BlockType') in ['LINE', 'WORD']:
            text = block['Text']
            box = block['Geometry']['BoundingBox']

            left = cls.scale_to_1000(box['Left'])
            top = cls.scale_to_1000(box['Top'])
            width = cls.scale_to_1000(box['Width'])
            height = cls.scale_to_1000(box['Height'])

            # center_x = left + (width // 2)
            # center_y = top + (height // 2)
            # coords = [
            #     str(center_x), 
            #     str(center_y)
            # ]
            coords = [
                str(left),
                str(top), 
                str(width),
                str(height),
            ]

            

            return f"{text};{' '.join(coords)}"
              
        return None
    
    @classmethod
    def extract_lines_from_blocks(cls, blocks: Union[List[Dict[str, Any]], Dict[str, List[Dict[str, Any]]]]) -> str:

        lines = []

        if isinstance(blocks, dict) and any(key in blocks for key in ['LINE', 'WORD']):
            if 'LINE' in blocks and blocks['LINE']:
                for block in blocks['LINE']:
                    line = cls.process_block(block)
                    if line: 
                        lines.append(line)
            elif 'WORD' in blocks and blocks['WORD']:
                for block in blocks['WORD']:
                    line = cls.process_block(block)
                    if line: 
                        lines.append(line)
        
        elif isinstance(blocks, list):
            for block in blocks: 
                line = cls.process_block(block)
                if line: 
                    lines.append(line)        

        return "\n".join(lines)

    @classmethod 
    def extract_words_and_bboxes(
        cls,
        blocks: Union[List[Dict[str, Any]], Dict[str, List[Dict[str, Any]]]]
    ) -> List[Dict[str, Any]]:
        res = []
        
        
        if isinstance(blocks, dict): # MP-DocVQA has different structure
            all_blocks = []
            for key in ("LINE", "WORD"):
                if key in blocks and isinstance(blocks[key], list):
                    all_blocks.extend(blocks[key])
        elif isinstance(blocks, list):
            all_blocks = blocks
        else:
            return res
        
        for block in all_blocks: 
            if not isinstance(block, dict):
                continue 
            
            if not all(k in block for k in ['Text', 'Geometry']):
                continue 
            
            if 'BoundingBox' not in block.get('Geometry', {}):
                continue     
                
            text = block['Text']
            box = block['Geometry']['BoundingBox']
            page = block.get('Page', 1)
            
            left = cls.scale_to_1000(box['Left'])
            top = cls.scale_to_1000(box['Top'])
            width = cls.scale_to_1000(box['Width'])
            height = cls.scale_to_1000(box['Height'])
            
            res.append({
                "page": page,
                "text": text,
                "bbox": [left, top, width, height]
            })
                
        return res