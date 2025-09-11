import os
import re
import logging
from PIL import Image
from typing import Dict, Any, List, Union, Optional 

import numpy as np
import torch

import fiftyone as fo
from fiftyone import Model, SamplesMixin

from transformers import AutoModel, AutoTokenizer
from transformers.utils import is_flash_attn_2_available

logger = logging.getLogger(__name__)

DEFAULT_OCR_SYSTEM_PROMPT = "<ocr>"

DEFAULT_MD_SYSTEM_PROMPT = "<md>"

OPERATIONS = {
    "ocr": DEFAULT_OCR_SYSTEM_PROMPT,
    "md": DEFAULT_MD_SYSTEM_PROMPT,
}


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

class Kosmos2_5(Model):
    """A FiftyOne model for running MiniCPM-V 4.5 vision tasks"""

    def __init__(
        self,
        model_path: str,
        operation: str = None,
        system_prompt: str = None,
        **kwargs
    ):
        
        self.model_path = model_path
        self._operation = operation
        
        self.device = get_device()
        logger.info(f"Using device: {self.device}")
        
        # Load model and processor
        logger.info(f"Loading model from {model_path}")

        model_kwargs = {
            "device_map":self.device,
            }
        
        # Only set specific torch_dtype for CUDA devices
        if self.device == "cuda":
            model_kwargs["torch_dtype"] = torch.bfloat16

        model_kwargs["attn_implementation"] = "flash_attention_2" if is_flash_attn_2_available() else "sdpa"

        self.model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            **model_kwargs
            )

        logger.info("Loading tokenizer")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=True)
        self.model.eval()

    @property
    def media_type(self):
        return "image"
    
    @property
    def operation(self):
        return self._operation

    @operation.setter
    def operation(self, value):
        if value not in OPERATIONS:
            raise ValueError(f"Invalid operation: {value}. Must be one of {list(OPERATIONS.keys())}")
        self._operation = value

    @property
    def system_prompt(self):
        # Return custom system prompt if set, otherwise return default for current operation
        return OPERATIONS[self.operation]
    
    def _parse_box_tags(self, text: str) -> List[Dict]:
        """Parse bounding boxes from <ref>object</ref><box>x1 y1 x2 y2</box> format.
        
        Args:
            text: Model output containing ref and box tags
            
        Returns:
            List of dictionaries with bbox coordinates and labels
        """
        import re
        
        detections = []
        
        
        pattern = 
        
        matches = re.findall(pattern, text)
        
        for match in matches:
            label = match[0].strip()
            x1, y1, x2, y2 = match[1:5]
            
            detections.append({
                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                'label': label if label else "object"
            })
        
        return detections

    def _to_detections(self, text: str) -> fo.Detections:
        """Convert detection text output to FiftyOne Detections.
        
        Parses the <ref>object</ref><box>x1 y1 x2 y2</box> format and converts
        to FiftyOne's format with coordinate normalization from 0-1000 to 0-1 range.
        
        Args:
            text: Model output string containing ref and box tags
        
        Returns:
            fo.Detections: FiftyOne Detections object containing all converted detections
        """
        detections = []
        
        # Parse the box tags from the text
        parsed_boxes = self._parse_box_tags(text)
        
        # Process each bounding box
        for box in parsed_boxes:
            try:
                bbox = box['bbox']
                label = box['label']
                
                # Convert coordinates to float and normalize from 0-1000 to 0-1 range
                x1_norm, y1_norm, x2_norm, y2_norm = map(float, bbox)
                
                x = x1_norm 
                y = y1_norm 
                w = (x2_norm - x1_norm) 
                h = (y2_norm - y1_norm) 
                
                # Create FiftyOne Detection object
                detection = fo.Detection(
                    label=label,
                    bounding_box=[x, y, w, h]
                )
                detections.append(detection)
                    
            except Exception as e:
                print(f"Error processing box {box}: {e}")
                continue
                    
        return fo.Detections(detections=detections)
    
    def _predict(self, image: Image.Image, sample=None) -> Union[fo.Detections, str]:
        """Process a single image through the model and return predictions.
        
        This internal method handles the core prediction logic including:
        - Constructing the chat messages with system prompt and user query
        - Processing the image and text through the model
        - Parsing the output based on the operation type (detection/keypoint/classification/VQA)
        
        Args:
            image: PIL Image to process
            sample: Optional FiftyOne sample containing the image filepath
            
        Returns:
            One of:
            - fo.Detections: For object detection results
            - fo.Keypoints: For keypoint detection results  
            - fo.Classifications: For classification results
            - str: For VQA text responses
            
        Raises:
            ValueError: If no prompt has been set
        """
        
        if self.operation == "md":
            return output_text.strip()
        elif self.operation == "ocr":
            return self._to_detections(output_text)

    def predict(self, image, sample=None):
        """Process an image with the model.
        
        A convenience wrapper around _predict that handles numpy array inputs
        by converting them to PIL Images first.
        
        Args:
            image: PIL Image or numpy array to process
            sample: Optional FiftyOne sample containing the image filepath
            
        Returns:
            Model predictions in the appropriate format for the current operation
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        return self._predict(image, sample)