import os
import re
import logging
from PIL import Image
from typing import Dict, Any, List, Union, Optional 

import numpy as np
import torch

import fiftyone as fo
from fiftyone import Model

from transformers import AutoProcessor, Kosmos2_5ForConditionalGeneration
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
    """A FiftyOne model for running Kosmos-2.5 vision tasks.
    
    Automatically selects optimal dtype based on hardware:
    - bfloat16 for CUDA devices with compute capability 8.0+ (Ampere and newer)
    - float16 for older CUDA devices
    - float32 for CPU/MPS devices
    """

    def __init__(
        self,
        model_path: str,
        operation: str = None,
        system_prompt: str = None,
        torch_dtype: torch.dtype = None,
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
        
        # Set dtype based on device and user preference
        if torch_dtype is not None:
            self.dtype = torch_dtype
        elif self.device == "cuda" and torch.cuda.is_available():
            capability = torch.cuda.get_device_capability()
            # Enable bfloat16 on Ampere+ GPUs (compute capability 8.0+)
            if capability[0] >= 8:
                self.dtype = torch.bfloat16
                logger.info(f"Using bfloat16 dtype (compute capability {capability[0]}.{capability[1]})")
            else:
                self.dtype = torch.float16
                logger.info(f"Using float16 dtype (compute capability {capability[0]}.{capability[1]})")
        else:
            self.dtype = torch.float32  # Default for CPU/MPS
            logger.info(f"Using float32 dtype for {self.device}")
        
        # Only set torch_dtype in model_kwargs if not using default float32
        if self.dtype != torch.float32:
            model_kwargs["torch_dtype"] = self.dtype

        model_kwargs["attn_implementation"] = "flash_attention_2" if is_flash_attn_2_available() else "sdpa"

        self.model = Kosmos2_5ForConditionalGeneration.from_pretrained(
            model_path,
            trust_remote_code=True,
            **model_kwargs
            )

        logger.info("Loading processor")
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
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
        """Parse bounding boxes from <bbox><x_N><y_N><x_N><y_N></bbox>text format.
        
        Args:
            text: Model output containing bbox tags
            
        Returns:
            List of dictionaries with bbox coordinates and text labels
        """
        import re
        
        detections = []
        
        # Pattern to match bbox tags
        pattern = r"<bbox><x_\d+><y_\d+><x_\d+><y_\d+></bbox>"
        
        # Find all bbox patterns
        bboxs_raw = re.findall(pattern, text)
        
        # Split by pattern to get text parts
        lines = re.split(pattern, text)[1:]
        
        # Extract coordinates using list comprehension (like original)
        bboxs = [re.findall(r"\d+", i) for i in bboxs_raw]
        bboxs = [[int(j) for j in i] for i in bboxs]
        
        # Process each bbox with its corresponding text
        for i in range(len(lines)):
            box = bboxs[i]
            x1, y1, x2, y2 = box
            text_content = lines[i].strip()
            
            detections.append({
                'bbox': [x1, y1, x2, y2],
                'text': text_content
            })
        
        return detections

    def _to_detections(self, text: str) -> fo.Detections:
        """Convert detection text output to FiftyOne Detections.
        
        Parses the <bbox><x_N><y_N><x_N><y_N></bbox>text format and converts
        to FiftyOne's format with coordinate normalization to 0-1 range.
        
        Args:
            text: Model output string containing bbox tags
        
        Returns:
            fo.Detections: FiftyOne Detections object containing all converted detections
        """
        detections = []
        
        # Parse the box tags from the text
        parsed_boxes = self._parse_box_tags(text)
        
        # Process each bounding box
        for box in parsed_boxes:
            try:
                x1, y1, x2, y2 = box['bbox']
                text_content = box['text']
                
                # Skip invalid boxes where coordinates are reversed
                if x1 >= x2 or y1 >= y2:
                    continue
                
                # Scale coordinates back to original image space (pixel coordinates)
                # The model outputs coordinates in the preprocessed image space
                x1_pixel = x1 * self.scale_width
                y1_pixel = y1 * self.scale_height
                x2_pixel = x2 * self.scale_width
                y2_pixel = y2 * self.scale_height
                
                # Normalize to [0, 1] range for FiftyOne
                # FiftyOne expects [top-left-x, top-left-y, width, height] in [0, 1] range
                x_norm = x1_pixel / self.raw_width
                y_norm = y1_pixel / self.raw_height
                width_norm = (x2_pixel - x1_pixel) / self.raw_width
                height_norm = (y2_pixel - y1_pixel) / self.raw_height
                
                # Ensure coordinates are within [0, 1] bounds
                x_norm = max(0, min(1, x_norm))
                y_norm = max(0, min(1, y_norm))
                width_norm = max(0, min(1 - x_norm, width_norm))
                height_norm = max(0, min(1 - y_norm, height_norm))
                
                # Create FiftyOne Detection object with text as label
                detection = fo.Detection(
                    label="text",
                    bounding_box=[x_norm, y_norm, width_norm, height_norm]
                )
                # Store the actual text content as an attribute
                if text_content:
                    detection["text_content"] = text_content
                detections.append(detection)
                    
            except Exception as e:
                logger.warning(f"Error processing box {box}: {e}")
                continue
                    
        return fo.Detections(detections=detections)
    
    def _predict(self, image: Image.Image) -> Union[fo.Detections, str]:
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
        
        # Get the prompt based on the operation
        prompt = self.system_prompt
        
        # Process the image and text through the processor
        inputs = self.processor(text=prompt, images=image, return_tensors="pt")
        
        # Extract height and width for scaling (needed for OCR mode)
        height, width = inputs.pop("height"), inputs.pop("width")
        raw_width, raw_height = image.size
        self.scale_height = raw_height / height
        self.scale_width = raw_width / width
        self.raw_width = raw_width
        self.raw_height = raw_height
        
        # Move inputs to device and set proper dtype
        inputs = {k: v.to(self.device) if v is not None else None for k, v in inputs.items()}
        inputs["flattened_patches"] = inputs["flattened_patches"].to(self.dtype)
        
        # Generate predictions
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=1024,
            )
        
        # Decode the generated text
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
        output_text = generated_text[0]
        
        # Remove the prompt from the output
        output_text = output_text.replace(prompt, "").strip()
        
        # Return based on operation type
        if self.operation == "md":
            return output_text
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