# Kosmos-2.5 for FiftyOne

A FiftyOne model plugin for [Microsoft's Kosmos-2.5](https://huggingface.co/microsoft/kosmos-2.5), a multimodal literate model for machine reading of text-intensive images. Kosmos-2.5 excels at two core tasks: generating spatially-aware text blocks (OCR) and producing structured markdown output from images.

## Overview

Kosmos-2.5 is a multimodal large language model designed for document-level text recognition and understanding. Unlike traditional OCR systems, it can:
- Perform high-quality text detection with precise bounding boxes
- Generate clean markdown-formatted text from complex documents
- Handle diverse document types including PDFs, receipts, forms, and handwritten text
- Process text-intensive images with multiple columns, tables, and mixed layouts

## Features

- **Dual Operation Modes**:
  - `ocr`: Extracts text with bounding box coordinates
  - `md`: Generates structured markdown from document images
  
- **Automatic Hardware Optimization**:
  - Intelligently selects dtype based on GPU compute capability
  - Uses bfloat16 for modern GPUs (Ampere+), float16 for older GPUs
  - Falls back to float32 for CPU inference

- **Seamless FiftyOne Integration**:
  - Works with `apply_model()` for batch processing
  - Returns proper FiftyOne detection format with normalized coordinates
  - Compatible with all FiftyOne visualization and analysis tools

## Installation

### Prerequisites

```bash
pip install fiftyone
pip install transformers torch torchvision
pip install huggingface-hub
```

### Register the Model

```python
import fiftyone.zoo as foz

# Register the Kosmos-2.5 model source
foz.register_zoo_model_source(
    "https://github.com/harpreetsahota204/kosmos2_5", 
    overwrite=True
)
```

## Usage

### Basic Example

```python
import fiftyone as fo
import fiftyone.zoo as foz

# Load the model
model = foz.load_zoo_model("microsoft/kosmos-2.5")

# Create or load a dataset
dataset = fo.Dataset("my_documents")
# ... add samples to dataset ...

# OCR Mode - Extract text with bounding boxes
model.operation = "ocr"
dataset.apply_model(model, label_field="text_detections")

# Markdown Mode - Generate structured text
model.operation = "md"
dataset.apply_model(model, label_field="text_extraction")

# Visualize results
session = fo.launch_app(dataset)
```

### Working with PDFs

```python
import fiftyone as fo
import fiftyone.operators as foo

# Install PDF loader plugin
!fiftyone plugins download https://github.com/harpreetsahota204/pdf-loader --overwrite

# Load PDF as images
pdf_loader = foo.get_operator("@brimoor/pdf-loader/pdf_loader")
pdf_dataset = fo.Dataset("pdf_dataset")

pdf_loader(
    pdf_dataset,
    input_path="document.pdf",
    output_dir="./pdf_images",
    dpi=200,
    fmt="png"
)

# Apply Kosmos-2.5
model = foz.load_zoo_model("microsoft/kosmos-2.5")
model.operation = "ocr"
pdf_dataset.apply_model(model, label_field="text_detections")
```

### Loading from Hugging Face Hub

```python
import fiftyone.utils.huggingface as fouh

# Load a text detection dataset
dataset = fouh.load_from_hub("Voxel51/Total-Text-Dataset", max_samples=10)

# Apply OCR
model.operation = "ocr"
dataset.apply_model(model, label_field="text_detections")

# Extract markdown
model.operation = "md"
dataset.apply_model(model, label_field="text_extraction")
```

## Output Formats

### OCR Mode
Returns `fo.Detections` with:
- Bounding boxes in normalized coordinates [0, 1]
- Format: `[top-left-x, top-left-y, width, height]`
- Each detection labeled as "text" with actual content in `text_content` attribute

### Markdown Mode
Returns raw markdown-formatted text string containing:
- Preserved document structure
- Tables, lists, and formatting
- Clean, readable text output

## Model Details

- **Architecture**: Multimodal transformer with vision and language encoders
- **Parameters**: ~900M
- **Input**: Images of any size (automatically preprocessed)
- **Training Data**: Large-scale document and text recognition datasets
- **License**: MIT

## Hardware Requirements

- **Minimum**: 8GB GPU memory (with float16)
- **Recommended**: 16GB+ GPU memory for optimal performance
- **CPU Inference**: Supported but significantly slower

## Advanced Configuration

```python
import torch

# Manually specify dtype
model = foz.load_zoo_model(
    "microsoft/kosmos-2.5",
    torch_dtype=torch.float16  # Force specific precision
)

# The model automatically detects hardware capabilities:
# - bfloat16: NVIDIA Ampere and newer (RTX 30xx, 40xx, A100, H100)
# - float16: Older CUDA GPUs (RTX 20xx, GTX series)
# - float32: CPU and Apple Silicon
```

## Citation

If you use Kosmos-2.5 in your research, please cite:

```bibtex
@article{lv2023kosmos,
  title={Kosmos-2.5: A multimodal literate model},
  author={Lv, Tengchao and Huang, Yupan and Chen, Jingye and Cui, Lei and Ma, Shuming and Chang, Yaoyao and Huang, Shaohan and Wang, Wenhui and Dong, Li and Luo, Weiyao and others},
  journal={arXiv preprint arXiv:2309.11419},
  year={2023}
}

```

## License

This integration is released under the MIT License. The Kosmos-2.5 model itself is also MIT licensed.

## Acknowledgments

- Microsoft Research for developing and open-sourcing Kosmos-2.5
- The FiftyOne team for the excellent computer vision framework
- The Hugging Face team for model hosting and transformers library