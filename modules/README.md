# Modules — pipeline components

See project root README for full pipeline (Input image → Segmentation → Text → DrawIO XML).

## 📁 Structure (overview)

```
modules/
├── base.py, data_types.py
├── segmentation_info_extractor.py   # Segmentation
├── icon_picture_processor.py
├── basic_shape_processor.py
├── arrow_processor.py
├── xml_merger.py            # Merge fragments to DrawIO XML
├── metric_evaluator.py, refinement_processor.py
├── text/                    # OCR & text
└── utils/
```

## 🔄 Flow (high-level)

Input image → Segmentation → Icon/Shape/Arrow processing → XML merge → (optional) metric/refinement → Output DrawIO.

## 🚀 Quick start

### Install

```bash
pip install -r requirements.txt
```

### Run

```bash
python main.py -i input/flowchart.png
python main.py
python main.py -i input/flowchart.png --refine
```

### Use in code

```python
from modules import (
    SegmentationInfoExtractor,
    IconPictureProcessor,
    BasicShapeProcessor,
    ArrowProcessor,
    XMLMerger,
    ProcessingContext,
)

context = ProcessingContext(image_path="test.png")
extractor = SegmentationInfoExtractor()
result = extractor.process(context)
context.elements = result.elements
context.canvas_width = result.canvas_width
context.canvas_height = result.canvas_height

IconPictureProcessor().process(context)
BasicShapeProcessor().process(context)
ArrowProcessor().process(context)
result = XMLMerger().process(context)
print(result.metadata["output_path"])
```

### Extending

Processors inherit `BaseProcessor` and implement `process(context)`. Set `element.xml_fragment` and `element.layer_level` for each handled element. See `modules.data_types` for `LayerLevel` and `get_layer_level()`.

