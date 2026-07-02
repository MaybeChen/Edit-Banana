import os
import sys
import argparse
import warnings
import yaml
import html
import xml.etree.ElementTree as ET
from xml.dom import minidom
from pathlib import Path
from typing import Optional, List, Dict, Any

os.environ.setdefault('PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK', 'True')
warnings.filterwarnings('ignore', message=".*doesn't match a supported version.*")

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from modules import TextRestorer, ProcessingContext, ElementInfo
from modules.vlm_enhancer import VLMEnhancer

TEXT_MODULE_AVAILABLE = TextRestorer is not None


def load_config() -> dict:
    config_path = os.path.join(PROJECT_ROOT, 'config', 'config.yaml')
    if not os.path.exists(config_path):
        print(f'Warning: config file not found at {config_path}, using defaults')
        return {
            'paths': {'input_dir': './input', 'output_dir': './output'},
            'multimodal': {'enabled': False},
            'ocr': {
                'engine': 'paddleocr',
                'paddleocr': {
                    'lang': 'ch',
                    'use_angle_cls': False,
                    'allow_download': True,
                    'allow_fallback_to_tesseract': False,
                    'allow_legacy_fallback': False,
                    'ocr_version': 'PP-OCRv6',
                    'text_detection_model_name': 'PP-OCRv6_medium_det',
                    'text_recognition_model_name': 'PP-OCRv6_medium_rec',
                    'textline_orientation_model_name': 'PP-LCNet_x1_0_textline_ori',
                    'text_det_limit_side_len': 64,
                    'text_det_limit_type': 'min',
                    'text_det_thresh': 0.3,
                    'text_det_box_thresh': 0.6,
                    'text_det_unclip_ratio': 1.5,
                    'text_rec_score_thresh': 0.0,
                    'scale': 1.0,
                    'min_confidence': 0.3,
                },
            },
        }
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


class Pipeline:
    """Image-to-draw.io pipeline: OCR text plus optional VLM structure."""

    def __init__(self, config: dict = None):
        self.config = config or load_config()
        self._text_restorer = None
        self._vlm_enhancer = None

    @property
    def text_restorer(self):
        if self._text_restorer is None and TextRestorer is not None:
            ocr_config = self.config.get('ocr') or {}
            ocr_engine = os.environ.get('OCR_ENGINE', ocr_config.get('engine', 'paddleocr'))
            self._text_restorer = TextRestorer(
                formula_engine='none',
                ocr_engine=ocr_engine,
                ocr_config=ocr_config,
            )
        return self._text_restorer

    @property
    def vlm_enhancer(self) -> VLMEnhancer:
        if self._vlm_enhancer is None:
            self._vlm_enhancer = VLMEnhancer(self.config)
        return self._vlm_enhancer

    def process_image(self, image_path: str, output_dir: str = None, with_text: bool = True) -> Optional[str]:
        print(f"\n{'=' * 60}")
        print(f'Processing: {image_path}')
        print(f"{'=' * 60}")

        if output_dir is None:
            output_dir = self.config.get('paths', {}).get('output_dir', './output')
        img_stem = Path(image_path).stem
        img_output_dir = os.path.join(output_dir, img_stem)
        os.makedirs(img_output_dir, exist_ok=True)
        output_path = os.path.join(img_output_dir, f'{img_stem}.drawio')

        context = ProcessingContext(image_path=image_path, output_dir=img_output_dir)
        self._initialize_canvas_from_image(context)
        text_blocks: List[Dict[str, Any]] = []

        try:
            if with_text and self.text_restorer is not None:
                print('\n[1] OCR text/formula restoration...')
                text_blocks = self.text_restorer.process_image(image_path)
                for idx, block in enumerate(text_blocks):
                    block.setdefault('id', idx)
                context.intermediate_results['ocr_text_blocks'] = text_blocks
                print(f'   Text blocks: {len(text_blocks)}')

                if hasattr(self.text_restorer, 'save_ocr_artifacts'):
                    ocr_artifact_path = self.text_restorer.save_ocr_artifacts(img_output_dir, image_path)
                    context.intermediate_results['ocr_result_json'] = ocr_artifact_path
                    print(f'   OCR result: {ocr_artifact_path}')
                if hasattr(self.text_restorer, 'save_ocr_overlay'):
                    ocr_overlay_path = self.text_restorer.save_ocr_overlay(img_output_dir, image_path)
                    context.intermediate_results['ocr_overlay'] = ocr_overlay_path
                    print(f'   OCR overlay: {ocr_overlay_path}')
            elif with_text:
                print('\n[1] OCR text/formula restoration skipped: TextRestorer unavailable')
            else:
                print('\n[1] OCR text/formula restoration skipped')

            if text_blocks:
                print('\n[2] VLM text style enrichment...')
                styled = self.vlm_enhancer.enrich_text_styles(context, text_blocks)
                text_blocks = styled.get('text_blocks', text_blocks)
                context.intermediate_results['ocr_text_blocks'] = text_blocks
                self._save_json(os.path.join(img_output_dir, 'vlm_text_result.json'), styled)
                print(f"   VLM text styles: {styled.get('updated', 0)} updated")

            print('\n[3] VLM structure recognition...')
            structure_result = self.vlm_enhancer.recognize_structure_staged(context)
            if not structure_result.get('recognized'):
                print(f"   VLM structure skipped: {structure_result.get('error', 'not recognized')}")
            print(f'   VLM elements: {len(context.elements)}')

            if context.elements:
                print('\n[4] VLM element attribute enrichment...')
                attr_result = self.vlm_enhancer.enrich_element_attributes(context)
                self._save_json(os.path.join(img_output_dir, 'vlm_element_attributes.json'), attr_result)
                print(f"   VLM element attributes: {attr_result.get('updated', 0)} updated")

            print('\n[5] DrawIO generation...')
            drawio_xml = self._generate_drawio_xml(context, text_blocks)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(drawio_xml)
            print(f'   DrawIO: {output_path}')

            validation = self.vlm_enhancer.validate_export(context, output_path)
            if validation.get('validated'):
                print(f"   VLM export validation score: {validation.get('score', 'n/a')}")

            print(f"\n{'=' * 60}\nDone.\n{'=' * 60}")
            return output_path
        except Exception as e:
            print(f'\nFailed: {e}')
            import traceback
            traceback.print_exc()
            return None

    @staticmethod
    def _initialize_canvas_from_image(context: ProcessingContext) -> None:
        from PIL import Image
        with Image.open(context.image_path) as img:
            context.canvas_width, context.canvas_height = img.size

    @staticmethod
    def _save_json(path: str, data: dict) -> str:
        import json
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return path

    def _generate_drawio_xml(self, context: ProcessingContext, text_blocks: List[Dict[str, Any]]) -> str:
        mxfile = ET.Element('mxfile', {
            'host': 'app.diagrams.net',
            'modified': '2024-01-01T00:00:00.000Z',
            'agent': 'Edit Banana VLM DrawIO Export',
            'version': '1.0.0',
            'type': 'device',
        })
        diagram = ET.SubElement(mxfile, 'diagram', {'name': 'Page-1', 'id': 'diagram-1'})
        graph = ET.SubElement(diagram, 'mxGraphModel', {
            'dx': '0', 'dy': '0', 'grid': '1', 'gridSize': '10', 'guides': '1',
            'tooltips': '1', 'connect': '1', 'arrows': '1', 'fold': '1', 'page': '1',
            'pageScale': '1', 'pageWidth': str(context.canvas_width or 1169),
            'pageHeight': str(context.canvas_height or 827), 'math': '1',
        })
        root = ET.SubElement(graph, 'root')
        ET.SubElement(root, 'mxCell', {'id': '0'})
        ET.SubElement(root, 'mxCell', {'id': '1', 'parent': '0'})

        next_id = 2
        for elem in sorted(context.elements, key=self._element_sort_key):
            next_id = self._add_element_cell(root, elem, next_id)
        for block in text_blocks:
            next_id = self._add_text_cell(root, block, next_id)

        xml_string = ET.tostring(mxfile, encoding='unicode')
        pretty_xml = minidom.parseString(xml_string).toprettyxml(indent='  ')
        lines = pretty_xml.split('\n')
        if lines and lines[0].startswith('<?xml'):
            lines = lines[1:]
        return '\n'.join(line for line in lines if line.strip())

    @staticmethod
    def _element_sort_key(elem: ElementInfo) -> tuple:
        type_order = {'background': 0, 'container': 1, 'rectangle': 2, 'rounded_rectangle': 2, 'rounded rectangle': 2, 'ellipse': 2, 'circle': 2, 'diamond': 2, 'cylinder': 2, 'arrow': 3, 'connector': 3, 'line': 3, 'text': 4}
        return (type_order.get(str(elem.element_type).lower(), 2), -getattr(elem.bbox, 'area', 0), elem.id)

    def _add_element_cell(self, root: ET.Element, elem: ElementInfo, cell_id: int) -> int:
        elem_type = str(elem.element_type or 'rectangle').lower()
        if elem_type == 'text':
            meta = getattr(elem, 'vlm_item', {}) or {}
            text = meta.get('content') or meta.get('text') or meta.get('label') or ''
            if text:
                block = {'text': text, 'geometry': {'x': elem.bbox.x1, 'y': elem.bbox.y1, 'width': elem.bbox.width, 'height': elem.bbox.height}, 'font_size': meta.get('font_size') or 12}
                return self._add_text_cell(root, block, cell_id)
            return cell_id
        if elem_type in {'arrow', 'connector', 'line'}:
            return self._add_edge_cell(root, elem, cell_id)

        style = self._shape_style(elem)
        value = self._element_label(elem)
        cell = ET.SubElement(root, 'mxCell', {'id': str(cell_id), 'value': html.escape(value), 'style': style, 'vertex': '1', 'parent': '1'})
        ET.SubElement(cell, 'mxGeometry', {'x': str(elem.bbox.x1), 'y': str(elem.bbox.y1), 'width': str(max(1, elem.bbox.width)), 'height': str(max(1, elem.bbox.height)), 'as': 'geometry'})
        return cell_id + 1

    def _add_edge_cell(self, root: ET.Element, elem: ElementInfo, cell_id: int) -> int:
        bbox = elem.bbox
        start = elem.arrow_start or (bbox.x1, bbox.y1 + bbox.height / 2)
        end = elem.arrow_end or (bbox.x2, bbox.y1 + bbox.height / 2)
        end_arrow = 'none' if elem.element_type == 'line' or elem.arrow_heads == 'none' else 'classic'
        style = f'endArrow={end_arrow};html=1;rounded=0;strokeColor={elem.stroke_color or "#000000"};strokeWidth={max(1, int(elem.stroke_width or 1))};'
        if elem.line_style in {'dashed', 'dotted'}:
            style += 'dashed=1;'
        cell = ET.SubElement(root, 'mxCell', {'id': str(cell_id), 'value': '', 'style': style, 'edge': '1', 'parent': '1'})
        geometry = ET.SubElement(cell, 'mxGeometry', {'relative': '1', 'as': 'geometry'})
        ET.SubElement(geometry, 'mxPoint', {'x': str(round(start[0], 2)), 'y': str(round(start[1], 2)), 'as': 'sourcePoint'})
        ET.SubElement(geometry, 'mxPoint', {'x': str(round(end[0], 2)), 'y': str(round(end[1], 2)), 'as': 'targetPoint'})
        return cell_id + 1

    @staticmethod
    def _shape_style(elem: ElementInfo) -> str:
        elem_type = str(elem.element_type or '').lower()
        shape_map = {'ellipse': 'ellipse', 'circle': 'ellipse', 'diamond': 'rhombus', 'cylinder': 'cylinder', 'cloud': 'cloud'}
        parts = ['rounded=0', 'whiteSpace=wrap', 'html=1']
        if elem_type in {'rounded_rectangle', 'rounded rectangle', 'container', 'background'}:
            parts[0] = 'rounded=1'
        if elem_type in shape_map:
            parts.append(f'shape={shape_map[elem_type]}')
        parts.append(f'fillColor={elem.fill_color or "none"}')
        parts.append(f'strokeColor={elem.stroke_color or "#000000"}')
        parts.append(f'strokeWidth={max(1, int(elem.stroke_width or 1))}')
        return ';'.join(parts) + ';'

    @staticmethod
    def _element_label(elem: ElementInfo) -> str:
        meta = getattr(elem, 'vlm_item', {}) or {}
        return str(meta.get('label') or meta.get('title') or meta.get('content') or '')

    @staticmethod
    def _add_text_cell(root: ET.Element, block: Dict[str, Any], cell_id: int) -> int:
        geo = block.get('geometry') or {}
        text = str(block.get('text') or block.get('content') or '')
        if not text.strip():
            return cell_id
        font_style = 0
        if block.get('font_weight') == 'bold':
            font_style += 1
        if block.get('font_style') == 'italic':
            font_style += 2
        style = [
            'text', 'html=1', 'whiteSpace=wrap', 'resizable=0',
            f'fontSize={int(float(block.get("font_size") or 12))}', 'align=center',
            'verticalAlign=middle', 'overflow=hidden', 'spacing=0', 'labelPadding=0',
            f'fontColor={block.get("font_color") or "#000000"}',
        ]
        if font_style:
            style.append(f'fontStyle={font_style}')
        if block.get('font_family'):
            style.append(f'fontFamily={str(block["font_family"]).split(",")[0].strip()}')
        cell = ET.SubElement(root, 'mxCell', {'id': str(cell_id), 'value': html.escape(text), 'style': ';'.join(style) + ';', 'vertex': '1', 'parent': '1'})
        ET.SubElement(cell, 'mxGeometry', {'x': str(round(float(geo.get('x', 0)), 2)), 'y': str(round(float(geo.get('y', 0)), 2)), 'width': str(round(float(geo.get('width', 100)), 2)), 'height': str(round(float(geo.get('height', 20)), 2)), 'as': 'geometry'})
        return cell_id + 1


def main():
    parser = argparse.ArgumentParser(
        description='Edit Banana — image to editable draw.io',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='\nExamples:\n  python main.py -i input/test.png\n  python main.py\n',
    )
    parser.add_argument('-i', '--input', type=str, help='Input image path (omit to process all images in input/)')
    parser.add_argument('-o', '--output', type=str, help='Output directory (default: ./output)')
    parser.add_argument('--no-text', action='store_true', help='Skip OCR text extraction; VLM structure can still generate draw.io shapes if enabled')
    args = parser.parse_args()

    config = load_config()
    pipeline = Pipeline(config)
    output_dir = args.output or config.get('paths', {}).get('output_dir', './output')
    os.makedirs(output_dir, exist_ok=True)

    image_paths = []
    supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    if args.input:
        if not os.path.exists(args.input):
            print(f'Error: file not found {args.input}')
            sys.exit(1)
        image_paths.append(args.input)
    else:
        input_dir = config.get('paths', {}).get('input_dir', './input')
        if not os.path.exists(input_dir):
            print(f'Error: input directory does not exist: {input_dir}')
            print('   Create it and add images, or use -i to specify an image path')
            sys.exit(1)
        for file in os.listdir(input_dir):
            ext = Path(file).suffix.lower()
            if ext in supported_formats:
                image_paths.append(os.path.join(input_dir, file))
        if not image_paths:
            print(f'Error: no supported image files in {input_dir}')
            print(f"   Supported formats: {', '.join(sorted(supported_formats))}")
            sys.exit(1)

    print(f'\nProcessing {len(image_paths)} image(s)...')
    success_count = 0
    for img_path in image_paths:
        result = pipeline.process_image(img_path, output_dir=output_dir, with_text=not args.no_text)
        if result:
            success_count += 1

    print(f"\n{'=' * 60}")
    print(f'Done: {success_count}/{len(image_paths)} succeeded')
    print(f'Output: {output_dir}')
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
