import os
import sys
import argparse
import warnings
import yaml
from pathlib import Path
from typing import Optional

os.environ.setdefault('PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK', 'True')
warnings.filterwarnings('ignore', message=".*doesn't match a supported version.*")

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from modules import TextRestorer

TEXT_MODULE_AVAILABLE = TextRestorer is not None


def load_config() -> dict:
    config_path = os.path.join(PROJECT_ROOT, 'config', 'config.yaml')
    if not os.path.exists(config_path):
        print(f'Warning: config file not found at {config_path}, using defaults')
        return {
            'paths': {'input_dir': './input', 'output_dir': './output'},
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
    """Image-to-draw.io pipeline with segmentation calls removed."""

    def __init__(self, config: dict = None):
        self.config = config or load_config()
        self._text_restorer = None

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

    def process_image(self, image_path: str, output_dir: str = None, with_text: bool = True) -> Optional[str]:
        print(f"\n{'=' * 60}")
        print(f'Processing: {image_path}')
        print(f"{'=' * 60}")

        if not with_text:
            print('Text extraction is required for draw.io output in the segmentation-free pipeline.')
            return None
        if self.text_restorer is None:
            print('TextRestorer unavailable; cannot generate draw.io output.')
            return None

        if output_dir is None:
            output_dir = self.config.get('paths', {}).get('output_dir', './output')
        img_stem = Path(image_path).stem
        img_output_dir = os.path.join(output_dir, img_stem)
        os.makedirs(img_output_dir, exist_ok=True)
        output_path = os.path.join(img_output_dir, f'{img_stem}.drawio')

        try:
            print('\n[1] OCR text/formula restoration...')
            drawio_xml = self.text_restorer.process(image_path)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(drawio_xml)
            print(f'   DrawIO: {output_path}')

            if hasattr(self.text_restorer, 'save_ocr_artifacts'):
                ocr_artifact_path = self.text_restorer.save_ocr_artifacts(img_output_dir, image_path)
                print(f'   OCR result: {ocr_artifact_path}')
            if hasattr(self.text_restorer, 'save_ocr_overlay'):
                ocr_overlay_path = self.text_restorer.save_ocr_overlay(img_output_dir, image_path)
                print(f'   OCR overlay: {ocr_overlay_path}')

            print(f"\n{'=' * 60}\nDone.\n{'=' * 60}")
            return output_path
        except Exception as e:
            print(f'\nFailed: {e}')
            import traceback
            traceback.print_exc()
            return None


def main():
    parser = argparse.ArgumentParser(
        description='Edit Banana — image to editable draw.io',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='\nExamples:\n  python main.py -i input/test.png\n  python main.py\n',
    )
    parser.add_argument('-i', '--input', type=str, help='Input image path (omit to process all images in input/)')
    parser.add_argument('-o', '--output', type=str, help='Output directory (default: ./output)')
    parser.add_argument('--no-text', action='store_true', help='Skip processing (kept for CLI compatibility; draw.io output requires text)')
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
