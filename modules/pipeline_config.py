"""Configuration loading for the Edit Banana pipeline."""

import os
import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_config() -> dict:
    """Load config/config.yaml."""
    config_path = os.path.join(PROJECT_ROOT, "config", "config.yaml")
    
    if not os.path.exists(config_path):
        print(f"Warning: config file not found at {config_path}, using defaults")
        return {
            'paths': {
                'input_dir': './input',
                'output_dir': './output',
            },
            'recognition': {
                'mode': 'sam3_first',
                'use_ocr_anchors': True,
            },
            'multimodal': {
                'mode': 'api',
                'api_key': '',
                'base_url': '',
                'model': '',
                'x_hw_id': '',
                'x_hw_appkey': '',
                'max_tokens': 4000,
                'timeout': 60,
                'request_text_log_chars': 1200,
                'vlm_only_region_max_items': 20,
                'vlm_only_ocr_anchor_max_blocks': 30,
                'enabled': False,
                'use_for': {
                    'prompt_planning': True,
                    'text_style': True,
                    'segmentation_refine': True,
                    'element_refine': False,
                    'region_refine': False,
                    'layout_refine': False,
                    'element_attributes': True,
                    'export_validate': True,
                },
                'thresholds': {
                    'element_refine_confidence': 0.75,
                    'region_refine_confidence': 0.70,
                    'layout_refine_confidence': 0.70,
                    'vlm_region_confidence': 0.60,
                    'vlm_connector_confidence': 0.70,
                    'text_style_confidence': 0.65,
                    'segmentation_refine_confidence': 0.65,
                    'element_attribute_confidence': 0.65,
                    'vlm_structure_confidence': 0.60,
                },
            },
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
                    'min_confidence': 0.30,
                },
            }
        }
    
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)
