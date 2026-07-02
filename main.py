#!/usr/bin/env python3
"""Edit Banana CLI entry."""

import argparse
import os
import sys
import warnings

os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
warnings.filterwarnings("ignore", message=".*doesn't match a supported version.*")

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from modules.pipeline import Pipeline
from modules.pipeline_config import load_config
from modules.sam3_config import ConfigLoader, PromptGroup


def main():
    parser = argparse.ArgumentParser(
        description="Edit Banana — image to editable PPTX",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py -i input/test.png
  python main.py
  python main.py -i test.png --refine
  python main.py -i test.png --groups image arrow
        """
    )
    
    parser.add_argument("-i", "--input", type=str, 
                        help="Input image path (omit to process all images in input/)")
    parser.add_argument("-o", "--output", type=str, 
                        help="Output directory (default: ./output)")
    parser.add_argument("--refine", action="store_true",
                        help="Enable quality evaluation and refinement")
    parser.add_argument("--no-text", action="store_true",
                        help="Skip text step (no OCR)")
    parser.add_argument("--groups", nargs='+', 
                        choices=['image', 'arrow', 'shape', 'background'],
                        help="Prompt groups to process (default: all)")
    parser.add_argument("--vlm-only", action="store_true",
                        help="Skip SAM3 and use VLM-only page structure recognition")
    parser.add_argument("--show-prompts", action="store_true",
                        help="Show prompt config")
    
    args = parser.parse_args()
    
    # Show prompt config without loading SAM3/CV dependencies.
    if args.show_prompts:
        prompt_groups = ConfigLoader.get_prompt_groups()
        for group_type, group_config in prompt_groups.items():
            print(f"\n[{group_config.name}] ({group_type.value})")
            print(f"  置信度阈值: {group_config.score_threshold}")
            print(f"  最小面积: {group_config.min_area}")
            print(f"  优先级: {group_config.priority}")
            print(f"  提示词 ({len(group_config.prompts)}个):")
            for prompt in group_config.prompts:
                print(f"    - {prompt}")
        return
    
    # Load config
    config = load_config()
    
    # Create pipeline
    if args.vlm_only:
        config.setdefault("recognition", {})["mode"] = "vlm_only"
    pipeline = Pipeline(config)
    
    # Parse group args
    groups = None
    if args.groups:
        group_map = {
            'image': PromptGroup.IMAGE,
            'arrow': PromptGroup.ARROW,
            'shape': PromptGroup.BASIC_SHAPE,
            'background': PromptGroup.BACKGROUND,
        }
        groups = [group_map[g] for g in args.groups]
    
    # Output dir
    output_dir = args.output or config.get('paths', {}).get('output_dir', './output')
    os.makedirs(output_dir, exist_ok=True)
    
    # Collect images
    image_paths = []
    supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    
    if args.input:
        # Single image
        if not os.path.exists(args.input):
            print(f"Error: file not found {args.input}")
            sys.exit(1)
        image_paths.append(args.input)
    else:
        # Batch from input/
        input_dir = config.get('paths', {}).get('input_dir', './input')
        
        if not os.path.exists(input_dir):
            print(f"Error: input directory does not exist: {input_dir}")
            print(f"   Create it and add images, or use -i to specify an image path")
            sys.exit(1)
        
        for file in os.listdir(input_dir):
            ext = Path(file).suffix.lower()
            if ext in supported_formats:
                image_paths.append(os.path.join(input_dir, file))
        
        if not image_paths:
            print(f"Error: no supported image files in {input_dir}")
            print(f"   Supported formats: {', '.join(supported_formats)}")
            sys.exit(1)
    
    # Process
    print(f"\nProcessing {len(image_paths)} image(s)...")
    
    success_count = 0
    for img_path in image_paths:
        result = pipeline.process_image(
            img_path,
            output_dir=output_dir,
            with_refinement=args.refine,
            with_text=not args.no_text,
            groups=groups
        )
        if result:
            success_count += 1
    
    # Summary
    print(f"\n{'='*60}")
    print(f"Done: {success_count}/{len(image_paths)} succeeded")
    print(f"Output: {output_dir}")
    print(f"{'='*60}")



if __name__ == "__main__":
    main()
