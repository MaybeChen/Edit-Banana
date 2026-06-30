"""
Icon/Picture processor for non-basic shapes (icons, pictures, logos, charts, etc.).

- Uses RMBG-2.0 for background removal on icon-like types
- Converts crops to base64 and generates XML fragments

Usage:
    from modules import IconPictureProcessor, ProcessingContext
    processor = IconPictureProcessor()
    context = ProcessingContext(image_path="test.png")
    context.elements = [...]  # from SAM3
    result = processor.process(context)
    # Elements get base64 and xml_fragment
"""

import os
import io
import base64
import xml.etree.ElementTree as ET
from typing import Optional, List
from PIL import Image
import numpy as np
import cv2
from prompts.image import IMAGE_PROMPT
# ONNX Runtime (optional, for RMBG)
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("[IconPictureProcessor] Warning: onnxruntime not available, RMBG disabled")

from .base import BaseProcessor, ProcessingContext, ModelWrapper
from .data_types import ElementInfo, ProcessingResult, LayerLevel


# ======================== RMBG-2.0 model wrapper ========================
class RMBGModel(ModelWrapper):
    """
    RMBG-2.0 background-removal model (ONNX Runtime, CUDA if available).

    Example:
        model = RMBGModel(model_path)
        model.load()
        rgba_image = model.remove_background(pil_image)
    """

    INPUT_SIZE = (1024, 1024)

    def __init__(self, model_path: str = None):
        super().__init__()
        self.model_path = model_path or self._get_default_path()
        self._session = None
        self._input_name = None
        self._output_name = None

    def _get_default_path(self) -> str:
        """Default model path under models/rmbg/."""
        return os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "models", "rmbg", "model.onnx"
        )
    
    def load(self):
        """Load RMBG-2.0 ONNX model; fallback to CPU if CUDA fails."""
        if self._is_loaded:
            return
        
        if not ONNX_AVAILABLE:
            print("[RMBGModel] Warning: onnxruntime not available, using fallback mode")
            self._is_loaded = True
            return
        
        if not os.path.exists(self.model_path):
            print(f"[RMBGModel] Warning: Model file not found at {self.model_path}, using fallback mode")
            self._is_loaded = True
            return
        
        # ONNX Runtime options
        session_options = ort.SessionOptions()
        session_options.log_severity_level = 3  # ERROR only
        session_options.enable_profiling = False
        
        # Available providers
        available_providers = ort.get_available_providers()
        
        # Try CUDA then CPU
        providers_to_try = [
            (['CUDAExecutionProvider', 'CPUExecutionProvider'], "CUDA+CPU"),
            (['CPUExecutionProvider'], "CPU only"),
        ]
        
        for providers, name in providers_to_try:
            # Filter valid providers
            valid_providers = [p for p in providers if p in available_providers]
            if not valid_providers:
                continue
            
            try:
                print(f"[RMBGModel] Trying to load with {name} ({valid_providers})...")
                self._session = ort.InferenceSession(
                    self.model_path,
                    providers=valid_providers,
                    sess_options=session_options
                )
                
                self._input_name = self._session.get_inputs()[0].name
                self._output_name = self._session.get_outputs()[0].name
                self._providers = valid_providers
                
                self._is_loaded = True
                print(f"[RMBGModel] Model loaded successfully with {name}")
                return
                
            except Exception as e:
                print(f"[RMBGModel] Failed to load with {name}: {e}")
                # Try next config
                continue
        
        # All attempts failed, use fallback
        print("[RMBGModel] Warning: All loading attempts failed, using fallback mode (no background removal)")
        self._is_loaded = True
    
    def _preprocess(self, img: np.ndarray) -> tuple:
        """Preprocess: scale, normalize, HWC->CHW. img: RGB numpy.
            
        Returns:
            (preprocessed_image, original_size)
        """
        # RMBG-2.0 expects BGR
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        h, w = img_bgr.shape[:2]
        
        # Scale to model input size
        img_resized = cv2.resize(img_bgr, self.INPUT_SIZE, interpolation=cv2.INTER_LINEAR)
        
        # Normalize to [0,1]
        img_normalized = img_resized.astype(np.float32) / 255.0
        
        # HWC -> CHW
        img_transposed = np.transpose(img_normalized, (2, 0, 1))
        
        # Add batch dim
        img_batch = np.expand_dims(img_transposed, axis=0)
        
        return img_batch, (h, w)
    
    def _postprocess(self, pred: np.ndarray, original_size: tuple) -> np.ndarray:
        """Extract alpha and resize to original. Returns alpha uint8."""
        # Remove batch, get alpha
        alpha = pred[0, 0, :, :]
        
        # Resize to original
        alpha_resized = cv2.resize(alpha, (original_size[1], original_size[0]), interpolation=cv2.INTER_LINEAR)
        
        # To uint8
        alpha_resized = (alpha_resized * 255).astype(np.uint8)
        
        return alpha_resized
    
    def predict(self, image: Image.Image) -> Image.Image:
        """Background removal; fallback to CPU if GPU fails. Returns RGBA PIL."""
        if not self._is_loaded:
            self.load()
        
        # Model not loaded: return fallback
        if self._session is None:
            return image.convert("RGBA")
        
        # To numpy
        img = np.array(image)
        
        # Preprocess
        img_input, original_size = self._preprocess(img)
        
        try:
            pred = self._session.run([self._output_name], {self._input_name: img_input})[0]
        except Exception as e:
            # GPU failed, try CPU
            if hasattr(self, '_providers') and 'CUDAExecutionProvider' in self._providers:
                print(f"[RMBGModel] GPU inference failed (OOM), switching to CPU...")
                
                try:
                    # Release session
                    self._session = None
                    
                    # New CPU session
                    session_options = ort.SessionOptions()
                    session_options.log_severity_level = 3
                    
                    self._session = ort.InferenceSession(
                        self.model_path,
                        providers=['CPUExecutionProvider'],
                        sess_options=session_options
                    )
                    self._providers = ['CPUExecutionProvider']
                    
                    # Retry
                    pred = self._session.run([self._output_name], {self._input_name: img_input})[0]
                    print("[RMBGModel] CPU inference successful")
                    
                except Exception as e2:
                    print(f"[RMBGModel] CPU inference also failed: {e2}")
                    print("[RMBGModel] Falling back to no background removal")
                    return image.convert("RGBA")
            else:
                print(f"[RMBGModel] Inference failed: {e}, using fallback (no background removal)")
                return image.convert("RGBA")
        
        # Postprocess alpha
        alpha = self._postprocess(pred, original_size)
        
        # Merge alpha -> RGBA
        img_rgba = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)
        img_rgba[:, :, 3] = alpha
        
        # To PIL
        return Image.fromarray(img_rgba)
    
    def remove_background(self, image: Image.Image) -> Image.Image:
        """Alias for predict."""
        return self.predict(image)
    
    def unload(self):
        """Release model resources."""
        self._session = None
        self._is_loaded = False

# ======================== Icon/Picture processor ========================
class IconPictureProcessor(BaseProcessor):
    """Process icon/picture elements: filter, crop, optional RMBG, base64, XML fragments."""

    # Types that use RMBG for background removal; others keep original crop
    RMBG_TYPES = {"icon", "logo", "symbol", "emoji", "button"}
    

    # Types that keep background (crop only)
    KEEP_BG_TYPES = {
        "picture", "photo", "chart", "function_graph", "screenshot", "image", "diagram",
        "graph", "line graph", "bar graph", "heatmap", "scatter plot", "histogram", "pie chart"
    }
    
    # Max element area ratio (skip if element area > this fraction of image)
    MAX_AREA_RATIO = 0.75

    
    def __init__(
        self,
        config=None,
        rmbg_model_path: str = None,
    ):
        super().__init__(config)
        self._rmbg_model: Optional[RMBGModel] = None
        self._rmbg_model_path = rmbg_model_path

    def load_rmbg_model(self):
        """Load RMBG model."""
        if self._rmbg_model is None:
            self._rmbg_model = RMBGModel(self._rmbg_model_path)
        if not self._rmbg_model.is_loaded:
            self._rmbg_model.load()

    def load_model(self):
        """Alias: load RMBG model."""
        self.load_rmbg_model()

    def process(self, context: ProcessingContext) -> ProcessingResult:
        """Process icon/picture elements in context."""
        self._log("Processing Icon/Picture elements")
        self.load_rmbg_model()

        # Load image
        if not context.image_path or not os.path.exists(context.image_path):
            return ProcessingResult(
                success=False,
                error_message="Invalid image path"
            )
        
        original_image = Image.open(context.image_path).convert("RGB")
        cv2_image = cv2.imread(context.image_path)
        
        # Filter elements to process
        elements_to_process = self._get_elements_to_process(context.elements)
        text_bboxes = self._extract_text_bboxes(context)

        self._log(f"Elements to process: {len(elements_to_process)}")
        
        processed_count = 0
        rmbg_count = 0
        keep_bg_count = 0
        
        for elem in elements_to_process:
            try:
                is_rmbg = self._process_element(elem, original_image, text_bboxes)
                processed_count += 1
                if is_rmbg:
                    rmbg_count += 1
                else:
                    keep_bg_count += 1
            except Exception as e:
                elem.processing_notes.append(f"Failed: {str(e)}")
                self._log(f"Element {elem.id} failed: {e}")
        
        self._log(f"Done: {processed_count}/{len(elements_to_process)} (RMBG:{rmbg_count}, keep_bg:{keep_bg_count})")
        
        return ProcessingResult(
            success=True,
            elements=context.elements,
            canvas_width=context.canvas_width,
            canvas_height=context.canvas_height,
            metadata={
                'processed_count': processed_count,
                'total_to_process': len(elements_to_process),
                'rmbg_count': rmbg_count,
                'keep_bg_count': keep_bg_count
            }
        )
    
    def _get_elements_to_process(self, elements: List[ElementInfo]) -> List[ElementInfo]:
        """Filter raster image/icon elements only; arrows and lines are generated as vectors later."""
        vector_line_types = {"arrow", "line", "connector"}
        all_types = set(IMAGE_PROMPT) - vector_line_types
        return [
            e for e in elements
            if e.element_type.lower() in all_types and e.base64 is None
        ]
    
    def _process_element(self, elem: ElementInfo, original_image: Image.Image, text_bboxes: List[List[int]] = None) -> bool:
        """Process one element. Returns True if RMBG was used."""
        elem_type = elem.element_type.lower()
        
        # Crop (shrink_margin: positive = shrink in)
        shrink_margin = 0
        img_w, img_h = original_image.size
        
        # Shrink bounds
        orig_w = elem.bbox.x2 - elem.bbox.x1
        orig_h = elem.bbox.y2 - elem.bbox.y1
        # Cap shrink to 10% of size
        max_shrink = min(orig_w * 0.1, orig_h * 0.1, shrink_margin)
        actual_shrink = int(max_shrink)
        
        x1 = max(0, elem.bbox.x1 + actual_shrink)
        y1 = max(0, elem.bbox.y1 + actual_shrink)
        x2 = min(img_w, elem.bbox.x2 - actual_shrink)
        y2 = min(img_h, elem.bbox.y2 - actual_shrink)
        
        # Ensure valid crop
        if x2 <= x1 or y2 <= y1:
            # Fallback to original bounds
            x1, y1 = elem.bbox.x1, elem.bbox.y1
            x2, y2 = elem.bbox.x2, elem.bbox.y2
        
        cropped = original_image.crop((x1, y1, x2, y2))

        is_rmbg = False
        
        # RMBG is only safe for small, standalone icons. Large card/container crops and
        # vector connectors must keep their original pixels so borders/inner strokes do not break.
        if self._should_use_rmbg(elem_type, elem, (img_w, img_h), text_bboxes or []):
            processed = self._rmbg_model.remove_background(cropped)
            elem.has_transparency = True
            is_rmbg = True
        else:
            processed = cropped.convert("RGBA")
            elem.has_transparency = False
            if self._is_card_like_crop(elem, (img_w, img_h)):
                processed = self._clear_raster_border(processed)
                elem.has_transparency = True
                elem.processing_notes.append("Cleared raster card border; vector overlay border is authoritative")
        
        processed = self._apply_text_cutouts(processed, (x1, y1, x2, y2), text_bboxes or [])

        # To base64
        elem.base64 = self._image_to_base64(processed)
        
        # Update bbox for padding
        elem.bbox.x1 = x1
        elem.bbox.y1 = y1
        elem.bbox.x2 = x2
        elem.bbox.y2 = y2
        
        # XML fragment
        self._generate_xml(elem)
        
        elem.processing_notes.append(f"IconPictureProcessor done (RMBG={is_rmbg})")
        
        return is_rmbg
    


    def _is_card_like_crop(self, elem: ElementInfo, image_size: tuple) -> bool:
        """Whether an image crop is likely a diagram card/container rather than a standalone icon."""
        img_w, img_h = image_size
        canvas_area = max(1, img_w * img_h)
        bbox = elem.bbox
        aspect = max(bbox.width, bbox.height) / max(1, min(bbox.width, bbox.height))
        area_ratio = bbox.area / canvas_area
        return (area_ratio >= 0.015 or (bbox.width >= 160 and bbox.height >= 110)) and aspect <= 2.5

    def _clear_raster_border(self, image: Image.Image) -> Image.Image:
        """Remove raster card outlines before vector border overlay.

        SAM3/card bboxes are often a few pixels outside the real rounded border. A
        fixed edge strip leaves inset dark arcs behind, creating doubled/fuzzy
        borders. Detect the first dark border bands near each side and clear up to
        that band so the vector overlay is the only visible card outline.
        """
        rgba = image.convert("RGBA")
        arr = np.array(rgba)
        h, w = arr.shape[:2]
        if w < 8 or h < 8:
            return rgba

        gray = (0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2])
        dark = (gray < 185) & (arr[:, :, 3] > 0)
        max_scan = max(4, min(24, int(min(w, h) * 0.12)))

        def border_extent(profile):
            hits = np.where(profile[:max_scan] > 0.025)[0]
            return int(hits[-1] + 2) if hits.size else max(4, min(10, int(min(w, h) * 0.04)))

        top = border_extent(dark.mean(axis=1))
        bottom = border_extent(dark[::-1, :].mean(axis=1))
        left = border_extent(dark.mean(axis=0))
        right = border_extent(dark[:, ::-1].mean(axis=0))

        top = min(top, h // 3)
        bottom = min(bottom, h // 3)
        left = min(left, w // 3)
        right = min(right, w // 3)
        arr[:top, :, 3] = 0
        arr[h - bottom:, :, 3] = 0
        arr[:, :left, 3] = 0
        arr[:, w - right:, 3] = 0
        self._log(f"Cleared raster card border bands: top={top}, right={right}, bottom={bottom}, left={left}px")
        return Image.fromarray(arr)

    def _should_use_rmbg(
        self,
        elem_type: str,
        elem: ElementInfo,
        image_size: tuple,
        text_bboxes: List[List[int]],
    ) -> bool:
        """Return whether RMBG should run for this element.

        RMBG is destructive for line-art containers: it may erase rounded borders,
        connector strokes, and text-adjacent pixels. Only small standalone icons use it.
        """
        if elem_type in {"arrow", "line", "connector"}:
            return False
        if elem_type not in self.RMBG_TYPES:
            return False

        img_w, img_h = image_size
        canvas_area = max(1, img_w * img_h)
        bbox = elem.bbox
        area_ratio = bbox.area / canvas_area
        large_card_like = area_ratio >= 0.015 or (bbox.width >= 160 and bbox.height >= 110)
        if large_card_like:
            elem.processing_notes.append(
                f"RMBG skipped: large/card-like crop (area_ratio={area_ratio:.3f})"
            )
            return False

        if self._overlaps_text_bbox(bbox.to_list(), text_bboxes):
            elem.processing_notes.append("RMBG skipped: crop overlaps editable OCR text")
            return False

        return True

    def _overlaps_text_bbox(self, bbox: List[int], text_bboxes: List[List[int]], min_overlap: float = 0.02) -> bool:
        """Whether bbox overlaps any OCR text box by a meaningful fraction of text area."""
        if not text_bboxes:
            return False
        bx1, by1, bx2, by2 = bbox
        for tx1, ty1, tx2, ty2 in text_bboxes:
            ix1 = max(bx1, tx1)
            iy1 = max(by1, ty1)
            ix2 = min(bx2, tx2)
            iy2 = min(by2, ty2)
            if ix2 <= ix1 or iy2 <= iy1:
                continue
            text_area = max(1, (tx2 - tx1) * (ty2 - ty1))
            if ((ix2 - ix1) * (iy2 - iy1)) / text_area >= min_overlap:
                return True
        return False

    def _extract_text_bboxes(self, context: ProcessingContext) -> List[List[int]]:
        """Extract OCR text boxes so raster icon crops can yield to editable text."""
        text_xml = getattr(context, "intermediate_results", {}).get("text_xml") if context else None
        if not text_xml:
            return []

        bboxes: List[List[int]] = []
        try:
            root = ET.fromstring(text_xml)
            for cell in root.iter("mxCell"):
                if not (cell.get("value") or "").strip():
                    continue
                geometry = cell.find("mxGeometry")
                if geometry is None:
                    continue
                x = float(geometry.get("x", 0))
                y = float(geometry.get("y", 0))
                w = float(geometry.get("width", 0))
                h = float(geometry.get("height", 0))
                if w <= 0 or h <= 0:
                    continue
                bboxes.append([int(x), int(y), int(x + w), int(y + h)])
        except Exception as exc:
            self._log(f"Failed to extract OCR text boxes for image cutouts: {exc}")
        return bboxes

    def _apply_text_cutouts(self, image: Image.Image, crop_box: tuple, text_bboxes: List[List[int]]) -> Image.Image:
        """Make OCR text regions transparent inside raster crops while preserving card borders."""
        if not text_bboxes:
            return image

        crop_x1, crop_y1, crop_x2, crop_y2 = crop_box
        rgba = image.convert("RGBA")
        arr = np.array(rgba)
        cut_count = 0
        skipped_border_count = 0

        # Keep a protected border band so OCR boxes that are too wide/tall do not erase
        # rounded card outlines. Use a small proportional margin for tiny icons.
        crop_w = max(1, crop_x2 - crop_x1)
        crop_h = max(1, crop_y2 - crop_y1)
        border_safe_margin = max(4, min(10, int(min(crop_w, crop_h) * 0.04)))
        inner_x1 = crop_x1 + border_safe_margin
        inner_y1 = crop_y1 + border_safe_margin
        inner_x2 = crop_x2 - border_safe_margin
        inner_y2 = crop_y2 - border_safe_margin

        for tx1, ty1, tx2, ty2 in text_bboxes:
            ix1 = max(inner_x1, tx1)
            iy1 = max(inner_y1, ty1)
            ix2 = min(inner_x2, tx2)
            iy2 = min(inner_y2, ty2)
            if ix2 <= ix1 or iy2 <= iy1:
                # It overlapped only the protected border band, so do not cut it.
                if not (tx2 <= crop_x1 or tx1 >= crop_x2 or ty2 <= crop_y1 or ty1 >= crop_y2):
                    skipped_border_count += 1
                continue

            # Pad OCR boxes inside the protected area only. Padding outside the inner area
            # is clamped so it cannot punch holes in card borders.
            local_x1 = max(border_safe_margin, ix1 - crop_x1 - 2)
            local_y1 = max(border_safe_margin, iy1 - crop_y1 - 2)
            local_x2 = min(arr.shape[1] - border_safe_margin, ix2 - crop_x1 + 2)
            local_y2 = min(arr.shape[0] - border_safe_margin, iy2 - crop_y1 + 2)
            if local_x2 <= local_x1 or local_y2 <= local_y1:
                skipped_border_count += 1
                continue
            arr[local_y1:local_y2, local_x1:local_x2, 3] = 0
            cut_count += 1

        if cut_count:
            msg = f"Applied OCR text cutouts to raster crop: {cut_count}"
            if skipped_border_count:
                msg += f" (kept {skipped_border_count} border-overlapping box(es))"
            self._log(msg)
            return Image.fromarray(arr)
        if skipped_border_count:
            self._log(f"Skipped {skipped_border_count} OCR cutout(s) that touched protected raster border")
        return rgba

    def _generate_xml(self, elem: ElementInfo):
        """
        Generate XML fragment for image element.
        """
        x1 = elem.bbox.x1
        y1 = elem.bbox.y1
        width = elem.bbox.x2 - elem.bbox.x1
        height = elem.bbox.y2 - elem.bbox.y1
        
        # DrawIO image style
        style = (
            "shape=image;verticalLabelPosition=bottom;verticalAlign=top;"
            "imageAspect=0;aspect=fixed;"
            f"image=data:image/png,{elem.base64};"
        )
        
        # DrawIO ids start at 2 (0,1 reserved)
        cell_id = elem.id + 2
        
        elem.xml_fragment = f'''<mxCell id="{cell_id}" parent="1" vertex="1" value="" style="{style}">
  <mxGeometry x="{x1}" y="{y1}" width="{width}" height="{height}" as="geometry"/>
</mxCell>'''
        
        # Layer
        elem.layer_level = LayerLevel.IMAGE.value
    
    def _image_to_base64(self, image: Image.Image) -> str:
        """Encode PIL image to base64."""
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        buffer.seek(0)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")


# ======================== Image complexity ========================
def calculate_image_complexity(image_arr: np.ndarray) -> tuple:
    """Compute image complexity (for picture vs icon). Returns (laplacian_variance, std_deviation)."""
    if image_arr.size == 0:
        return 0.0, 0.0
    
    gray = cv2.cvtColor(image_arr, cv2.COLOR_BGR2GRAY)
    
    # Laplacian variance (texture/edge)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    # Std dev (contrast/color variation)
    std_dev = np.std(gray)
    
    return laplacian_var, std_dev


def is_complex_image(image_arr: np.ndarray, laplacian_threshold: float = 800, std_threshold: float = 50) -> bool:
    """Whether image is complex enough to treat as picture."""
    l_var, s_dev = calculate_image_complexity(image_arr)
    return l_var > laplacian_threshold or s_dev > std_threshold


# ======================== Convenience ========================
def process_icons_pictures(elements: List[ElementInfo], 
                           image_path: str) -> List[ElementInfo]:
    """Process all icon/picture elements. Example: process_icons_pictures(elements, 'test.png')."""
    processor = IconPictureProcessor()
    context = ProcessingContext(
        image_path=image_path,
        elements=elements
    )
    
    result = processor.process(context)
    return result.elements
