"""
PaddleOCR adapter (optional).

Same interface as LocalOCR: analyze_image(image_path) -> OCRResult.
Recommended for PaddleOCR 2.x: paddleocr>=2.8.0,<3.0.0 + paddlepaddle>=2.6.1,<3.0.0.
"""

from pathlib import Path
from typing import List, Tuple, Any

from PIL import Image

from .base import TextBlock, OCRResult

# Disable oneDNN to avoid ConvertPirAttribute2RuntimeAttribute error on some CPUs
import os
os.environ.setdefault("FLAGS_use_mkldnn", "0")

try:
    from paddleocr import PaddleOCR
except ImportError:
    PaddleOCR = None


class PaddleOCRAdapter:
    """
    OCR engine using PaddleOCR; often better for mixed Chinese/English than Tesseract.
    Requires: paddleocr, paddlepaddle (or paddlepaddle-gpu).
    """

    def __init__(
        self,
        use_angle_cls: bool = True,
        lang: str = "ch",
        model_dir: str = None,
        det_model_dir: str = None,
        rec_model_dir: str = None,
        cls_model_dir: str = None,
        allow_download: bool = True,
    ):
        if PaddleOCR is None:
            raise ImportError(
                "Install PaddleOCR: pip install paddleocr paddlepaddle (or paddlepaddle-gpu)"
            )
        det_model_dir = det_model_dir or self._find_local_model_dir(model_dir, "det")
        rec_model_dir = rec_model_dir or self._find_local_model_dir(model_dir, "rec")
        cls_model_dir = cls_model_dir or self._find_local_model_dir(model_dir, "cls")
        if not allow_download:
            self._require_local_model_dir(det_model_dir, "det")
            self._require_local_model_dir(rec_model_dir, "rec")
            if use_angle_cls:
                self._require_local_model_dir(cls_model_dir, "cls")
        kwargs = {
            "use_angle_cls": use_angle_cls,
            "lang": lang,
        }
        if det_model_dir:
            kwargs["det_model_dir"] = det_model_dir
        if rec_model_dir:
            kwargs["rec_model_dir"] = rec_model_dir
        if cls_model_dir:
            kwargs["cls_model_dir"] = cls_model_dir
        try:
            self._engine = PaddleOCR(**kwargs)
        except AttributeError as e:
            if "set_optimization_level" in str(e):
                raise RuntimeError(
                    "PaddleOCR incompatible with this PaddlePaddle (missing set_optimization_level).\n"
                    "Install compatible PaddleOCR 2.x and PaddlePaddle 2.x:\n"
                    "  pip uninstall paddleocr paddlepaddle paddlepaddle-gpu paddlex -y\n"
                    "  pip install \"paddleocr>=2.8.0,<3.0.0\" \"paddlepaddle>=2.6.1,<3.0.0\"   # CPU\n"
                    "  # GPU: pip install paddlepaddle-gpu paddleocr\n"
                    "See README Optional PaddleOCR section."
                ) from e
            raise

    @staticmethod
    def _find_local_model_dir(model_dir: str, kind: str) -> str:
        """Find a PaddleOCR 2.x inference model directory under a local model root."""
        if not model_dir:
            return None
        root = Path(model_dir)
        if not root.exists():
            return None

        direct = root / kind
        if direct.exists():
            return str(direct)

        matches = sorted(root.glob(f"*_{kind}_infer"))
        return str(matches[0]) if matches else None

    @staticmethod
    def _require_local_model_dir(model_dir: str, kind: str) -> None:
        """Fail fast instead of letting PaddleOCR download when local models are required."""
        if not model_dir:
            raise FileNotFoundError(
                f"PaddleOCR {kind} model directory is not configured; set ocr.paddleocr.{kind}_model_dir "
                f"or put a *_{kind}_infer directory under ocr.paddleocr.model_dir."
            )
        path = Path(model_dir)
        required = ["inference.pdmodel", "inference.pdiparams"]
        missing = [name for name in required if not (path / name).exists()]
        if missing:
            raise FileNotFoundError(
                f"PaddleOCR {kind} model directory is incomplete: {path}. Missing: {', '.join(missing)}"
            )

    def _parse_result(self, result: Any) -> List[TextBlock]:
        """Parse PaddleOCR 2.x or 3.x result into list of TextBlock."""
        text_blocks: List[TextBlock] = []

        if not result:
            return text_blocks

        # Normalize to list (single image may return one object or dict key 0)
        if not isinstance(result, list):
            if isinstance(result, dict):
                first_val = result.get(0) or (list(result.values())[0] if result else None)
                if first_val is None:
                    return text_blocks
                result = [first_val]
            else:
                result = [result]

        # PaddleOCR 3.x: list of PaddleX OCRResult (dict-like: rec_polys, rec_texts, rec_scores)
        if isinstance(result, list) and len(result) > 0:
            first = result[0]
            get = getattr(first, "get", None) if not isinstance(first, dict) else first.get
            if get is not None and callable(get):
                rec_polys = get("rec_polys") or get("dt_polys") or []
                rec_texts = get("rec_texts") or []
                rec_scores = get("rec_scores") or []
                if isinstance(rec_texts, (list, tuple)) and (
                    isinstance(rec_polys, (list, tuple))
                    or (hasattr(rec_polys, "__iter__") and not isinstance(rec_polys, (str, bytes)))
                ):
                    for i, poly in enumerate(rec_polys):
                        text = (rec_texts[i] if i < len(rec_texts) else "")
                        if isinstance(text, (list, tuple)):
                            text = (text[0] or "") if text else ""
                        text = (text or "").strip()
                        conf = (
                            float(rec_scores[i])
                            if i < len(rec_scores) and rec_scores
                            else 1.0
                        )
                        if not text:
                            continue
                        try:
                            polygon: List[Tuple[float, float]] = [
                                (float(p[0]), float(p[1]))
                                for p in (poly if hasattr(poly, "__iter__") else [])
                            ]
                        except (IndexError, TypeError, KeyError):
                            continue
                        if len(polygon) < 3:
                            continue
                        ys = [p[1] for p in polygon]
                        font_size_px = (
                            max(max(ys) - min(ys), 12.0) if len(ys) >= 2 else 12.0
                        )
                        text_blocks.append(
                            TextBlock(
                                text=text,
                                polygon=polygon,
                                confidence=conf,
                                font_size_px=font_size_px,
                                spans=[],
                            )
                        )
                    return text_blocks

        # PaddleOCR 2.x: [[line1,...]] or [line1,...], line = [box, (text, conf)]
        lines: List[Any] = []
        if isinstance(result, list):
            if len(result) == 1 and isinstance(result[0], list):
                lines = result[0] or []
            else:
                lines = result or []

        for line in lines:
            if not line or len(line) < 2:
                continue
            # Skip dict-like items (3.x format, already handled above)
            if hasattr(line, "get") and callable(getattr(line, "get", None)):
                continue
            box = line[0]
            text_part = line[1]
            if isinstance(text_part, (list, tuple)):
                text = (text_part[0] or "").strip()
                conf = float(text_part[1]) if len(text_part) > 1 else 1.0
            else:
                text = (text_part or "").strip()
                conf = 1.0
            if not text:
                continue
            try:
                polygon = [(float(p[0]), float(p[1])) for p in box]
            except (IndexError, TypeError, KeyError):
                continue
            ys = [p[1] for p in polygon]
            font_size_px = max(max(ys) - min(ys), 12.0) if len(ys) >= 2 else 12.0
            text_blocks.append(
                TextBlock(
                    text=text,
                    polygon=polygon,
                    confidence=conf,
                    font_size_px=font_size_px,
                    spans=[],
                )
            )
        return text_blocks

    def analyze_image(self, image_path: str) -> OCRResult:
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        img = Image.open(image_path)
        if img.mode != "RGB":
            img = img.convert("RGB")
        width, height = img.size

        # PaddleOCR 2.x: ocr(path, cls=True); 3.x: no cls arg
        try:
            result = self._engine.ocr(str(image_path), cls=True)
        except TypeError:
            result = self._engine.ocr(str(image_path))

        # PaddleOCR 3.x (PaddleX): list of dict-like OCRResult with rec_polys, rec_texts, rec_scores
        # PaddleOCR 2.x: [ [box, (text, conf)], ... ] or [[line1,...]]
        text_blocks = self._parse_result(result)

        return OCRResult(
            image_width=width,
            image_height=height,
            text_blocks=text_blocks,
            styles=[],
        )
