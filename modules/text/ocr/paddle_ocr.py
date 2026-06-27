"""
PaddleOCR adapter (optional).

Same interface as LocalOCR: analyze_image(image_path) -> OCRResult.
Recommended for PP-OCRv6: paddleocr>=3.7.0 + paddlepaddle>=3.0.0,<3.3.0.
"""

from pathlib import Path
from typing import List, Tuple, Any, Dict
from importlib import metadata

from PIL import Image

from .base import TextBlock, OCRResult

import os
import tempfile
# Disable oneDNN/MKLDNN before importing PaddleOCR/PaddleX. PaddlePaddle 3.3.x
# can crash on CPU with ConvertPirAttribute2RuntimeAttribute in oneDNN; keeping
# these flags off avoids the fast path that triggers the PIR conversion bug.
os.environ.setdefault("PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT", "0")
os.environ.setdefault("FLAGS_use_mkldnn", "0")
os.environ.setdefault("FLAGS_use_onednn", "0")

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
        use_angle_cls: bool = False,
        lang: str = "ch",
        model_dir: str = None,
        det_model_dir: str = None,
        rec_model_dir: str = None,
        cls_model_dir: str = None,
        text_detection_model_dir: str = None,
        text_recognition_model_dir: str = None,
        textline_orientation_model_dir: str = None,
        text_detection_model_name: str = "PP-OCRv6_medium_det",
        text_recognition_model_name: str = "PP-OCRv6_medium_rec",
        textline_orientation_model_name: str = None,
        ocr_version: str = "PP-OCRv6",
        text_det_limit_side_len: int = 64,
        text_det_limit_type: str = "min",
        text_det_thresh: float = 0.3,
        text_det_box_thresh: float = 0.6,
        text_det_unclip_ratio: float = 1.5,
        text_rec_score_thresh: float = 0.0,
        device: str = None,
        engine: str = None,
        scale: float = 1.0,
        min_confidence: float = 0.30,
        allow_download: bool = True,
        allow_legacy_fallback: bool = False,
    ):
        if PaddleOCR is None:
            raise ImportError(
                "Install PaddleOCR: pip install paddleocr paddlepaddle (or paddlepaddle-gpu)"
            )
        self.scale = max(float(scale or 1.0), 1.0)
        self.min_confidence = max(float(min_confidence or 0.0), 0.0)
        text_detection_model_dir = text_detection_model_dir or det_model_dir or self._find_local_model_dir(model_dir, "det")
        text_recognition_model_dir = text_recognition_model_dir or rec_model_dir or self._find_local_model_dir(model_dir, "rec")
        textline_orientation_model_dir = (
            textline_orientation_model_dir or cls_model_dir or self._find_local_model_dir(model_dir, "cls")
        )
        textline_orientation_model_name = (
            textline_orientation_model_name
            or self._infer_textline_orientation_model_name(textline_orientation_model_dir)
            or "PP-LCNet_x1_0_textline_ori"
        )
        self.model_debug_info = {
            "det_model_name": text_detection_model_name,
            "rec_model_name": text_recognition_model_name,
            "textline_orientation_model_name": textline_orientation_model_name,
            "textline_orientation_enabled": use_angle_cls,
            "det_model_dir": text_detection_model_dir,
            "rec_model_dir": text_recognition_model_dir,
            "textline_orientation_model_dir": textline_orientation_model_dir,
            "paddleocr_version": self._package_version("paddleocr"),
            "paddlepaddle_version": self._package_version("paddlepaddle"),
        }
        if not allow_download:
            self._require_local_model_dir(text_detection_model_dir, "det")
            self._require_local_model_dir(text_recognition_model_dir, "rec")
            if use_angle_cls:
                self._require_local_model_dir(textline_orientation_model_dir, "cls")

        kwargs: Dict[str, Any] = {
            "lang": lang,
            "ocr_version": ocr_version,
            "text_detection_model_name": text_detection_model_name,
            "text_recognition_model_name": text_recognition_model_name,
            "textline_orientation_model_name": textline_orientation_model_name,
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
            "use_textline_orientation": use_angle_cls,
            "text_det_limit_side_len": text_det_limit_side_len,
            "text_det_limit_type": text_det_limit_type,
            "text_det_thresh": text_det_thresh,
            "text_det_box_thresh": text_det_box_thresh,
            "text_det_unclip_ratio": text_det_unclip_ratio,
            "text_rec_score_thresh": text_rec_score_thresh,
        }
        if device:
            kwargs["device"] = device
        if engine:
            kwargs["engine"] = engine
        debug_kwargs = {k: v for k, v in kwargs.items() if k not in {"text_detection_model_dir", "text_recognition_model_dir", "textline_orientation_model_dir"}}
        print(f"[PaddleOCRAdapter] init kwargs: {debug_kwargs}; scale={self.scale}; min_confidence={self.min_confidence}", flush=True)
        print(f"[PaddleOCRAdapter] model selection: {self.model_debug_info}", flush=True)
        if text_detection_model_dir:
            kwargs["text_detection_model_dir"] = text_detection_model_dir
        if text_recognition_model_dir:
            kwargs["text_recognition_model_dir"] = text_recognition_model_dir
        if textline_orientation_model_dir:
            kwargs["textline_orientation_model_dir"] = textline_orientation_model_dir
        try:
            self._engine = PaddleOCR(**kwargs)
            self._engine_mode = "paddleocr_3"
        except TypeError as e:
            if not allow_legacy_fallback:
                raise RuntimeError(
                    "PaddleOCR did not accept PP-OCRv6/PaddleOCR 3.x arguments, so the configured v6 models "
                    "were NOT used. Install compatible versions or explicitly set allow_legacy_fallback: true "
                    "if you intentionally want PaddleOCR 2.x behavior.\n"
                    "Expected install:\n"
                    "  pip install \"paddleocr>=3.7.0,<4.0.0\" \"paddlepaddle>=3.0.0,<3.3.0\"\n"
                    f"Detected paddleocr={self.model_debug_info.get('paddleocr_version')}, "
                    f"paddlepaddle={self.model_debug_info.get('paddlepaddle_version')}\n"
                    f"Original TypeError: {e}"
                ) from e
            legacy_kwargs = {"use_angle_cls": use_angle_cls, "lang": lang}
            if text_detection_model_dir:
                legacy_kwargs["det_model_dir"] = text_detection_model_dir
            if text_recognition_model_dir:
                legacy_kwargs["rec_model_dir"] = text_recognition_model_dir
            if textline_orientation_model_dir:
                legacy_kwargs["cls_model_dir"] = textline_orientation_model_dir
            print(
                "[PaddleOCRAdapter] WARNING: falling back to PaddleOCR legacy API; "
                f"PP-OCRv6 model-name arguments were ignored. legacy_kwargs={legacy_kwargs}",
                flush=True,
            )
            self._engine = PaddleOCR(**legacy_kwargs)
            self._engine_mode = "paddleocr_legacy"
        except AttributeError as e:
            if "set_optimization_level" in str(e):
                raise RuntimeError(
                    "PaddleOCR/PaddlePaddle version mismatch. Install PP-OCRv6-compatible packages:\n"
                    "  pip uninstall paddleocr paddlepaddle paddlepaddle-gpu paddlex -y\n"
                    "  pip install \"paddleocr>=3.7.0,<4.0.0\" \"paddlepaddle>=3.0.0,<3.3.0\"   # CPU\n"
                    "  # GPU: install the matching paddlepaddle-gpu 3.x build, then paddleocr>=3.7.0\n"
                    "See README Optional PaddleOCR section."
                ) from e
            raise

    @staticmethod
    def _package_version(package_name: str) -> str:
        try:
            return metadata.version(package_name)
        except metadata.PackageNotFoundError:
            return "not installed"

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
    def _infer_textline_orientation_model_name(model_dir: str) -> str:
        """Infer PaddleOCR 3.x textline orientation model name from a local model directory."""
        if not model_dir:
            return None
        name = Path(model_dir).name
        known_names = {
            "pp-lcnet_x1_0_textline_ori": "PP-LCNet_x1_0_textline_ori",
            "pp_lcnet_x1_0_textline_ori": "PP-LCNet_x1_0_textline_ori",
            "pp-lcnet_x0_25_textline_ori": "PP-LCNet_x0_25_textline_ori",
            "pp_lcnet_x0_25_textline_ori": "PP-LCNet_x0_25_textline_ori",
        }
        return known_names.get(name.lower(), name)

    @staticmethod
    def _require_local_model_dir(model_dir: str, kind: str) -> None:
        """Fail fast instead of letting PaddleOCR download when local models are required."""
        if not model_dir:
            raise FileNotFoundError(
                f"PaddleOCR {kind} model directory is not configured; set ocr.paddleocr.{kind}_model_dir "
                f"or put a *_{kind}_infer directory under ocr.paddleocr.model_dir."
            )
        path = Path(model_dir)
        if not path.exists() or not path.is_dir():
            raise FileNotFoundError(f"PaddleOCR {kind} model directory does not exist: {path}")

        # PaddleOCR 2.x inference exports usually contain inference.pdmodel +
        # inference.pdiparams. PaddleOCR 3.x/PaddleX official models may use
        # inference.json + inference.pdiparams instead. Accept both so PP-OCRv6
        # local model directories are not incorrectly rejected as incomplete.
        model_files = {child.name for child in path.iterdir() if child.is_file()}
        valid_model_pairs = [
            ("inference.pdmodel", "inference.pdiparams"),
            ("inference.json", "inference.pdiparams"),
            ("model.pdmodel", "model.pdiparams"),
            ("model.json", "model.pdiparams"),
        ]
        if not any(all(name in model_files for name in pair) for pair in valid_model_pairs):
            expected = " or ".join(" + ".join(pair) for pair in valid_model_pairs)
            found = ", ".join(sorted(model_files)) or "no files"
            raise FileNotFoundError(
                f"PaddleOCR {kind} model directory is incomplete: {path}. "
                f"Expected one of: {expected}. Found: {found}"
            )

    def _parse_result(self, result: Any, scale: float = 1.0) -> List[TextBlock]:
        """Parse PaddleOCR 2.x or 3.x result into list of TextBlock."""
        text_blocks: List[TextBlock] = []

        if not result:
            return text_blocks

        # PaddleOCR web/API shape: {"ocrResults": [{"prunedResult": {...}}]}
        if isinstance(result, dict) and "ocrResults" in result:
            result = [
                item.get("prunedResult") or item.get("pruned_result") or item
                for item in (result.get("ocrResults") or [])
            ]

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
                nested = get("prunedResult") or get("pruned_result")
                if nested is not None:
                    first = nested
                    result[0] = nested
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
                        if not text or conf < self.min_confidence:
                            continue
                        try:
                            polygon: List[Tuple[float, float]] = [
                                (float(p[0]) / scale, float(p[1]) / scale)
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
            if not text or conf < self.min_confidence:
                continue
            try:
                polygon = [(float(p[0]) / scale, float(p[1]) / scale) for p in box]
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

    def _print_raw_result_summary(self, result: Any) -> None:
        """Print a concise summary of raw PaddleOCR output shape and recognized texts."""
        print(f"[PaddleOCRAdapter] raw result type: {type(result).__name__}")
        if isinstance(result, dict) and "ocrResults" in result:
            candidates = [
                item.get("prunedResult") or item.get("pruned_result") or item
                for item in (result.get("ocrResults") or [])
            ]
        else:
            candidates = result if isinstance(result, list) else [result]
        for idx, item in enumerate(candidates[:3]):
            get = getattr(item, "get", None) if not isinstance(item, dict) else item.get
            if get is None or not callable(get):
                print(f"[PaddleOCRAdapter] raw[{idx}] type={type(item).__name__}")
                continue
            nested = get("prunedResult") or get("pruned_result") or item
            nested_get = getattr(nested, "get", None) if not isinstance(nested, dict) else nested.get
            if nested_get is None or not callable(nested_get):
                print(f"[PaddleOCRAdapter] raw[{idx}] nested type={type(nested).__name__}")
                continue
            texts = nested_get("rec_texts") or []
            scores = nested_get("rec_scores") or []
            polys = nested_get("rec_polys") or nested_get("dt_polys") or []
            boxes = nested_get("rec_boxes") or []
            angles = nested_get("textline_orientation_angles") or []
            print(
                f"[PaddleOCRAdapter] raw[{idx}] det_model={self.model_debug_info.get('det_model_name')} "
                f"det_polys={len(polys)} det_boxes={len(boxes)} dt_polys={list(polys)[:20]} rec_boxes={list(boxes)[:20]}"
            )
            print(
                f"[PaddleOCRAdapter] raw[{idx}] rec_model={self.model_debug_info.get('rec_model_name')} "
                f"rec_texts={list(texts)[:20]} rec_scores={list(scores)[:20]}"
            )
            if self.model_debug_info.get("textline_orientation_enabled"):
                print(
                    f"[PaddleOCRAdapter] raw[{idx}] textline_orientation_model="
                    f"{self.model_debug_info.get('textline_orientation_model_dir') or 'default'} "
                    f"angles={list(angles)[:20]}"
                )

    def analyze_image(self, image_path: str) -> OCRResult:
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        img = Image.open(image_path)
        if img.mode != "RGB":
            img = img.convert("RGB")
        width, height = img.size
        print(
            f"[PaddleOCRAdapter] input image: path={image_path} size={width}x{height} "
            f"scale={self.scale} engine_mode={getattr(self, '_engine_mode', 'unknown')}",
            flush=True,
        )

        ocr_image_path = str(image_path)
        tmp_path = None
        if self.scale > 1.0:
            scaled = img.resize(
                (int(width * self.scale), int(height * self.scale)),
                Image.Resampling.LANCZOS,
            )
            tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            tmp_path = tmp.name
            tmp.close()
            scaled.save(tmp_path)
            ocr_image_path = tmp_path
            print(
                f"[PaddleOCRAdapter] scaled OCR image: path={ocr_image_path} "
                f"size={int(width * self.scale)}x{int(height * self.scale)}",
                flush=True,
            )

        try:
            # PaddleOCR 3.x / PP-OCRv6 uses predict(); keep ocr() fallback for older installs.
            if hasattr(self._engine, "predict"):
                result = self._engine.predict(ocr_image_path)
            else:
                try:
                    result = self._engine.ocr(ocr_image_path, cls=True)
                except TypeError:
                    result = self._engine.ocr(ocr_image_path)
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

        self._print_raw_result_summary(result)

        # PaddleOCR 3.x (PaddleX): list of dict-like OCRResult with rec_polys, rec_texts, rec_scores
        # PaddleOCR 2.x: [ [box, (text, conf)], ... ] or [[line1,...]]
        text_blocks = self._parse_result(result, scale=self.scale)
        print(f"[PaddleOCRAdapter] parsed text blocks: {len(text_blocks)}")

        return OCRResult(
            image_width=width,
            image_height=height,
            text_blocks=text_blocks,
            styles=[],
        )
