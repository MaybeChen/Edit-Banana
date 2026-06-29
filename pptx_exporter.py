"""Export merged draw.io XML fragments to a single-slide PowerPoint deck.

The exporter intentionally covers the subset this project emits: images, basic
shapes, text boxes, and straight edges. It keeps the output editable where the
source draw.io element is editable, and embeds raster crops as pictures.
"""

import base64
import html
import io
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Optional, Tuple


EMU_PER_PX = 9525  # PowerPoint uses EMUs; 96dpi pixel -> 914400 / 96.


def export_drawio_to_pptx(drawio_xml_path: str, pptx_path: Optional[str] = None) -> str:
    """Create a PPTX file from a merged draw.io XML file and return its path."""
    # Imported lazily so the rest of the pipeline can still be used by tooling that
    # only performs syntax checks. `python-pptx` is listed in requirements.txt.
    from pptx import Presentation
    from pptx.dml.color import RGBColor
    from pptx.enum.shapes import MSO_AUTO_SHAPE_TYPE, MSO_CONNECTOR
    from pptx.util import Emu, Pt

    drawio_path = Path(drawio_xml_path)
    pptx_path = Path(pptx_path) if pptx_path else drawio_path.with_suffix(".pptx")

    tree = ET.parse(drawio_path)
    root = tree.getroot()
    graph = root.find(".//mxGraphModel")
    if graph is None:
        raise ValueError(f"No mxGraphModel found in {drawio_xml_path}")

    page_width = float(graph.get("pageWidth", 1280))
    page_height = float(graph.get("pageHeight", 720))

    prs = Presentation()
    prs.slide_width = Emu(page_width * EMU_PER_PX)
    prs.slide_height = Emu(page_height * EMU_PER_PX)
    slide = prs.slides.add_slide(prs.slide_layouts[6])

    for cell in root.iter("mxCell"):
        if cell.get("id") in {"0", "1"}:
            continue
        geometry = cell.find("mxGeometry")
        if geometry is None:
            continue
        style = _parse_style(cell.get("style", ""))
        if cell.get("edge") == "1":
            _add_edge(slide, geometry, style, MSO_CONNECTOR, RGBColor, Pt)
        elif "image" in style:
            _add_image(slide, geometry, style, Emu)
        elif (cell.get("value") or "").strip() or style.get("text") is True:
            _add_text(slide, cell, geometry, style, Emu, Pt, RGBColor)
        elif cell.get("vertex") == "1":
            _add_shape(slide, geometry, style, MSO_AUTO_SHAPE_TYPE, Emu, Pt, RGBColor)

    pptx_path.parent.mkdir(parents=True, exist_ok=True)
    prs.save(pptx_path)
    return str(pptx_path)


def _parse_style(style: str) -> Dict[str, object]:
    parsed: Dict[str, object] = {}
    for part in style.split(";"):
        if not part:
            continue
        if "=" in part:
            key, value = part.split("=", 1)
            parsed[key] = value
        else:
            parsed[part] = True
    return parsed


def _geometry_rect(geometry: ET.Element) -> Tuple[float, float, float, float]:
    return (
        float(geometry.get("x", 0)),
        float(geometry.get("y", 0)),
        float(geometry.get("width", 0)),
        float(geometry.get("height", 0)),
    )


def _emu(value: float, Emu):
    return Emu(value * EMU_PER_PX)


def _rgb(color: Optional[str], RGBColor):
    if not color or color == "none" or not color.startswith("#") or len(color) != 7:
        return None
    return RGBColor(int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16))


def _add_image(slide, geometry: ET.Element, style: Dict[str, object], Emu) -> None:
    image_ref = str(style.get("image", ""))
    match = re.match(r"data:image/[^,]+,(.+)", image_ref)
    if not match:
        return
    x, y, w, h = _geometry_rect(geometry)
    image_bytes = base64.b64decode(match.group(1))
    slide.shapes.add_picture(io.BytesIO(image_bytes), _emu(x, Emu), _emu(y, Emu), _emu(w, Emu), _emu(h, Emu))


def _add_shape(slide, geometry: ET.Element, style: Dict[str, object], MSO_AUTO_SHAPE_TYPE, Emu, Pt, RGBColor) -> None:
    x, y, w, h = _geometry_rect(geometry)
    shape_type = MSO_AUTO_SHAPE_TYPE.RECTANGLE
    if style.get("rounded") == "1":
        shape_type = MSO_AUTO_SHAPE_TYPE.ROUNDED_RECTANGLE
    if "ellipse" in style or style.get("shape") in {"ellipse", "cloud"}:
        shape_type = MSO_AUTO_SHAPE_TYPE.OVAL

    shape = slide.shapes.add_shape(shape_type, _emu(x, Emu), _emu(y, Emu), _emu(w, Emu), _emu(h, Emu))
    fill_color = _rgb(str(style.get("fillColor", "#ffffff")), RGBColor)
    if fill_color is None:
        shape.fill.background()
    else:
        shape.fill.solid()
        shape.fill.fore_color.rgb = fill_color

    stroke_color = _rgb(str(style.get("strokeColor", "#000000")), RGBColor)
    if stroke_color is None:
        shape.line.fill.background()
    else:
        shape.line.color.rgb = stroke_color
        shape.line.width = Pt(float(style.get("strokeWidth", 1)))


def _add_text(slide, cell: ET.Element, geometry: ET.Element, style: Dict[str, object], Emu, Pt, RGBColor) -> None:
    x, y, w, h = _geometry_rect(geometry)
    textbox = slide.shapes.add_textbox(_emu(x, Emu), _emu(y, Emu), _emu(w, Emu), _emu(h, Emu))
    frame = textbox.text_frame
    frame.clear()
    paragraph = frame.paragraphs[0]
    run = paragraph.add_run()
    run.text = _clean_text(cell.get("value", ""))
    run.font.size = Pt(float(style.get("fontSize", 12)))
    font_color = _rgb(str(style.get("fontColor", "#000000")), RGBColor)
    if font_color is not None:
        run.font.color.rgb = font_color
    if style.get("fontFamily"):
        run.font.name = str(style["fontFamily"])


def _add_edge(slide, geometry: ET.Element, style: Dict[str, object], MSO_CONNECTOR, RGBColor, Pt) -> None:
    points = {p.get("as"): p for p in geometry.iter("mxPoint")}
    source = points.get("sourcePoint")
    target = points.get("targetPoint")
    if source is None or target is None:
        return
    x1, y1 = float(source.get("x", 0)), float(source.get("y", 0))
    x2, y2 = float(target.get("x", 0)), float(target.get("y", 0))
    connector = slide.shapes.add_connector(
        MSO_CONNECTOR.STRAIGHT,
        int(x1 * EMU_PER_PX),
        int(y1 * EMU_PER_PX),
        int(x2 * EMU_PER_PX),
        int(y2 * EMU_PER_PX),
    )
    stroke_color = _rgb(str(style.get("strokeColor", "#000000")), RGBColor)
    if stroke_color is not None:
        connector.line.color.rgb = stroke_color
    connector.line.width = Pt(float(style.get("strokeWidth", 1)))


def _clean_text(value: str) -> str:
    text = html.unescape(value or "")
    text = re.sub(r"<[^>]+>", "", text)
    return text
