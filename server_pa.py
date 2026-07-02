#!/usr/bin/env python3
"""
FastAPI Backend Server — web service entry for Edit Banana.

Provides upload and conversion API. Run with: python server_pa.py
Server runs at http://localhost:8000
"""

import os
import sys
import uuid
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
import uvicorn

app = FastAPI(
    title="Edit Banana API",
    description="Image to editable PowerPoint — upload a diagram image, get a task ID and download the generated PPTX.",
    version="1.0.0",
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/")
def root():
    return {"service": "Edit Banana", "docs": "/docs"}


@app.post("/convert")
async def convert(file: UploadFile = File(...)):
    """Upload an image, start a conversion task, and return its task ID."""
    name = file.filename or ""
    ext = Path(name).suffix.lower()
    allowed = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}
    if ext not in allowed:
        raise HTTPException(400, f"Unsupported format. Use one of: {', '.join(sorted(allowed))}.")

    config_path = os.path.join(PROJECT_ROOT, "config", "config.yaml")
    if not os.path.exists(config_path):
        raise HTTPException(503, "Server not configured (missing config/config.yaml)")

    try:
        from main import load_config, Pipeline
        import tempfile
        import shutil

        config = load_config()
        output_dir = config.get("paths", {}).get("output_dir", "./output")
        os.makedirs(output_dir, exist_ok=True)
        task_id = uuid.uuid4().hex

        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name

        try:
            pipeline = Pipeline(config)
            result_path = pipeline.process_image(
                tmp_path,
                output_dir=output_dir,
                with_refinement=False,
                with_text=True,
                task_id=task_id,
            )
            if not result_path or not os.path.exists(result_path):
                raise HTTPException(500, "Conversion failed")
            return {"success": True, "task_id": task_id}
        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/download/{task_id}")
def download(task_id: str):
    """Download the PPTX produced for a conversion task."""
    try:
        uuid.UUID(task_id)
    except ValueError:
        raise HTTPException(400, "Invalid task_id")

    from main import load_config

    config = load_config()
    output_dir = Path(config.get("paths", {}).get("output_dir", "./output"))
    task_dir = (output_dir / task_id).resolve()
    output_root = output_dir.resolve()
    if output_root not in task_dir.parents and task_dir != output_root:
        raise HTTPException(400, "Invalid task_id")
    if not task_dir.is_dir():
        raise HTTPException(404, "Task not found")

    pptx_files = sorted(task_dir.glob("*.pptx"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not pptx_files:
        raise HTTPException(404, "PPTX not found for task")

    pptx_path = pptx_files[0]
    return FileResponse(
        path=str(pptx_path),
        media_type="application/vnd.openxmlformats-officedocument.presentationml.presentation",
        filename=pptx_path.name,
    )


def main():
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
