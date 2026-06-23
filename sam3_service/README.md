# SAM3 HTTP service

A lightweight HTTP wrapper that keeps a SAM3 model resident in memory and serves requests via FastAPI. The model is loaded once per server process, and requests are processed serially within a process to avoid VRAM spikes. Run multiple processes for concurrency and load balancing.

**Dependencies:** Install the main project dependencies first (PyTorch, SAM3 library, and `pip install -r requirements.txt` from the repo root). This package only adds FastAPI/uvicorn; the SAM3 and torch imports come from the main environment.

## Start a server

**Option A — shell scripts (run in separate terminals or background):**
```bash
bash sam3_service/run_servers.sh        # 1 SAM3 worker on port 8001
bash sam3_service/run_servers.sh 3      # 3 SAM3 workers on 8001, 8002, 8003
bash sam3_service/run_rmbg_servers.sh   # 1 RMBG worker on port 9101
```

**Option B — Python launcher (multiple workers in one command):**
```bash
python -m sam3_service.run_all_service --workers 2
python -m sam3_service.run_all_service --workers 2 --rmbg 1
```

**Health check (SAM3 default port 8001):**
```bash
curl -s http://127.0.0.1:8001/health
```

Notes:
- `--cache-size` controls the LRU cache size for encoded images.
- `--device` defaults to `auto` and falls back to CPU when CUDA is unavailable.
- Workers are one process per port to keep a single model copy per process.
- Set `--device cpu` to run without GPU (slow, but useful for debugging).

## Run multiple servers (manual)

Start 2-3 processes on different ports and, optionally, different GPUs:

```bash
CUDA_VISIBLE_DEVICES=0 python -m sam3_service.server --port 8001 --device cuda
CUDA_VISIBLE_DEVICES=1 python -m sam3_service.server --port 8002 --device cuda
CUDA_VISIBLE_DEVICES=2 python -m sam3_service.server --port 8003 --device cuda
```

## Call the API

```python
from sam3_service.client import Sam3ServiceClient

client = Sam3ServiceClient("http://127.0.0.1:8001")
resp = client.predict(
    image_path="/abs/path/to/image.png",
    prompts=["icon", "rectangle"],
    return_masks=True,
    mask_format="rle",
)
print(resp)
```

## Load balancing across servers

```python
from sam3_service.client import Sam3ServicePool

pool = Sam3ServicePool([
    "http://127.0.0.1:8001",
    "http://127.0.0.1:8002",
    "http://127.0.0.1:8003",
])

result = pool.predict(
    image_path="/abs/path/image.png",
    prompts=["rectangle", "arrow"],
    return_masks=True,
    mask_format="png",
)
```
`Sam3ServicePool.predict` is blocking and uses round-robin dispatch to spread requests across the live endpoints.

## API schema

- `GET /health` → `{ "status": "ok" }`
- `POST /predict`
  - Request body:
    - `image_path` (str): Path readable by the server.
    - `prompts` (list[str]): Text prompts.
    - `return_masks` (bool, default false): Whether to include masks.
    - `mask_format` ("rle" | "png", default "rle").
    - `score_threshold` (float, optional): Overrides config.
    - `epsilon_factor` (float, optional): Overrides config.
    - `min_area` (int, optional): Overrides config.
  - Response body:
    - `image_size`: `{ "width": int, "height": int }`
    - `results`: List of detections. Each item contains `prompt`, `score`, `bbox`, `polygon`, `area`, and optional `mask` with `data`, `format`, `shape` when `return_masks` is true.
