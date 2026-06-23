# SAM3 installation and model setup

This project uses **Meta SAM 3** (Segment Anything Model 3) for segmentation. Follow the steps below to install the library and place the model + BPE under `models/`.

---

## 1. Install SAM3 library

Official repo: [facebookresearch/sam3](https://github.com/facebookresearch/sam3). From the **project root**:

```bash
cd /path/to/Edit-Banana

# 1. Clone SAM3 (do not clone into a folder named sam3/ to avoid package name conflict)
git clone https://github.com/facebookresearch/sam3.git sam3_src
cd sam3_src

# 2. Install with current pip (activate venv first if you use one: source .venv/bin/activate)
pip install -e .

cd ..
```

**If you cannot reach GitHub** (e.g. `Failed to connect to github.com port 443`), use a mirror:

```bash
# Option A: use script with mirror
SAM3_CLONE_URL="https://gitclone.com/github.com/facebookresearch/sam3.git" bash scripts/setup_sam3.sh

# Option B: manual clone (pick a working mirror)
git clone --depth 1 https://gitclone.com/github.com/facebookresearch/sam3.git sam3_src
# or: git clone --depth 1 https://ghproxy.com/https://github.com/facebookresearch/sam3.git sam3_src
cd sam3_src && pip install -e . && cd ..
cp sam3_src/assets/bpe_simple_vocab_16e6.txt.gz models/
```

Verify:

```bash
python -c "from sam3.model_builder import build_sam3_image_model; from sam3.model.sam3_image_processor import Sam3Processor; print('OK')"
```

If that runs without error, the SAM3 library is installed.

---

## 2. Place BPE vocab under models/

Copy the BPE file from the cloned `sam3_src` into `models/`:

```bash
cd /path/to/Edit-Banana
mkdir -p models
cp sam3_src/assets/bpe_simple_vocab_16e6.txt.gz models/
```

If BPE is not under `assets/`, search for it:

```bash
find sam3_src -name "bpe*.gz"
```

Copy the found file to `models/bpe_simple_vocab_16e6.txt.gz` (or keep its path and set `bpe_path` in config accordingly).

---

## 3. Download model weights into models/

Use **ModelScope** (no access request) or **Hugging Face**.

### Option A: ModelScope (recommended, no access request)

Model page: <https://modelscope.cn/models/facebook/sam3>.

**Download to project `models/` via CLI:**

```bash
cd /path/to/Edit-Banana
pip install modelscope
mkdir -p models
python -c "
from modelscope import snapshot_download
path = snapshot_download('facebook/sam3', local_dir='models/sam3_ms', local_files_only=False)
print('Downloaded to', path)
"
```

Weights will be under `models/sam3_ms/`. **This project loads the checkpoint with `torch.load`; use a `sam3.pt` file** (if the repo only provides `model.safetensors`, convert it or obtain a `.pt` from another source). Set `checkpoint_path: "models/sam3_ms/sam3.pt"` in `config.yaml`. If the ModelScope repo includes a BPE file (e.g. `bpe_simple_vocab_16e6.txt.gz`), it will also be under `models/sam3_ms/`; point `bpe_path` to it or copy to `models/bpe_simple_vocab_16e6.txt.gz`.

**Or download manually in the browser:**

1. Open https://modelscope.cn/models/facebook/sam3  
2. Download the weight file (and BPE if available)  
3. Put them under `models/` and set the paths in `config.yaml`

### Option B: Hugging Face (request access first)

1. Open https://huggingface.co/facebook/sam3, log in, click **Request access**, and wait for approval.  
2. Install and log in locally: `pip install huggingface_hub`, `huggingface-cli login`  
3. Download to `models/`:

```bash
cd /path/to/Edit-Banana
mkdir -p models
python -c "
from huggingface_hub import snapshot_download
path = snapshot_download(repo_id='facebook/sam3', local_dir='models/sam3_hf', local_dir_use_symlinks=False)
print('Downloaded to', path)
"
```

Weights will be under `models/sam3_hf/`. Set `checkpoint_path` in `config.yaml` to the actual weight file path. **Use a `.pt` checkpoint**; if the repo only has `.safetensors`, convert or obtain `.pt` elsewhere.

---

## 4. Configure config.yaml

Copy the example config and edit:

```bash
cp config/config.yaml.example config/config.yaml
```

In `config/config.yaml` set **absolute or project-root-relative paths**. **Checkpoint must be `.pt`** (this project does not load safetensors). Example:

```yaml
sam3:
  # Weights: must be a .pt file (e.g. sam3.pt)
  checkpoint_path: "models/sam3_ms/sam3.pt"
  # BPE: path from the copy step above
  bpe_path: "models/bpe_simple_vocab_16e6.txt.gz"
  score_threshold: 0.5
  epsilon_factor: 0.02
  min_area: 100
```

Save and run the main pipeline (CLI or server).

---

## 5. Common issues

- **`import sam3` fails**  
  Run in an environment where you did `pip install -e sam3_src` and `sam3_src` is a full clone of the official repo.

- **`build_sam3_image_model` cannot find checkpoint / BPE**  
  Ensure `checkpoint_path` and `bpe_path` in `config.yaml` match real files (absolute or relative to project root).

- **`AssertionError: Torch not compiled with CUDA enabled`**  
  You installed a CPU-only PyTorch build but CUDA was requested by config or SAM3 internals. Set `sam3.device: "cpu"` in `config/config.yaml`, or install a CUDA-enabled PyTorch build that matches your NVIDIA driver/CUDA stack.

- **`ModuleNotFoundError: No module named 'triton'`**  
  SAM3 may import Triton through its PyTorch attention stack. On Linux, install the official package with `pip install triton` in the same environment. On native Windows, official Triton support is limited; try `pip install triton-windows` only if your PyTorch/CUDA/Python versions are supported, otherwise run SAM3 from WSL/Linux.

- **Hugging Face download is slow or fails**  
  Use **ModelScope** instead: <https://modelscope.cn/models/facebook/sam3> (see Option A above).

- **Official requirements: Python 3.12+, PyTorch 2.7+, CUDA 12.6+**  
  You can try your current environment first; if you hit errors, upgrade or use the official conda setup.
