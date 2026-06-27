#!/usr/bin/env bash
# 安装 SAM3 库并把 BPE 词表复制到 models/
# 用法：在项目根目录执行  bash scripts/setup_sam3.sh
# 若无法直连 GitHub，可用镜像：SAM3_CLONE_URL="https://gitclone.com/github.com/facebookresearch/sam3.git" bash scripts/setup_sam3.sh
# 模型权重需自行下载到 models/：推荐 ModelScope https://modelscope.cn/models/facebook/sam3 ，见 docs/SETUP_SAM3.md

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

SAM3_SRC="${SAM3_SRC:-$PROJECT_ROOT/sam3_src}"
BPE_DIR="${BPE_DIR:-${MODELS_DIR:-$PROJECT_ROOT/models/sam3}}"
# 直连 GitHub 失败时可指定镜像，例如：SAM3_CLONE_URL="https://gitclone.com/github.com/facebookresearch/sam3.git"
SAM3_CLONE_URL="${SAM3_CLONE_URL:-https://github.com/facebookresearch/sam3.git}"

echo "[1/3] 克隆 facebookresearch/sam3 (${SAM3_CLONE_URL}) ..."
if [[ -d "$SAM3_SRC/.git" ]]; then
  echo "      已存在 $SAM3_SRC，跳过克隆（若需更新请先删掉该目录再运行）"
else
  rm -rf "$SAM3_SRC"
  git clone --depth 1 "$SAM3_CLONE_URL" "$SAM3_SRC"
fi

echo "[2/3] 安装 SAM3 包 (pip install -e $SAM3_SRC) ..."
pip install -e "$SAM3_SRC"

echo "[3/3] 复制 BPE 词表到 $BPE_DIR/ ..."
mkdir -p "$BPE_DIR"
BPE_NAME="bpe_simple_vocab_16e6.txt.gz"
for BPE_SRC in "$SAM3_SRC/assets/$BPE_NAME" "$SAM3_SRC/sam3/assets/$BPE_NAME"; do
  if [[ -f "$BPE_SRC" ]]; then
    cp "$BPE_SRC" "$BPE_DIR/"
    echo "      已复制到 $BPE_DIR/$BPE_NAME"
    break
  fi
done
if [[ ! -f "$BPE_DIR/$BPE_NAME" ]]; then
  echo "      未找到 BPE 文件，在仓库中查找："
  find "$SAM3_SRC" -name "*.gz" 2>/dev/null || true
fi

echo ""
echo "完成。下一步：将 SAM3 权重下载到 models/（推荐 ModelScope），并配置 config.yaml，详见 docs/SETUP_SAM3.md"
