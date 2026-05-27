#!/usr/bin bash
# run_quick.sh — 快速试跑（使用小数据集和简化配置）
#
# 使用场景：
#   1. 开发调试：验证新算子或修改后的参数是否正常工作
#   2. 快速验证：用少量图片确认 pipeline 能跑通，不耗时
#   3. 参数调优：反复调整阈值，需要秒级反馈
#
# 用法：
#   ./scripts/run_quick.sh [CONFIG] [INPUT_DIR] [RUN_ID]
#
# 默认值：
#   CONFIG       — configs/quick_test.yaml（仅启用 3 个轻量 filter）
#   INPUT_DIR    — tests/fixtures/images（项目内置的小图集）
#
# 示例：
#   ./scripts/run_quick.sh
#   ./scripts/run_quick.sh configs/quick_test.yaml /DATA/raw/sample_100

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

CONFIG="${1:-configs/quick_test.yaml}"
INPUT_DIR="${2:-tests/fixtures/images}"
RUN_ID="${3:-}"

# 如果配置文件不存在，自动生成一份 quick_test.yaml
if [ ! -f "$PROJECT_DIR/$CONFIG" ]; then
    echo "Generating quick_test.yaml for fast iteration..."
    mkdir -p "$PROJECT_DIR/configs"
    cat > "$PROJECT_DIR/configs/quick_test.yaml" << 'YAML'
input:
  input_dir: PLACEHOLDER
  recursive: true
  image_exts: [".jpg", ".jpeg", ".png", ".webp"]

output:
  run_root: outputs/runs
  pass_archive_dir: outputs/quick_pass
  pass_archive_layout: flat
  overwrite: true

filter:
  short_circuit: true
  operators:
    - name: aspect_ratio
      enabled: true
      params:
        ratio: 2.0
    - name: gray_style
      enabled: true
      params:
        threshold: 1.0
        max_side: 1600
    - name: framebox
      enabled: true
      params:
        max_side: 1024
        sides: 3
        min_thick: 20
        max_ratio: 0.30
        min_std: 15.0

label:
  enabled: false
YAML
fi

# 替换 input_dir 占位符为实际路径
cd "$PROJECT_DIR"
sed -i "s|PLACEHOLDER|$INPUT_DIR|g" "$CONFIG"

ARGS="--stage filter --config $CONFIG"
if [ -n "$RUN_ID" ]; then
    ARGS="$ARGS --run-id $RUN_ID"
fi

python -m preprocessing.pipeline.cli run $ARGS