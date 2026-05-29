#!/usr/bin/env bash
# batch_label.sh — 批量打标多个文件夹，每个文件夹独立运行
#
# 对指定目录下的每个子文件夹单独跑 VLM 打标，
# 结果按 folder_name-run_id 分别存放。
#
# 用法：
#   bash scripts/batch_label.sh
#   bash scripts/batch_label.sh [WORKERS]
#
# 环境变量：
#   WORKERS        并发线程数（默认 8）

set -euo pipefail

BASE_DIR="/mnt/DATA_71/public/data/MultiImage_Data/multi_image_data"
OUTPUT_DIR="/mnt/DATA_71/public/data/MultiImage_Data/multi_image_data_annotation"
WORKERS="${WORKERS:-8}"

FOLDERS=(
    "物件类-2.1普通场景下的物体_2w"
)

TEMPLATE_CONFIG="configs/object_label_only.yaml"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_DIR"

for folder in "${FOLDERS[@]}"; do
    input_dir="$BASE_DIR/$folder"

    if [ ! -d "$input_dir" ]; then
        echo "SKIP: $input_dir does not exist"
        continue
    fi

    echo "=== Labeling: $folder ==="

    # Generate per-folder config from template
    temp_config=$(mktemp /tmp/object_label_only_XXXXXX.yaml)
    sed \
        -e "s|^  input_dir:.*|  input_dir: $input_dir|" \
        -e "s|^  run_root:.*|  run_root: $OUTPUT_DIR|" \
        -e "s|^  pass_archive_dir:.*|  pass_archive_dir: $OUTPUT_DIR|" \
        "$TEMPLATE_CONFIG" > "$temp_config"

    python -m preprocessing.pipeline.cli run --stage label --config "$temp_config" --workers "$WORKERS"

    echo "=== Done: $folder ==="
    rm "$temp_config"
    echo ""
done

echo "All folders labeled."