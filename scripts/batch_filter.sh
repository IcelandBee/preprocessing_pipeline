#!/usr/bin/env bash
# batch_filter.sh — 批量筛选多个文件夹，每个文件夹独立运行
#
# 对 /mnt/DATA_71/public/data/MultiImage_Data/20260519/ 下的每个子文件夹
# 单独跑质量筛选，结果和通过图片按 folder_name-run_id 分别存放。

set -euo pipefail

BASE_DIR="/mnt/DATA_71/public/data/MultiImage_Data/20260519"
ANNOTATION_DIR="/mnt/DATA_71/public/data/MultiImage_Data/multi_image_data_annotation"
PASS_DIR="/mnt/DATA_71/public/data/MultiImage_Data/multi_image_data"
WORKERS="${WORKERS:-auto}"

FOLDERS=(
    "group_photo_data"
    "mobile_phone_photography_data-60000"
    "web_scraping_data"
)

TEMPLATE_CONFIG="configs/multi_image_filter.yaml"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_DIR"

for folder in "${FOLDERS[@]}"; do
    input_dir="$BASE_DIR/$folder"

    if [ ! -d "$input_dir" ]; then
        echo "SKIP: $input_dir does not exist"
        continue
    fi

    echo "=== Processing: $folder ==="

    # Generate per-folder config from template
    temp_config=$(mktemp /tmp/multi_image_filter_XXXXXX.yaml)
    sed \
        -e "s|^  input_dir:.*|  input_dir: $input_dir|" \
        -e "s|^  run_root:.*|  run_root: $ANNOTATION_DIR|" \
        -e "s|^  pass_archive_dir:.*|  pass_archive_dir: $PASS_DIR|" \
        "$TEMPLATE_CONFIG" > "$temp_config"

    python -m preprocessing.pipeline.cli run --stage filter --config "$temp_config" --workers "$WORKERS"

    echo "=== Done: $folder ==="
    rm "$temp_config"
    echo ""
done

echo "All folders processed."