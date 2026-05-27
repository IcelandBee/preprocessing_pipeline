#!/usr/bin bash
# run_full.sh — 一次性执行筛选 + 打标全流程
#
# 使用场景：
#   数据已准备就绪，希望从原始图片直接跑完筛选和打标，
#   产出可直接用于训练的标注数据集。
#
# 用法：
#   ./scripts/run_full.sh [CONFIG] [RUN_ID]
#
# 示例：
#   ./scripts/run_full.sh configs/all_filters_pipeline.yaml
#   ./scripts/run_full.sh configs/all_filters_pipeline.yaml 20260527-203012

set -euo pipefail

CONFIG="${1:-configs/all_filters_pipeline.yaml}"
RUN_ID="${2:-}"

ARGS="--stage all --config $CONFIG"
if [ -n "$RUN_ID" ]; then
    ARGS="$ARGS --run-id $RUN_ID"
fi

python -m preprocessing.pipeline.cli run $ARGS