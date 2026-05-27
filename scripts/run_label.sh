#!/usr/bin bash
# run_label.sh — 仅执行打标阶段（基于已有筛选结果）
#
# 使用场景：
#   1. 筛选已完成，现在 VLM 服务可用，补跑打标
#   2. 筛选和打标由不同团队负责，打标人员独立启动
#   3. 之前打标部分失败，用 --run-id 指定同一轮次重跑（配合 skip_existing）
#
# 用法：
#   ./scripts/run_label.sh [CONFIG] [RUN_ID]
#
# 重要：
#   RUN_ID 必须指定，否则会生成新的 run_id，
#   label 的 input_dir=auto 会找不到对应的筛选输出目录。
#
# 示例：
#   ./scripts/run_label.sh configs/all_filters_pipeline.yaml 20260527-203012

set -euo pipefail

CONFIG="${1:-configs/all_filters_pipeline.yaml}"
RUN_ID="${2:-}"

if [ -z "$RUN_ID" ]; then
    echo "ERROR: run_label.sh requires --run-id to locate filter output." >&2
    echo "Usage: $0 <CONFIG> <RUN_ID>" >&2
    echo "Example: $0 configs/all_filters_pipeline.yaml 20260527-203012" >&2
    exit 1
fi

python -m preprocessing.pipeline.cli run --stage label --config "$CONFIG" --run-id "$RUN_ID"