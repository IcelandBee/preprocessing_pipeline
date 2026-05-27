#!/usr/bin bash
# run_filter.sh — 仅执行筛选阶段
#
# 使用场景：
#   1. 数据量较大，想先完成筛选、确认筛选效果后再决定是否打标
#   2. 打标模型不可用（服务未部署/无 API Key），只跑筛选
#   3. 需要反复调试筛选参数，不想每次都等打标完成
#
# 用法：
#   ./scripts/run_filter.sh [CONFIG] [RUN_ID]
#
# 示例：
#   ./scripts/run_filter.sh configs/all_filters_pipeline.yaml
#   ./scripts/run_filter.sh configs/all_filters_pipeline.yaml 20260527-test01

set -euo pipefail

CONFIG="${1:-configs/all_filters_pipeline.yaml}"
RUN_ID="${2:-}"

ARGS="--stage filter --config $CONFIG"
if [ -n "$RUN_ID" ]; then
    ARGS="$ARGS --run-id $RUN_ID"
fi

python -m preprocessing.pipeline.cli run $ARGS