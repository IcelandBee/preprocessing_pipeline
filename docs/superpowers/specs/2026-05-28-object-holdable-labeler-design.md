# Object Holdable Labeler 设计文档

## 目标

将 `D:/Project/preprocessing/components/object_holdable.py` 中的物体可持性标注脚本，转换为预处理管线中的 `LabelOperator` 算子，统一使用 OpenAI-compatible API 调用远程 VLM，为管线配置所用。

## 方案选择

**方案 A：完全仿照 PersonAttributeLabeler 模式**

创建独立的 `ObjectHoldableLabeler(LabelOperator)` 类，与 `PersonAttributeLabeler` 采用相同架构模式。所有 normalization 函数内联在同一文件中。

理由：当前只有2个 labeler，完全仿照是最直接、最符合 YAGNI 的做法。未来 labeler 数量增长时再提取公共部分。

## 类结构与构造参数

- **类**: `ObjectHoldableLabeler(LabelOperator)`
- **name**: `"object_holdable_labeler"`
- **构造参数**（来自 YAML `label.operator.params`）:
  - `base_url: str` — API endpoint
  - `model_name: str` — 模型名
  - `api_key_env: str = "API_KEY"` — 环境变量名
  - `max_retries: int = 3`
  - `max_tokens: int = 768`（物体标注输出比 person 更长）
  - `temperature: float = 0.0`
  - `max_pixels: int = 1048576`（物体图片通常更大）
  - `**unused: object` — 吸收多余配置项

- **setup()**: 从环境变量取 api_key，创建 `OpenAI(base_url=..., api_key=...)` client
- **process()**: `load_image_rgb()` → `pil_to_data_url()` → `_infer()` → `normalize_annotation()` → `OperatorResult.label(labels)`
- **_infer()**: retry loop (1..max_retries)，调用 `self.client.chat.completions.create()`，解析 JSON

## Prompt 与消息结构

使用 `object_holdable.py` 中最终版 `OBJECT_PROMPT`（包含选择策略、粒度规则、命名消歧策略）。

消息结构只用 `user` 角色：
1. `{"type": "text", "text": "Object image:"}`
2. `{"type": "image_url", "image_url": {"url": data_url}}`
3. `{"type": "text", "text": OBJECT_PROMPT}`

与 PersonAttributeLabeler 的 system+user 模式不同，保持与原脚本的消息顺序一致。

## Normalization 与标注输出 Schema

10个字段完整迁移：

| 字段 | normalization |
|------|---------------|
| `object_name` | `normalize_object_name()` — strip、lower、清除 null/none/n/a |
| `object_category` | `normalize_text()` |
| `object_description` | `normalize_text()` |
| `coarse_position` | `normalize_coarse_position()` — 验证 horizontal/vertical 值域 |
| `bbox_norm` | `normalize_bbox()` — clamp [0,1]、自动÷100、翻转 min>max |
| `holdability` | `normalize_text()` + lower |
| `suitable_for_holding` | bool |
| `should_use` | bool；`object_name is None` 时强制 False |
| `confidence` | `normalize_text()` + lower，默认 "low" |
| `reason` | `normalize_text()` 或空字符串 |

关键约束：`object_name is None` 时，`should_use` 和 `suitable_for_holding` 强制设为 False。

JSON 解析：`strip_think()` 去除 `<think>` 标签后再解析，失败时抛异常让 retry loop 重试。

## YAML 配置与注册

### Registry 注册

`build_default_registry()` 中增加:
```python
from preprocessing.operators.labelers.object_holdable import ObjectHoldableLabeler
registry.register("object_holdable_labeler", ObjectHoldableLabeler)
```

### API 统一配置

```yaml
base_url: http://10.154.39.57:8001/v1
api_key_env: API_KEY
model_name: gemma-4-31B-it
```

### 配置 1: `object_pipeline.yaml`（filter + label 全流程）

包含 aspect_ratio、gray_style、compression_quality、duplicate 四个 filter，然后执行 object_holdable_labeler。

### 配置 2: `object_label_only.yaml`（只打标）

filter operators 为空列表，`label.input_dir` 直接指向已清洗数据目录。

## 涉及的文件变更

1. **新建**: `preprocessing/operators/labelers/object_holdable.py` — ObjectHoldableLabeler 实现
2. **修改**: `preprocessing/pipeline/registry.py` — 注册新算子
3. **新建**: `configs/object_pipeline.yaml` — 全流程配置
4. **新建**: `configs/object_label_only.yaml` — 只打标配置
5. **新建**: `tests/operators/test_object_holdable_labeler.py` — 单元测试