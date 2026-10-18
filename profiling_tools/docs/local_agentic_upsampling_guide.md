# Cosmos3 agentic T2I：官方实现导读与本地化设计

本文档说明 NVIDIA `Cosmos3-Super-Text2Image/agentic_upsampling`
参考实现的控制流，以及如何将其中的远程 HTTP 调用替换成本地
Cosmos3-Edge Reasoner/Generator 调用。本文档是设计导读，不代表本地 runner
已经实现。

官方参考：

- `agentic_upsampling/run.py`
- `agentic_upsampling/runner.py`
- `agentic_upsampling/clients.py`
- `agentic_upsampling/rubric.py`
- `agentic_upsampling/prompt_upsampler.py`

官方代码地址：
<https://huggingface.co/nvidia/Cosmos3-Super-Text2Image/tree/main/agentic_upsampling>

## 1. 官方实现到底做了什么

官方 runner 由三个逻辑 client 组成：

1. `PromptRewriterClient`
   - `initial_prompt()`：把原始短 prompt 上采样为 Cosmos T2I JSON。
   - `rewrite_prompt_pair()`：根据 critic 结果联合改写下一轮 positive JSON
     prompt 和 negative prompt。
2. `ImageGenerationClient`
   - 调用 vLLM-Omni `/v1/images/generations`。
   - 每轮可用不同 seed 生成多个 candidate。
3. `VLMQualityJudge`
   - 将图片和原始用户 prompt 交给视觉语言模型。
   - 输出结构化质量分数、缺陷和改进建议。

官方默认模型并不是全都来自 Cosmos：

- initial upsampler：GPT-5.5；
- iterative rewriter：GPT-5.5；
- image generator：Cosmos3-Super-Text2Image；
- critic：Gemini 3.1 Pro Preview。

我们的目标是保留 runner 和数据协议，将 upsampler、rewriter、critic 换成
本地 Cosmos Reasoner，并将 generator 换成本地 Cosmos generator。

## 2. 完整状态机

```text
original user prompt
        |
        v
initial_prompt (text-only Reasoner)
        |
        v
positive prompt JSON + initial negative prompt
        |
        v
+-------------------------------- iteration i -------------------------------+
| generate N images with fixed seed slots                                     |
|        |                                                                    |
|        v                                                                    |
| critic(image, original user prompt) for every image                         |
|        |                                                                    |
|        v                                                                    |
| choose the best candidate in this iteration                                 |
|        |                                                                    |
|        +---- threshold cleared ----> finish                                 |
|        |                                                                    |
|        v                                                                    |
| rewrite(previous positive, previous negative, critic analysis, history)     |
|        |                                                                    |
|        v                                                                    |
| next positive prompt JSON + next negative prompt                            |
+----------------------------------------------------------------------------+
```

第 0 轮使用 `initial_prompt()`。第 1 轮及以后才调用
`rewrite_prompt_pair()`。官方默认最多 2 轮、每轮 3 张图片。

不同 iteration 的同一 sample slot 复用相同 seed。例如 slot 0 每轮使用同一个
seed，从而尽量将图片变化归因于 prompt rewrite，而不是不同随机噪声。

## 3. Initial prompt upsampling 协议

输入是一个普通自然语言 prompt，输出必须是结构化 T2I JSON。官方最小校验字段：

```text
subjects
subject_details
background_setting
lighting
text_and_signage_elements
resolution
aspect_ratio
comprehensive_t2i_caption
```

官方实现会在模型返回后覆盖 `resolution` 和 `aspect_ratio`，而不是相信语言
模型复制的值。我们的本地实现也应这样做。

本地映射：

```text
remote OpenAI chat completion
    -> local Cosmos model_mode=reasoner, text-only
```

应复用仓库中的 Reasoner T2I upsampling 模板和 JSON schema，而不是临时拼接一个
普通扩写 prompt。

## 4. Generator 协议

官方 HTTP payload 的关键字段为：

```text
prompt                 compact JSON string
negative_prompt        string
num_inference_steps    default 50
guidance_scale         default 4.0
flow_shift             default 3.0
seed                   fixed per sample slot
size                   default 1024x1024 for Super
```

本地映射：

```text
remote /v1/images/generations
    -> local model_mode=text2image
```

需要把官方字段映射到本仓库 `OmniSampleArgs`：

```text
num_inference_steps -> num_steps
guidance_scale      -> guidance
flow_shift          -> flow_shift
prompt JSON object  -> compact JSON string in prompt
```

分辨率必须采用目标 checkpoint 支持的档位。官方 Super demo 的 768/1024x1024
不能直接当作 Edge 的默认配置照搬。

## 5. Critic 协议

critic 的输入不是上采样后的完整 JSON，而是：

```text
original user prompt + generated image
```

它返回：

```json
{
  "prompt_adherence_score": 0,
  "visual_quality_score": 0,
  "aesthetics_score": 0,
  "physical_plausibility_score": 0,
  "category_score": 0,
  "text_rendering_score": null,
  "photorealism_score": null,
  "overall_score": 0,
  "issues": [
    {
      "category": "...",
      "description": "...",
      "severity": "minor | moderate | severe"
    }
  ],
  "prompt_elements": {},
  "category_findings": {},
  "improvement_directives": [],
  "rationale": "..."
}
```

本地映射：

```text
remote Gemini multimodal chat completion
    -> local Cosmos model_mode=reasoner
       prompt=official judge rubric
       vision_path=generated image
```

官方 threshold 不是由 VLM 自己决定，而是在本地代码中确定：

- `overall_score >= 9.0`；
- `prompt_adherence_score >= 9.0`；
- 不存在 severe issue；
- 如果有 `text_rendering_score`，它也必须 `>= 9.0`。

因此本地 Reasoner 只负责输出评分 JSON；分数归一化、阈值判断和 early stop
仍应使用确定性的 Python 逻辑。

候选图片的排序顺序是：

```text
overall_score
prompt_adherence_score
category_score
visual_quality_score
aesthetics_score
更早的 iteration（平分时优先）
```

## 6. Rewrite 协议

critic 不直接产生下一轮 prompt。独立 rewriter 的输入包含：

```text
original user prompt
application-specific rewrite guidance
previous critic analysis
iteration history summary
previous positive prompt JSON
previous negative prompt
previous prompt 的 top-level schema keys
```

输出严格为：

```json
{
  "positive_prompt": {},
  "negative_prompt": "..."
}
```

约束包括：

- positive prompt 保持原 top-level keys 和类型；
- 保持上一轮 resolution 和 aspect_ratio；
- positive/negative prompt 不能互相矛盾；
- negative prompt 只抑制明确的错误替代和 artifact；
- 不得否定原始用户要求；
- positive `comprehensive_t2i_caption` 是生成指令，而不是解释修改过程。

本地映射：

```text
remote OpenAI chat completion
    -> local Cosmos model_mode=reasoner, text-only
```

## 7. `--no-refinement` 的建议语义

为了让 profiling workload 定义清楚，建议：

```text
--no-refinement
    initial upsampling
    -> exactly one T2I generation
    -> stop
```

该模式不运行 critic，也不运行 rewrite，并要求 `samples_per_iteration=1`。

注意：`max_iterations=1` 与 `no_refinement` 不等价。官方 runner 在一轮模式下
仍会运行 critic：

```text
max_iterations=1:
    upsample -> generate N -> critic N -> choose best -> stop
```

建议最终统一 runner 至少暴露三种实验模式：

```text
--no-refinement
    upsample -> generate

--max-iterations 1
    upsample -> generate -> critic

default
    upsample -> generate -> critic -> rewrite -> ...
```

## 8. 本地调用有两种实现层级

### 8.1 文件/子进程串行：先用于正确性验证

每个阶段生成下一阶段 JSON，然后启动一次
`python -m profiling_tools.runners.run_profiled`：

```text
Reasoner process exits
-> Generator process starts and exits
-> Critic Reasoner process starts and exits
-> Rewrite Reasoner process starts and exits
```

优点：

- 最容易复用现有 CLI 和 component profiling；
- 每个阶段输入输出可检查、可重放；
- 每个阶段可单独设置 GC 和 profiling 文件。

缺点：

- 每阶段重新加载 checkpoint；
- 冷启动时间和真实在线 agentic pipeline 不一致；
- 无法研究模型常驻情况下的阶段切换。

因此它适合“先跑通”，不适合作为最终端到端 latency 结论。

### 8.2 单进程本地 runner：最终 profiling 目标

`OmniInference` 已经能让同一个 pipeline 执行 `reasoner` 和非 reasoner 样本；
`generate_batch()` 会根据 `model_mode.is_reasoner` 分派到 Reasoner 或 Generator。

推荐只创建一次完整 `OmniInference`，然后在 agentic 循环中动态构造
`OmniSampleArgs` 并依次调用 `pipe.generate()`：

```text
load Cosmos3-Edge once
    -> pipe.generate(text-only reasoner sample)
    -> pipe.generate(T2I sample)
    -> pipe.generate(image reasoner sample)
    -> pipe.generate(text-only reasoner sample)
    -> ...
```

不要触发 CLI 的 `is_reasoner_only()` 优化。该优化会给纯 Reasoner 输入追加配置，
不加载 generation-side modules；统一 runner 必须从一开始加载完整模型。

还需注意：

- Reasoner 和 Generator 样本不能混在同一个 batch，但可以依次调用；
- critic 第一次使用图片时可能 lazy-load vision tower；
- 必须为每个阶段设置不同 output directory；
- `pipe.generate()` 自带 warmup/iteration 语义，agentic runner 需要避免每次阶段调用
  都意外执行多次 benchmark warmup；
- 若要测真实 warm pipeline，应将模型加载和首次 vision-tower 初始化排除在 steady-state
  iteration 外，同时单独保留 cold 数据。

## 9. 对 profiling 的阶段划分

建议记录：

```text
request_id
iteration
sample_index
stage                 upsample | generate | critic | rewrite
model_mode
input_text_tokens
input_vision_tokens
output_tokens
diffusion_steps
seed
cold_or_warm
wall_ms
component_gpu_ms
```

Reasoner 阶段继续拆为：

```text
vision_encoder
reasoner_prefill
reasoner_decode
```

Generator 阶段继续拆为：

```text
condition_prepare
diffusion_sampling
vae_decode
```

upsample/rewrite 是 text-only Reasoner，没有 `vision_encoder`；critic 是
image-conditioned Reasoner，包含 `vision_encoder`。

## 10. 推荐实施顺序

1. 原样保存官方 rubric、normalization、candidate ranking 和 rewrite contract。
2. 用现有 CLI 实现文件串行版，验证每个 JSON contract。
3. 增加 `--no-refinement`，验证纯 upsampling + T2I。
4. 加入单轮 critic，验证本地 Cosmos Reasoner 能稳定输出 rubric JSON。
5. 加入 rewrite 和第 2 轮生成。
6. 最后改为单进程、模型常驻版本。
7. 在单进程版本上加入现有 component profiling 和 Green Context 控制。

最主要的风险不是图片生成，而是本地 Reasoner 对长 JSON 的格式稳定性。实现时应
保留官方的 JSON extraction、schema validation、最多三次 retry，以及所有中间文件，
避免格式错误被误记为模型性能或质量问题。
