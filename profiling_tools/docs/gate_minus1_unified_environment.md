# Gate -1：统一 vLLM 与 vLLM-Omni 运行环境

## 锁定组合

```text
vLLM          0.31.0
vLLM-Omni     03c334818305d3b1666da01cd0f67d48a4f3e7a2
Transformers  5.17.0
PyTorch       cu130 build
Python        3.13
```

该 Omni 提交声明 `transformers>=5.13,<5.15`，而原生
`Cosmos3EdgeConfig` 位于 Transformers 5.17。实验只在隔离源码副本中把元数据约束
改为 `>=5.17,<5.18`；不修改 Cosmos 仓库的模型代码，也不修改主 `.venv`。

## 已完成的实测

在同一套 vLLM 0.31、Transformers 5.17 和注入的 Omni 源码中：

1. `AutoConfig` 正确解析 `Cosmos3EdgeConfig`；
2. vLLM 注册 `Cosmos3EdgeForConditionalGeneration`；
3. 成功导入 `Cosmos3OmniDiffusersPipeline`；
4. Edge Reasoner eager 服务完成 31-token prompt 和 32-token decode；
5. Edge Omni Generator 完成 256x256、2-step T2I 并输出 PNG。

结果位于（目录被 Git 忽略）：

```text
profiling_results/gate_minus1_tf517/reasoner_server.log
profiling_results/gate_minus1_tf517/reasoner_response.json
profiling_results/gate_minus1_tf517/generator_server.log
profiling_results/gate_minus1_tf517/generator_response.json
profiling_results/gate_minus1_tf517/generator_smoke.png
```

因此，Omni 的 `<5.15` 在这两条 Edge smoke path 上是保守测试上界，而非已经观察到的运行时不兼容。该结论不能自动推广到 Omni 的所有模型和功能。

## 创建持久化的隔离环境

在容器内执行：

```bash
cd /workspace/cosmos-framework
bash profiling_tools/scripts/setup_unified_vllm_omni_env.sh
```

默认输出：

```text
/workspace/cosmos-framework/.runtime/vllm_omni_tf517/venv
/workspace/cosmos-framework/.runtime/vllm_omni_tf517/vllm-omni
```

使用环境：

```bash
source /workspace/cosmos-framework/.runtime/vllm_omni_tf517/venv/bin/activate
```

## Gate -1 结论

方案 1 通过：Edge Reasoner 和 Edge Omni Generator 可以在同一个依赖组合中分别运行。

尚未证明：

- 两个 Engine 能在同一 PID 同时构造；
- 两者总显存峰值适合 32 GiB；
- 没有额外 GPU Worker；
- 普通 stream 或 Green Context 下可以并发。

这些属于 Gate 0.5 和 Gate 1。
