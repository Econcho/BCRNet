# BCRNet

**Budgeted Contextual Refinement Network**：面向单帧 RGB 反无人机检测的预算受限上下文细化网络。由 TSCRNet-V2 更名，当前为可修改、可扩展的 PyTorch **0.1 初版**。

轻量全图检测 → 候选覆盖与收益排序 → 仅对选中窗口读取细节/上下文 → 局部残差细化 → 多尺度检测。未选窗口保留基础预测。推理不使用红外、时序、首帧框或 GT。

## 文档入口

| 文档 | 内容 |
| --- | --- |
| [BCRNet 从零入门教程](docs/BCRNet入门教程/README.md) | 面向初学者的环境、抽帧、模型流程、训练、测试、消融与排错教程 |
| [DUT 数据集接入与训练](docs/DUT数据集接入与训练.md) | Pascal VOC DUT 转 COCO、训练、验证和跨数据集实验协议 |
| [Linux 服务器 DUT 全流程教程](docs/Linux服务器DUT完整训练测试教程.md) | 从 Linux 环境、DUT 原始数据到训练、测试和执行优化对照 |
| [训练系统现状与 YOLO 对标](docs/训练系统现状与UltralyticsYOLO差距.md) | 当前训练器能力、缺口、优先级和不应改变的 BCRNet 边界 |
| [环境与快速开始](docs/环境与快速开始.md) | 已创建环境、安装复现、训练/恢复/评估/推理/测速命令 |
| [BCRNet 详细方案](docs/BCRNet详细方案.md) | 完整 story、实际模块、张量含义、数据流 |
| [研究设计](docs/BCRNet研究设计.md) | 从 TSCRNet-V2 延续的完整研究推导、公式、创新边界 |
| [数据集与抽帧策略](docs/数据集与抽帧策略.md) | Anti-UAV 实际目录、RGB 抽帧策略、划分、其他数据集 |
| [训练与消融](docs/训练与消融.md) | 10 种 forward mode、A/B/C/D、监督与梯度、对照协议 |
| [扩展与实验原则](docs/扩展与实验原则.md) | 替换组件、配置接口、实现差异、后续优先级 |

## 最小调用

```python
import torch
from bcrnet import BCRNet, ModelConfig

model = BCRNet(ModelConfig(budget=16)).eval()
images = torch.randn(1, 3, 640, 640)  # 演示张量；真实输入必须按配置做 RGB 归一化
with torch.no_grad():
    base = model(images, mode="base")
    full = model(images, mode="full")
    features = model(images, mode="features")
    random_windows = model(images, mode="random",
                           generator=torch.Generator().manual_seed(42))
print(full.predictions["p2"].shape)  # [1, 5, 160, 160]，尚未解码的 logits/offset/logwh
```

完整默认配置在 [configs/bcrnet.yaml](configs/bcrnet.yaml)，工程冒烟配置在 [configs/smoke.yaml](configs/smoke.yaml)。

已创建 Anaconda 环境 **bcrnet**，安装 Python 3.11、PyTorch 2.7.1 + CUDA 12.8。抽帧脚本 [scripts/extract_antiuav.py](scripts/extract_antiuav.py) 支持 YAML/CLI 参数、正负帧独立步长、tiny/状态边界采样开关、独立 val/test 步长、自定义序列划分和导出选项；脚本仍然**只编写，未对真实数据运行**。已执行的训练只使用程序生成的合成图片，不能用于证明检测精度或研究创新。

## 代码结构

```text
src/bcrnet/
  config.py                 类型化模型配置
  models/                   backbone / neck / head / router / refinement / windows
  losses.py                 标签、检测损失、反事实局部收益监督
  training.py               阶段冻结、训练窗口、额外收益探测
  engine.py                 单设备训练循环、状态恢复
  data/                     COCO 接口、RGB 几何变换、Anti-UAV 纯采样逻辑
  inference.py              解码、NMS、原图坐标还原
  evaluation.py             COCO AP、负帧误报、窗口覆盖
  checkpoint.py             版本化 checkpoint、随机状态
  profiling.py              已执行算子的 MAC 统计
  execution/                可选索引感知执行层：计划、共享上下文、索引注意力和策略实验
  cli.py                    命令行
configs/  scripts/  tests/  docs/  # scripts includes Anti-UAV extraction and DUT VOC→COCO conversion
```

## 可选执行优化

执行层保持 BCRNet 的权重、路由和预测定义不变，只替换推理阶段的局部细化执行方式。通过
`configs/execution.example.yaml` 中的 `enabled` 开关控制：`true` 启用执行优化，`false` 完全回到原始
BCRNet 推理路径。省略 `--execution-config` 也不会启用执行适配器。

```powershell
# 原始 BCRNet 路径
python -m bcrnet predict --checkpoint runs/dut_full/best.pt --image sample.jpg --output runs/predict_reference

# 启用 packed 执行
python -m bcrnet predict --checkpoint runs/dut_full/best.pt --image sample.jpg --execution-config configs/execution.example.yaml --output runs/predict_packed

# 使用执行层 CLI 做基准，但关闭优化开关
python -m bcrnet.execution benchmark --config configs/bcrnet.yaml --no-execution-enabled --output runs/execution_disabled.json
```
