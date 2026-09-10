# BCRNet 模型消融配置

这些 YAML 是可直接训练的完整配置，不使用隐式继承。除文件名对应的模型开关外，训练、预处理和损失与 `configs/bcrnet.yaml` 相同。

| 文件 | 唯一预期结构变化 | 训练 mode |
| --- | --- | --- |
| base.yaml | use_tiny=false，基础检测基线 | `--stage A --mode base` |
| base_tiny.yaml | 基础检测 + tiny 辅助监督 | `--stage A --mode base_tiny` |
| no_detail.yaml | use_detail=false | `--mode full` |
| no_context.yaml | use_context=false | `--mode full` |
| no_tiny.yaml | use_tiny=false | `--mode full` |
| conv_refiner.yaml | refiner_type=conv | `--mode full` |

K、halo、core_size、width、blocks 和训练 seed 等网格实验，应复制最接近的完整 YAML，修改一个因素并使用新的文件名。不要在正式实验中只靠命令历史记录配置。

