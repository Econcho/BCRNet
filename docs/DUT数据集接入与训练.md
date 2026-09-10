# DUT 数据集接入与 BCRNet 使用

## 1. 已确认的 DUT 格式

只读检查 `F:\antiuav\dataset\DUT` 得到：

```text
DUT/
  train/img/*.jpg
  train/xml/*.xml
  val/img/*.jpg
  val/xml/*.xml
  test/img/*.jpg
  test/xml/*.xml
```

当前数量为 train 5200、val 2600、test 2200，共 10000 张图片；XML 与图片数量一致。标注是 Pascal VOC XML，主类别为 `UAV`。按 VOC 闭区间坐标解释，共有 10109 个有效目标。发现 3 个 XML 没有 object，它们会作为合法的负样本图片保留：

- `train/xml/00579.xml`
- `train/xml/00639.xml`
- `train/xml/00724.xml`

图片尺寸并不固定，样例中既有 550×412，也有 1280×720。BCRNet 的 Letterbox 可以处理这种输入，训练输入尺寸仍由模型配置统一为 640×640。

## 2. 为什么要先转换成 COCO

BCRNet 当前的数据边界是 COCO 风格的统一接口：

```text
dataset config → build_dataset → image, valid_mask, target
                                     ├─ boxes: 输入坐标 xyxy
                                     ├─ labels: 从 0 开始
                                     ├─ ignore_boxes
                                     └─ meta
```

评估器也直接使用 COCO API 计算 AP。因此最稳妥的做法是把 DUT 的 VOC 标注转换为 COCO manifest，继续复用已经验证过的训练、评估、checkpoint 和类别映射逻辑，而不是在训练器中增加一套与 COCO 分叉的 DUT 特例。

转换脚本是 [scripts/convert_dut.py](../scripts/convert_dut.py)，它支持：

- reference 模式：不复制图片，manifest 引用原始 DUT；
- copy 模式：复制图片，生成自包含的转换目录；
- 空 XML 负样本；
- 框裁剪、尺寸校验、未知类别策略；
- Pascal VOC 闭区间与半开区间坐标模式；
- difficult 框的 include/ignore/skip；
- dry-run 和 conversion_report。

## 3. 第一次转换：先 dry-run

不要直接写正式目录。先执行只读计划：

```powershell
Set-Location 'F:\antiuav\TSCRNet\BCRNet-main'
conda activate bcrnet

python scripts/convert_dut.py `
  --source-root 'F:\antiuav\dataset\DUT' `
  --output-root 'F:\antiuav\dataset\DUT_coco_reference_v1' `
  --image-mode reference `
  --dry-run
```

dry-run 应报告三份 split 的图片和框数量、负样本数量，以及 skipped 列表。它不建立输出目录，也不复制图片。

严格模式会把数据异常当作错误。DUT 使用 Pascal VOC 风格的最大坐标，默认 `coordinate_mode=inclusive`；`val/xml/00991.xml` 中的 `[1056,443,1059,443]` 因而表示高度为 1 像素的框。只有错误地使用 `half_open` 时它才会被视为退化框。不要为了绕过错误随意切换坐标模式。

reference 模式不会修改原始 DUT，训练时从原始图片读取，节省磁盘空间。若需要把数据交给另一台机器或做不可变归档，使用 copy 模式：

```powershell
python scripts/convert_dut.py `
  --source-root 'F:\antiuav\dataset\DUT' `
  --output-root 'F:\antiuav\dataset\DUT_coco_copy_v1' `
  --image-mode copy `
  --coordinate-mode inclusive
```

对当前 DUT，使用默认严格模式和 `--coordinate-mode inclusive` 即可完整转换：

```powershell
python scripts/convert_dut.py `
  --source-root 'F:\antiuav\dataset\DUT' `
  --output-root 'F:\antiuav\dataset\DUT_coco_reference_v1' `
  --image-mode reference `
  --coordinate-mode inclusive
```

本次只读扫描得到的预期结果是：train 5200 张/5243 框/3 个负样本，val 2600 张/2621 个框，test 2200 张/2245 个框；`conversion_report.json` 应记录 0 条 skipped。若将来数据包发生变化，应以新的 dry-run 报告为准。

脚本会拒绝覆盖已存在的输出目录。新策略或修复后的重试应使用带版本号的新目录。

## 4. 转换结果

reference 模式输出：

```text
DUT_coco_reference_v1/
  annotations/train.json
  annotations/val.json
  annotations/test.json
  conversion_report.json
  data.yaml
```

生成的 `data.yaml` 会把 `root` 指向原始 DUT，把 annotation 路径指向转换目录。COCO image 的 `file_name` 是 `train/img/00001.jpg` 这类相对路径。

copy 模式还会有：

```text
DUT_coco_copy_v1/
  images/train/img/*.jpg
  images/val/img/*.jpg
  images/test/img/*.jpg
```

训练前检查 `data.yaml`：

```powershell
Get-Content 'F:\antiuav\dataset\DUT_coco_reference_v1\data.yaml'
```

然后随机打开正样本和 3 个空标注样本，确认图片路径与框对齐。BCRNet 会再次验证图片尺寸、框面积和类别映射。

## 5. 用 DUT 训练 BCRNet

先用默认结构跑一个真实数据单 batch 冒烟：

```powershell
python -m bcrnet train `
  --config configs/bcrnet.yaml `
  --data 'F:\antiuav\dataset\DUT_coco_reference_v1\data.yaml' `
  --output runs/dut_smoke `
  --device cuda `
  --max-batches 1 `
  --stop-after-epochs 1
```

确认 loss 有限、GPU 显存可用、last.pt 生成后，再开始完整训练：

```powershell
python -m bcrnet train `
  --config configs/bcrnet.yaml `
  --data 'F:\antiuav\dataset\DUT_coco_reference_v1\data.yaml' `
  --output runs/dut_full_seed42 `
  --device cuda `
  --mode full `
  --workers 0
```

DUT 是单类别 UAV，`configs/bcrnet.yaml` 的 `num_classes: 1` 正好匹配。若未来将 DUT 与其他类别合并，必须重新生成 COCO categories、修改 `num_classes` 并开始新的实验，不能直接加载原单类 head 作为同一实验。

## 6. DUT 验证和测试

先看 val：

```powershell
python -m bcrnet evaluate `
  --checkpoint runs/dut_full_seed42/best.pt `
  --data 'F:\antiuav\dataset\DUT_coco_reference_v1\data.yaml' `
  --split val `
  --output runs/dut_eval_val_seed42 `
  --device cuda `
  --mode full
```

方案冻结后再看 test：

```powershell
python -m bcrnet evaluate `
  --checkpoint runs/dut_full_seed42/best.pt `
  --data 'F:\antiuav\dataset\DUT_coco_reference_v1\data.yaml' `
  --split test `
  --output runs/dut_eval_test_seed42 `
  --device cuda `
  --mode full
```

DUT 中没有视频序列 ID 语义，`sequence_id` 是 split 名；不要把 `sequence_id` 当作拍摄组做视频级统计。DUT 的负样本数量很少，误报率估计会比 Anti-UAV 更不稳定，应同时报告 false positives、precision/recall 和 PR 曲线。

## 7. DUT 上如何保持 BCRNet 研究问题一致

DUT 和 Anti-UAV 的图像尺寸、目标大小、背景和数据规模不同。不能直接比较一个数据集上的 AP 数字来宣称跨数据集提升。推荐两种协议：

1. **独立数据集协议**：在 Anti-UAV 和 DUT 上分别训练、分别在各自 val 选 checkpoint、分别在各自 test 报告结果。用于验证方法是否适用于不同 RGB 小目标分布。
2. **跨数据集迁移协议**：在 Anti-UAV 训练、DUT test；或反向进行。此时必须明确训练域和测试域，不能称作同分布检测。

在 DUT 上保持相同 BCRNet 结构、输入 640、tiny 定义和 K，先跑 base/full/dense，再跑路由和结构消融。若 DUT 的目标并不满足 Anti-UAV 的 tiny 分布，tiny 阈值应通过 train/val 预先分析后冻结，不能使用 test 调参。

## 8. 可能需要的数据策略

DUT 已经是抽帧后的图片集，因此不使用 Anti-UAV 的视频抽帧脚本。可选的数据实验应在 COCO manifest 层完成：

- 保留或移除 3 个空标注图片，作为极少量负样本敏感性分析；
- 按框面积对 train 做分层采样；
- 固定图片编码与 Letterbox，避免把预处理变化混入模型消融；
- 训练/验证/测试仍使用原始 split，不随机混合图片。

每个变体都应生成新的 manifest 和 `conversion_report.json`，并在实验表中写清楚图片数、框数和负样本数。
