# LabelMe 转 YOLO 检测数据集

脚本：`convert_to_yolo.py`，Python 3.9 及以上。转换本身仅使用标准库；安装 Pillow 时还会自动检查图片尺寸和完整性（`pip install pillow`）。

在本文件夹运行：

```powershell
python convert_to_yolo.py
```

指定其他数据目录、验证集比例和随机种子：

```powershell
python convert_to_yolo.py --input "D:\数据目录" --output "D:\输出目录" --val-ratio 0.2 --seed 42
```

默认输出到输入目录下的 `yolo_dataset`，包含：

```text
yolo_dataset/
  train/images/
  train/labels/
  valid/images/
  valid/labels/
  data.yaml
  classes.txt
  conversion_report.json
```

`valid` 是验证集文件夹名称；训练配置中使用 YOLO 标准键 `val: valid/images`。

每张图片对应一个同名 TXT，每个目标一行：

```text
class_id x_center y_center width height
```

中心坐标和宽高分别除以图片宽高，归一化到 0～1。类别从 0 开始，默认按名称排序，映射见 `classes.txt`（第一行为类别 0）和 `data.yaml`。如需在不同数据集间固定类别编号，请传入 `--classes "已有的classes.txt"`。

支持 LabelMe rectangle 和 polygon（多边形取外接矩形，用于目标检测，不是分割）。越界坐标裁剪到图片边界并计数；无效框、未知形状、图片缺失或尺寸不一致会终止转换，在创建输出前报告。确认允许跳过异常数据时可使用 `--skip-invalid`，详细原因会写入报告。

无 JSON 的图片不参与训练，防止把漏标图片当作背景；合法的空 shapes 会生成空 TXT。原始文件保留，图片复制到新目录。输出目录必须不存在，重复运行请指定新的 `--output`。

递归读取子目录，子目录同名文件通过附加路径哈希避免覆盖。固定输入、类别配置和随机种子时，划分结果一致。报告包含划分清单、各类目标数量、异常项和未使用图片。

当前目录是连续道路采集图像。默认随机划分可能让相邻帧分别进入训练和验证集，导致验证结果偏高；正式评估建议按采集路段或视频序列分组制作独立验证集。

Ultralytics 训练示例（另需安装 ultralytics）：

```powershell
yolo detect train model=yolo11n.pt data="yolo_dataset/data.yaml" epochs=100 imgsz=640
```

如果整体移动数据集，请将 `data.yaml` 的 `path` 改成移动后的数据集绝对路径。
