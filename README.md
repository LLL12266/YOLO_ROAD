# YOLO_ROAD：道路病害智能检测

这是一个基于 **YOLOv5** 的道路场景目标检测项目，覆盖从 LabelMe 标注数据整理、YOLO 数据集转换、模型训练，到桌面端和 Web 端推理部署的完整流程。项目当前的数据集包含 19 类目标，例如坑槽、横向/纵向裂纹、网状裂纹、修补、井盖、锥形桶等，可用于道路巡检、病害统计和演示系统开发。

## 项目效果

训练结果保存在 `yolov5-master/runs/train/road_detection/exp2/`，当前推理示例保存在 `yolov5-master/runs/detect/road_detect/exp/`：

<p>
  <img src="yolov5-master/runs/detect/road_detect/exp/1.jpg" alt="道路目标检测示例 1" width="48%">
  <img src="yolov5-master/runs/detect/road_detect/exp/2.jpg" alt="道路目标检测示例 2" width="48%">
</p>

训练曲线示例：

![训练曲线](yolov5-master/runs/train/road_detection/exp2/results.png)

当前数据集约有 4,771 张训练图片、1,193 张验证图片，类别共 19 个：

`人`、`圆形井盖`、`圆形桶`、`圆形雨水盖`、`坑槽`、`块状修补`、`抛洒物`、`护栏损坏`、`方形雨水盖`、`松散`、`标线模糊`、`横向修补`、`横向裂纹`、`纵向修补`、`纵向裂纹`、`纵向裂缝`、`网状修补`、`网状裂纹`、`锥形桶`。

## 处理流程

```text
LabelMe 原始图片/标注
        │
        ▼
yolo_convert/convert_to_yolo.py
        │  生成 train、valid、data.yaml、classes.txt
        ▼
yolov5-master/train.py  ──► runs/train/.../weights/best.pt
        │
        ├── yolov5-master/detect.py  （命令行批量推理）
        ├── Roadect_Pyqt/main.py     （PyQt5 图片/视频/摄像头）
        └── app.py + templates/      （Flask 网页端）
```

## 目录说明

| 路径 | 用途 | 使用时需要注意 |
| --- | --- | --- |
| `yolo_orignal_data/` | LabelMe 原始数据，包含 `.jpg` 图片和 `.json` 标注文件。 | 原始输入数据，建议保留备份，不直接拿来训练。目录内也有一份转换说明。 |
| `yolo_convert/` | 数据集转换工具。`convert_to_yolo.py` 将 LabelMe rectangle/polygon 转成 YOLO 检测格式。 | 修改数据集或重新划分训练/验证集时阅读 `yolo_convert/YOLO转换说明.md`。 |
| `yolov5_datasets/RoadDataset/` | 已生成的 YOLO 数据集。`train/`、`valid/` 保存图片和标签，`detect/` 保存待推理图片，`data.yaml` 保存类别和路径。 | 移动项目后要同步修改 `data.yaml` 中的 `train`、`val` 路径；类别编号以 `classes.txt` 为准。 |
| `yolov5-master/` | YOLOv5 源码、预训练权重、训练脚本和实验输出。 | 训练、验证、命令行推理都在此目录执行；`runs/train/` 和 `runs/detect/` 是实验结果。 |
| `Roadect_Pyqt/` | PyQt5 桌面端。支持图片、视频、摄像头检测，调整 conf/IoU 阈值，查看检测表格并保存结果图。 | 运行前检查 `main.py` 顶部的 `MODEL_REPO`、`MODEL_WEIGHTS` 和 `CLASS_NAMES`。 |
| `templates/` | Flask 使用的前端模板 `index.html`，提供图片上传和摄像头检测页面。 | 不要单独打开 HTML；应由 `app.py` 启动 Flask 后访问。 |
| `app.py` | Flask 后端。加载训练权重，提供网页首页、`/detect` 图片接口和 `/detect_camera` 摄像头帧接口。 | 当前默认使用 `yolov5-master/runs/train/road_detection/exp2/weights/best.pt`，并监听 `127.0.0.1:5000`。 |
| `run` | 常用的训练和批量推理命令记录。 | 执行前确认数据集目录名称是 `RoadDataset`，并按本 README 的路径示例修正命令。 |
| `yolo模型前端部署+公网穿透.md` | Flask 部署、FRP 内网穿透和公网访问的详细教程。 | 需要把网页发布给局域网或公网用户时阅读；其中的服务器 IP、端口、token 必须替换成自己的配置。 |

项目还包含 `.idea/`（IDE 配置）以及 `yolov5-master/runs/` 下的历史实验结果，一般不需要修改源码即可使用。

## 文档阅读顺序

如果是第一次使用，建议按下面顺序阅读：

1. 先阅读本文，了解目录和整体流程。
2. 需要从原始 LabelMe 数据重新生成数据集时，阅读 [`yolo_convert/YOLO转换说明.md`](yolo_convert/YOLO转换说明.md)，然后运行转换脚本。
3. 需要重新训练模型时，阅读 `yolov5-master/README.md` 中与训练、验证相关的部分，并参考本文的训练命令。
4. 需要启动网页端或做公网演示时，阅读 [`yolo模型前端部署+公网穿透.md`](yolo模型前端部署+公网穿透.md)。
5. 需要使用桌面端时，阅读 `Roadect_Pyqt/main.py` 顶部配置区，确认权重路径和类别名称后再运行。

`yolo_orignal_data/RoadDataset/YOLO转换说明.md` 是原始数据目录中保留的同主题说明，内容与 `yolo_convert/YOLO转换说明.md` 接近；以 `yolo_convert/` 下的脚本和说明为准即可。

## 环境安装

建议使用 Python 3.9 及以上，并在项目根目录创建虚拟环境：

```powershell
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r yolov5-master\requirements.txt
pip install Flask PyQt5 pandas
```

`torch`/`torchvision` 应根据自己的 CUDA 版本从 [PyTorch 官网](https://pytorch.org/get-started/locally/)选择安装命令；没有 NVIDIA GPU 时也可以使用 CPU，但推理和训练速度会明显降低。`PyQt5`、`opencv-python`、`Pillow` 和 `pandas` 分别用于桌面界面、视频/摄像头、图像处理和检测结果表格。

## 快速开始

### 1. 使用已有权重进行命令行检测

在项目根目录执行：

```powershell
cd yolov5-master
python detect.py `
  --weights runs\train\road_detection\exp2\weights\best.pt `
  --source ..\yolov5_datasets\RoadDataset\detect `
  --project runs\detect\road_detect
```

结果会写入 `yolov5-master/runs/detect/road_detect/`。

### 2. 重新训练模型

```powershell
cd yolov5-master
python train.py `
  --data ..\yolov5_datasets\RoadDataset\data.yaml `
  --weights weights\yolov5s.pt `
  --project runs\train\road_detection `
  --name exp
```

训练完成后，在对应实验目录的 `weights/best.pt` 中找到最佳权重。若使用新的实验目录，记得同步修改 `app.py` 和 `Roadect_Pyqt/main.py` 中的权重路径。

### 3. 启动 PyQt5 桌面端

先打开 `Roadect_Pyqt/main.py`，把 `MODEL_REPO`、`MODEL_WEIGHTS` 改成当前机器上的绝对路径，并让 `CLASS_NAMES` 与 `data.yaml`/模型类别保持一致。当前 `data.yaml` 有 19 类，而桌面端配置区仍是示例性的 3 类名称，重新训练 19 类模型后必须检查这一项。

```powershell
cd Roadect_Pyqt
python main.py
```

### 4. 启动 Flask 网页端

确认 `app.py` 中的权重路径存在后，在项目根目录执行：

```powershell
python app.py
```

浏览器打开 <http://127.0.0.1:5000>。网页端支持上传图片检测和摄像头帧检测；接口返回检测后的 Base64 图片以及目标框、置信度和类别名称。

如需让其他设备访问，先确认本机防火墙和 Flask 监听地址，再按 [`yolo模型前端部署+公网穿透.md`](yolo模型前端部署+公网穿透.md) 配置 FRP。公网部署会涉及云服务器端口和安全策略，请不要直接复用文档中的示例 token。

## 从 LabelMe 重新生成数据集

```powershell
cd yolo_convert
python convert_to_yolo.py `
  --input ..\yolo_orignal_data\RoadDataset `
  --output ..\yolov5_datasets\RoadDataset_new `
  --val-ratio 0.2 `
  --seed 42
```

脚本会生成 `train/images`、`train/labels`、`valid/images`、`valid/labels`、`data.yaml`、`classes.txt` 和 `conversion_report.json`。输出目录必须不存在；若数据中存在异常标注，可先查看报告，再按转换说明决定是否使用 `--skip-invalid`。

## 重要配置检查

- `data.yaml` 的类别数量（当前 `nc: 19`）必须与 `names`、`classes.txt` 和标注中的类别编号一致。
- `app.py`、`Roadect_Pyqt/main.py` 使用的权重必须来自当前训练实验，不能误用旧实验的 `best.pt`。
- `app.py` 默认使用 `device="cuda:0"`；没有 CUDA 时请改为 CPU，或参考代码中的设备判断方式。
- `Roadect_Pyqt/main.py` 的硬编码路径只适用于原项目目录；换电脑或换目录后必须修改。
- 连续道路视频帧随机划分可能导致训练集和验证集相邻，验证指标偏高；正式评估建议按路段或视频序列划分。

## 许可证与致谢

本项目使用 YOLOv5 作为检测框架。使用、发布或商业化前，请同时遵循本项目依赖库及 YOLOv5 源码所附许可证要求。
