# yolo模型前端部署\+公网穿透

# 一、简单介绍

我们训练好 YOLO 模型之后，常用 Flask 搭建 Web 服务：Flask 作为后端，加载 YOLO 权重，提供推理接口；同时写简单前端页面，支持上传图片、展示检测结果。

默认情况下，Flask 服务启动在 `127.0.0.1:5000`，**只能本机浏览器访问**；就算改成 `0.0.0.0:5000`，也仅仅是同一个局域网内的设备可以访问，外网（别的手机、别的地方电脑）访问不到。

想要让互联网上其他人直接打开浏览器访问你的 YOLO 检测网页，有两种主流方案：

1. 把 YOLO 模型、Flask 项目全部放到云服务器上跑（对云服务器配置要求高，GPU 贵）

2. **本地电脑跑 YOLO\+Flask，借助内网穿透（FRP）把本地服务暴露到公网**（我们本篇用这个，省钱，适合演示、大创、竞赛展示）

FRP（Fast Reverse Proxy）反向代理就是实现公网穿透的工具。 原理： 准备一台带公网 IP 的云服务器，上面部署 frps（服务端）；你本地电脑运行 frpc（客户端），**frpc 主动向外发起连接，和云服务器 frps 建立一条稳定长隧道**。外网用户访问云服务器 IP \+ 映射端口，流量先到云服务器 frps，再通过这条隧道转发给你本地电脑上的 Flask 服务，最后 YOLO 推理、返回画框图片给前端页面。

前置条件： 

1\.一台云服务器（Ubuntu / CentOS，x86\_64 架构，有公网 IP）

2\.本地 Windows 电脑（跑 YOLO、Flask、frpc 客户端） 

3\.frp 版本统一：例如：0\.58\.1（服务端、客户端版本一致）

# 二、Flask前端部署

## 2\.1flask后端

app\.py示例，导入模型加入简单的接口

```Python
import torch
from PIL import Image, ImageDraw, ImageFont
from flask import Flask, render_template, request, jsonify
import base64
from io import BytesIO

model = torch.hub.load(
    repo_or_dir="yolov5", #指向本地 yolov5 源码文件夹
    model="custom",   #自定义训练权重
    path="best.pt",    #选择模型权重
    source="local",    #local = 使用本地源码，不联网下载仓库
    force_reload=True,  #True：每次运行都重新加载，；False：会缓存模型，第二次加载更快。
    verbose=False,    #关闭加载时的大量打印日志
    device="cuda:0" #指定推理设备：**第 0 块显卡**
)
model.conf = 0.25 #只保留置信度 ≥ 0.25 的检测框
model.iou = 0.45 #NMS 的 IOU 阈值

app = Flask(__name__)

*# 公共检测函数：接收图片流，返回 {result_img, boxes}*
def detect_img(file_stream):
    img = Image.open(file_stream).convert("RGB")
    results = model(img)

    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("simhei.ttf", 22)
    except:
        font = ImageFont.load_default()

    boxes = []
    for *xyxy, conf, cls in results.xyxy[0]:
        x1, y1, x2, y2 = map(float, xyxy)
        cid = int(cls)
        cls_name = model.names[cid]
        boxes.append({
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "conf": float(conf),
            "cls_name": cls_name
        })
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        text = f"{cls_name} {conf:.2f}"
        *# y偏移适配字号22*
*        *draw.text((x1, y1 - 28), text, fill="white", font=font)

    buf = BytesIO()
    img.save(buf, format="JPEG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    result_img_src = f"data:image/jpeg;base64,{b64}"
    return result_img_src, boxes

#应用前端页面模版，需要在app.py同级目录下有template/index.html
@app.route("/")
def index():
    return render_template("index.html")

*# 上传图片检测接口*
@app.route('/detect', methods=['POST'])
def detect():
    file = request.files.get("img")
    if not file:
        return jsonify({"code": 400, "msg": "没有上传图片"})
    result_img, boxes = detect_img(file.stream)
    return jsonify({
        "code": 200,
        "result_img": result_img,
        "boxes": boxes
    })

*# 摄像头帧检测接口（前端摄像头canvas截帧发过来）*
@app.route('/detect_camera', methods=['POST'])
def detect_camera():
    file = request.files.get("img")
    if not file:
        return jsonify({"code": 400, "msg": "没有收到摄像头图像帧"})
    result_img, boxes = detect_img(file.stream)
    return jsonify({
        "code": 200,
        "result_img": result_img,
        "boxes": boxes
    })

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
```

## 2\.2前端页面

Flask 默认约定：模板文件放在项目根目录下 **templates**（文件夹，里面放 `index.html`。 不用自己苦写前端页面：把后端接口逻辑、返回格式发给 AI，让 AI 直接生成页面，丢进 `templates/index.html` 即可，速度快、样式好看

## 2\.3运行

```Python
python app.py
```

打开浏览器访问 `http://127.0.0.1:5000`，可以上传图片、看到 YOLO 检测结果

# 三、公网穿透

## 3\.1 下载frp

下载地址： https://github\.com/fatedier/frp/releases?page=3

下载的两端版本一致即可，不需要固定使用v**0\.58\.1**

- 云服务器（Ubuntu/CentOS x86\_64）：**frp\_0\.58\.1\_linux\_amd64\.tar\.gz**（这里面就是 frps，服务端）

- 本地 Windows 电脑：**frp\_0\.58\.1\_windows\_amd64\.zip**（里面是 frpc，客户端）

<img alt="\{B50C1F29\-A897\-415A\-A6ED\-90346B22006D\}\.png" src="C:\Users\11408\Downloads\yolo模型前端部署+公网穿透\图片和附件\{B50C1F29-A897-415A-A6ED-90346B22006D}.png"/>

## 3\.2、云服务器端\-frps

上传 `frp_0.58.1_linux_amd64.tar.gz` 到服务器

下载过后放到服务器任意文件夹下，以/usr/local/package/为例

```Python
#到/usr/local/package/
cd /usr/local/package/

#解压命令，到soft目录
tar -xvf frp_0.58.1_linux_amd64.tar.gz  -C /usr/local/soft/

#cd到文件夹
cd /usr/local/soft/frp_0.58.1_linux_amd64.tar.gz

#编辑配置文件，没有frpc.ini，自己创建
vim frpc.ini
#输入
[common]
server_addr = 这里填你的云服务器公网IP   # frps所在的服务器IP
server_port = 7000                     # frps监听端口，frpc和frps之间通信用，云防火墙要放行7000
token = yolov5123456                   # 验证密钥，frps、frpc两边必须一模一样

[web-yolo]
type = tcp                              # 转发类型tcp
local_ip = 127.0.0.1                    # 你本机Windows上Flask服务的地址
local_port = 5000                       # Flask后端的端口（app.py跑在5000）
remote_port = 8080                      # 云服务器上对外开放给浏览器访问的端口，防火墙放行8080#
```

云服务器公网IP查询，服务器命令行输入

```Python
curl ifconfig.me
```

开启frps：

```Python
./frps -c frps.ini
```

后台常驻：

```Python
nohup ./frps -c frps.ini > frps.log 2>&1 &

tail -f frps.log

#输出类似既是成功
frps uses config file: frps.ini
frps tcp listen on 0.0.0.0:7000
frps started successfully
```

关闭 frps：

```Python
pkill frps
```

为了避免防护墙拦截端口：

登录服务器，检查 ufw 状态：

```Plain Text
ufw status
```

- 如果显示 `Status: active`（防火墙开启），放行 7000：

```Plain Text
ufw allow 7000/tcp
ufw allow 8080/tcp
```

确认服务器上 frps 确实在监听 7000

服务器执行：

```Plain Text
netstat -tlnp | grep 7000
```

或者

```Plain Text
ss -tlnp | grep 7000
```

能看到 `0.0.0.0:7000` 就是 frps 正常监听。

> 如果没输出，说明 frps 没跑起来，重新执行 nohup 那一条。
> 
> 

测试连通性（在 Windows 上测试端口通不通）

新开一个 PowerShell，测试 7000 端口：

```Plain Text
Test-NetConnection 云服务器公网IP -Port 7000
```

- TcpTestSucceeded: True ✅

若是还没有用：

去服务器网站

1. 找到你的这台机器，点【防火墙】标签页

2. 添加规则： 

    - 协议：TCP

    - 端口：7000

    - 源：0\.0\.0\.0/0

    - 备注：frp 通信端口

3. 再加一条： 

    - 协议：TCP

    - 端口：8080

    - 源：0\.0\.0\.0/0

    - 备注：yolo 网页访问

    再去测试连通性

## 3\.3windows端\-frpc

**frp\_0\.58\.1\_windows\_amd64\.zip **解压到windows任意文件夹

该目录下创建frpc\.ini

![\{1E8BDCF3\-5CC1\-4381\-A34D\-F5E219DCEE29\}\.png](C:\Users\11408\Downloads\yolo模型前端部署+公网穿透\图片和附件\{1E8BDCF3-5CC1-4381-A34D-F5E219DCEE29}.png)

编辑frpc\.ini，填写一下内容并保存

```Python
[common]
server_addr = 这里填你的云服务器公网IP
server_port = 7000
token = yolov5123456

[web-yolo]
type = tcp
local_ip = 127.0.0.1
local_port = 5000
remote_port = 8080
```

windows开启：

要在该目录下的命令行才可以

```Python
.\frpc.exe -c frpc.ini
```

完整启动：

1、服务器

```Python
cd /usr/local/soft/frp_0.58.1_linux_amd64
nohup ./frps -c frps.ini > frps.log 2>&1 &
```

2、flask端

运行app\.py

3、windows

```Python
#先测试是否连通,True则是连通
Test-NetConnection 云服务器公网IP -Port 7000
#到文件夹下面开启管理员命令行，看到 start proxy success 代表隧道搭建成功
.\frpc.exe -c frpc.ini
```

浏览器访问：

这样其他人就可以访问你的页面，做到公网穿透

```Python
http://你的云服务器公网IP:8080
```

停止顺序

1. 关闭 PowerShell 的 frpc 窗口

2. 关闭 Flask 的 Anaconda 终端

3. 服务器执行关闭 frps

```Plain Text
pkill frps
```

# 四、总结

本次使用 Flask 完成 YOLO 模型 Web 部署，搭配 FRP 公网穿透，实现外网浏览器访问检测页面。内网穿透工具种类较多。 该方案依赖本地 Windows 主机，每次使用需要手动启动 Flask 和 frpc，操作繁琐，且电脑关机后服务失效。优化方案：将 Flask 与 YOLO 模型直接部署在云服务器并配置常驻运行，实现服务永久在线



