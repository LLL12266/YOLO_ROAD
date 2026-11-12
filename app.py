import torch
from PIL import Image, ImageDraw, ImageFont
from flask import Flask, render_template, request, jsonify
import base64
from io import BytesIO

model = torch.hub.load(
    repo_or_dir="./yolov5-master",
    model="custom",
    path="./yolov5-master/runs/train/road_detection/exp2/weights/best.pt",
    source="local",
    force_reload=True,
    verbose=False,
    device="cuda:0"
)
model.conf = 0.25
model.iou = 0.45

app = Flask(__name__)

# 公共检测函数：接收图片流，返回 {result_img, boxes}
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
        # y偏移适配字号22
        draw.text((x1, y1 - 28), text, fill="white", font=font)

    buf = BytesIO()
    img.save(buf, format="JPEG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    result_img_src = f"data:image/jpeg;base64,{b64}"
    return result_img_src, boxes


@app.route("/")
def index():
    return render_template("index.html")

# 上传图片检测接口
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

# 摄像头帧检测接口（前端摄像头canvas截帧发过来）
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
