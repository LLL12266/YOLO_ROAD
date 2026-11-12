from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal, QThreadPool, QRunnable
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import QFileDialog, QMainWindow, QMessageBox, QApplication
from roadect import Ui_MainWindow
import sys
import os
import time
import torch
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pandas as pd

# ===================== 配置区（改成你自己的路径）=====================
MODEL_REPO    = r"D:\YOLO_ROAD\yolov5-master"
MODEL_WEIGHTS = r"D:\YOLO_ROAD\yolov5-master\runs\train\road_detection\exp2\weights\best.pt"
CLASS_NAMES   = ["裂缝", "坑槽", "修补"]

PALETTE = [
    (255, 56, 56), (255, 157, 151), (255, 112, 31), (255, 178, 29),
    (207, 210, 49), (72, 249, 10), (146, 204, 23), (61, 219, 134),
]

_FONT_CANDIDATES = [
    r"C:/Windows/Fonts/msyh.ttc",
    r"C:/Windows/Fonts/simhei.ttf",
]
_font_cache = {}

def get_font(size):
    if size in _font_cache:
        return _font_cache[size]
    for p in _FONT_CANDIDATES:
        if os.path.exists(p):
            f = ImageFont.truetype(p, size)
            _font_cache[size] = f
            return f
    f = ImageFont.load_default()
    _font_cache[size] = f
    return f

def draw_boxes(pil_img, det):
    draw = ImageDraw.Draw(pil_img)
    fsize = max(16, int(pil_img.width / 50))
    font = get_font(fsize)
    for _, r in det.iterrows():
        x1, y1, x2, y2 = int(r["xmin"]), int(r["ymin"]), int(r["xmax"]), int(r["ymax"])
        cls = int(r["class"])
        conf = float(r["confidence"])
        name = str(r["name"])
        color = PALETTE[cls % len(PALETTE)]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=max(2, fsize // 8))
        text = f"{name} {conf:.2f}"
        tb = draw.textbbox((0, 0), text, font=font)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        ty = y1 - th - 10
        if ty < 0:
            ty = y1 + 2
        draw.rectangle([x1, ty, x1 + tw + 10, ty + th + 8], fill=color)
        draw.text((x1 + 5, ty + 4), text, fill=(255,255,255), font=font)
    return pil_img

def pil_to_qpixmap(pil_img):
    arr = np.ascontiguousarray(np.array(pil_img))
    h, w, c = arr.shape
    qimg = QImage(arr.data, w, h, c * w, QImage.Format_RGB888).copy()
    return QPixmap.fromImage(qimg)

# ===================== 模型加载线程 =====================
class LoadModelThread(QThread):
    load_ok = pyqtSignal(object)
    load_fail = pyqtSignal(str)
    def run(self):
        try:
            model = torch.hub.load(
                repo_or_dir=MODEL_REPO,
                model="custom",
                path=MODEL_WEIGHTS,
                source="local",
                force_reload=False,
                verbose=False,
                device="cuda:0" if torch.cuda.is_available() else "cpu"
            )
            model.conf = 0.25
            model.iou = 0.45
            model.names = CLASS_NAMES
            self.load_ok.emit(model)
        except Exception as e:
            self.load_fail.emit(str(e))

# ===================== 推理任务 =====================
class InferTask(QRunnable):
    def __init__(self, model, pil_img, callback):
        super().__init__()
        self.model = model
        self.pil_img = pil_img
        self.callback = callback
    def run(self):
        t0 = time.time()
        res = self.model(self.pil_img)
        det = res.pandas().xyxy[0]
        cost = time.time() - t0
        out_img = draw_boxes(self.pil_img.copy(), det)
        self.callback(out_img, det, cost)

class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.model = None
        self.model_loaded = False
        self.is_loading = False
        self.is_inferring = False
        self.cap = None
        self.last_pil = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.frame_loop)
        self.fps_limit = 15
        self.thread_pool = QThreadPool()
        self.thread_pool.setMaxThreadCount(1)

        # ========== 绑定按钮 ==========
        self.btn_img.clicked.connect(self.detect_img)
        self.btn_video.clicked.connect(self.detect_video)
        self.btn_cam.clicked.connect(self.toggle_camera)
        self.btn_save.clicked.connect(self.save_result)
        self.btn_clear.clicked.connect(self.clear_all)
        self.sl_conf.valueChanged.connect(self.change_conf)
        self.sl_iou.valueChanged.connect(self.change_iou)

        self.lbl_status.setText("就绪，上传图片后加载模型")

    def lazy_load_model(self):
        if self.model_loaded:
            return True
        if self.is_loading:
            QMessageBox.information(self, "提示", "模型正在加载，请稍等...")
            return False
        self.is_loading = True
        self.lbl_status.setText(" 正在加载模型...")
        self.load_thread = LoadModelThread()
        self.load_thread.load_ok.connect(self.on_model_load_success)
        self.load_thread.load_fail.connect(self.on_model_load_fail)
        self.load_thread.finished.connect(lambda: setattr(self, "is_loading", False))
        self.load_thread.start()
        return False

    def on_model_load_success(self, model):
        self.model = model
        self.model_loaded = True
        self.model.conf = self.sl_conf.value() / 100
        self.model.iou = self.sl_iou.value() / 100
        self.lbl_status.setText("模型加载成功，可以检测")

    def on_model_load_fail(self, err_msg):
        QMessageBox.critical(self, "模型加载失败", err_msg)
        self.lbl_status.setText(" 模型加载失败")

    def release_model(self):
        """释放模型、清空显存，用完释放"""
        self.stop_stream()
        if self.model is not None:
            del self.model
            self.model = None
            self.model_loaded = False
            # 清空cuda缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        self.lbl_status.setText(" 模型已释放，显存释放完成")

    def change_conf(self, v):
        self.lbl_conf.setText(f"{v/100:.2f}")
        if self.model:
            self.model.conf = v / 100

    def change_iou(self, v):
        self.lbl_iou.setText(f"{v/100:.2f}")
        if self.model:
            self.model.iou = v / 100

    def on_infer_result(self, out_img, det, cost):
        self.is_inferring = False
        self.last_pil = out_img
        self.show_pic(out_img)
        self.fill_table(det)
        self.lbl_status.setText(f"检测 | {len(det)}个目标 | {cost*1000:.1f}ms")

    def detect_img(self):
        if not self.lazy_load_model():
            return
        self.stop_stream()
        path, _ = QFileDialog.getOpenFileName(self, "选择图片", "", "图片(*.jpg *.png *.jpeg)")
        if not path:
            return
        pil = Image.open(path).convert("RGB")
        self.run_infer(pil)

    def detect_video(self):
        if not self.lazy_load_model():
            return
        self.stop_stream()
        path, _ = QFileDialog.getOpenFileName(self, "选择视频", "", "视频(*.mp4 *.avi *.mov)")
        if not path:
            return
        self.cap = cv2.VideoCapture(path)
        self.timer.start(int(1000 / self.fps_limit))
        self.lbl_status.setText("视频检测中...")

    def toggle_camera(self):
        if self.btn_cam.isChecked():
            if not self.lazy_load_model():
                self.btn_cam.setChecked(False)
                return
            self.stop_stream()
            self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                QMessageBox.warning(self, "错误", "摄像头打开失败")
                self.btn_cam.setChecked(False)
                return
            self.timer.start(int(1000 / self.fps_limit))
            self.btn_cam.setText("关闭摄像头")
            self.lbl_status.setText("摄像头检测中")
        else:
            self.stop_stream()
            self.btn_cam.setText("打开摄像头")

    def frame_loop(self):
        if self.is_inferring:
            return
        ret, frame = self.cap.read()
        if not ret:
            self.stop_stream()
            self.lbl_status.setText("视频播放结束")
            return
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        self.run_infer(pil)

    def run_infer(self, pil_img):
        self.is_inferring = True
        task = InferTask(self.model, pil_img, self.on_infer_result)
        self.thread_pool.start(task)

    def stop_stream(self):
        self.timer.stop()
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def show_pic(self, pil_img):
        pix = pil_to_qpixmap(pil_img)
        self.label_show.setPixmap(pix.scaled(self.label_show.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def fill_table(self, det):
        self.table.setRowCount(len(det))
        for i, (_, row) in enumerate(det.iterrows()):
            self.table.setItem(i, 0, QApplication.translate("MainWindow", str(row["name"]), None))
            self.table.setItem(i, 1, QApplication.translate("MainWindow", f"{row['confidence']:.3f}", None))
            self.table.setItem(i, 2, QApplication.translate("MainWindow", f"{row['xmin']:.0f}", None))
            self.table.setItem(i, 3, QApplication.translate("MainWindow", f"{row['ymin']:.0f}", None))
            self.table.setItem(i, 4, QApplication.translate("MainWindow", f"{row['xmax']:.0f}", None))
            self.table.setItem(i, 5, QApplication.translate("MainWindow", f"{row['ymax']:.0f}", None))

    def save_result(self):
        if self.last_pil is None:
            QMessageBox.information(self, "提示", "暂无结果图")
            return
        save_path, _ = QFileDialog.getSaveFileName(self, "保存图片", "result.jpg", "图片(*.jpg *.png)")
        if save_path:
            self.last_pil.save(save_path)
            self.lbl_status.setText("图片已保存")

    def clear_all(self):
        # 清空画面 + 释放模型、释放显存
        self.release_model()
        self.label_show.clear()
        self.table.setRowCount(0)
        self.last_pil = None

    def closeEvent(self, event):
        self.release_model()
        self.thread_pool.waitForDone()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
