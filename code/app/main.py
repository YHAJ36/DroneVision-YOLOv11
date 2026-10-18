"""无人机视角目标检测上位机（PyQt5）

功能：图片检测 / 视频检测 / 实时摄像。

界面外观集中在文件顶部「外观配置区」的常量（APP_TITLE、BG_COLOR、
ACCENT_COLOR、BTN_*、GROUP_* 等），布局在 build_ui() 中，
按自己的需要调整即可。
"""

import os
import sys
from pathlib import Path

# 项目根目录：本脚本位于 code/app，向上 2 级即项目根，路径一律基于它拼
ROOT = Path(__file__).resolve().parents[2]


# 必须在 import ultralytics 之前设置，框架导入时就会读这个变量
os.environ.setdefault("YOLO_CONFIG_DIR",
                      str(ROOT / "ultralytics-cfg"))

# ⚠️ 导入顺序不能改：ultralytics(torch) 必须先于 PyQt5
#    反过来会报 OSError [WinError 1114] Error loading torch/lib/c10.dll
#    已实测复现，不是玄学
from ultralytics import YOLO

import cv2
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (QApplication, QFileDialog, QGroupBox, QHBoxLayout,
                             QLabel, QListWidget, QMainWindow, QMessageBox,
                             QPushButton, QVBoxLayout, QWidget)


# ═══════════════════════════════════════════════════════════════════
#  外观配置区 —— 改这里就等于换了一套界面
#  TODO: 把下面每一项都改成你自己的选择
# ═══════════════════════════════════════════════════════════════════

APP_TITLE = "无人机视角目标检测系统"      # TODO 窗口标题

BG_COLOR = "#1e1e2e"          # TODO 主背景色
PANEL_COLOR = "#2a2a3c"       # TODO 面板/分组框背景
TEXT_COLOR = "#e0e0e8"        # TODO 正文文字颜色
ACCENT_COLOR = "#4fc3f7"      # TODO 主题色（按钮、强调）
MUTED_COLOR = "#8888a0"       # 次要文字颜色

FONT_FAMILY = "Microsoft YaHei"   # TODO 字体
FONT_SIZE = 12                    # 字号

BTN_LOAD = "载入模型"          # TODO 按钮文案，全部自己起名
BTN_IMAGE = "图片检测"
BTN_VIDEO = "视频检测"
BTN_CAMERA = "实时摄像"
BTN_STOP = "停止"

GROUP_CTRL = "操作"            # TODO 分组框标题
GROUP_RESULT = "识别结果"
GROUP_STAT = "类别统计"

WEIGHTS = str(ROOT / "runs" / "yolo11s_896" / "weights" / "best.pt")
IN_IMG_SIZE = 896              # 推理分辨率，必须和训练一致
FRAME_DELAY_MS = 30            # 每帧间隔（毫秒），越小越快但越吃 CPU


# ═══════════════════════════════════════════════════════════════════
#  视频/摄像头工作线程
#  为什么要开线程：cv2.VideoCapture.read() 会阻塞，
#  直接在主线程里 while 循环会让界面"未响应"
# ═══════════════════════════════════════════════════════════════════

class FrameWorker(QThread):
    """逐帧读取并检测，结果通过信号发回主线程"""

    # 定义两个信号：一个送图，一个送检测结果
    frame_ready = pyqtSignal(object)     # 传 QImage
    result_ready = pyqtSignal(list)      # 传 [(类名, 置信度, xyxy), ...]

    def __init__(self, source, model):
        super().__init__()
        self.source = source      # 0 = 摄像头，或视频文件路径
        self.model = model
        self.alive = True         # 用标志位停止，而不是 terminate()

    def run(self):
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            self.result_ready.emit([])
            return

        while self.alive:
            ok, frame = cap.read()
            if not ok:
                break

            dets = []
            if self.model is not None:
                res = self.model.predict(source=frame, imgsz=IN_IMG_SIZE,
                                         conf=0.25, verbose=False)[0]
                frame = res.plot()          # 框架直接把框画好
                if res.boxes is not None and len(res.boxes):
                    for i in range(len(res.boxes)):
                        dets.append((
                            self.model.names[int(res.boxes.cls[i])],
                            float(res.boxes.conf[i]),
                            res.boxes.xyxy[i].tolist(),
                        ))

            # BGR -> RGB，Qt 只吃 RGB
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            # QImage 不拷贝数据，必须把 rgb 传进去并指定每行字节数
            img = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
            self.frame_ready.emit(img)
            self.result_ready.emit(dets)

            self.msleep(FRAME_DELAY_MS)

        cap.release()

    def stop(self):
        self.alive = False


# ═══════════════════════════════════════════════════════════════════
#  主窗口
# ═══════════════════════════════════════════════════════════════════

class DroneDetectApp(QMainWindow):
    """检测主界面"""

    def __init__(self):
        super().__init__()

        self.model = None
        self.worker = None

        # ---- 外观：窗口本体 ----
        self.setWindowTitle(APP_TITLE)          # TODO 改标题
        self.resize(1180, 680)                  # TODO 改初始尺寸
        self.setStyleSheet(self.build_style())

        self.build_ui()

        # 启动时自动尝试加载默认权重
        self.try_load_default()

    # ---------------------------------------------------------------
    # 样式表 —— 只改颜色变量就能整体换肤
    # ---------------------------------------------------------------
    def build_style(self):
        return """
            QMainWindow, QWidget {{
                background: {bg};
                color: {fg};
                font-family: "{font}";
                font-size: {size}px;
            }}
            QGroupBox {{
                background: {panel};
                border: 1px solid {accent};
                border-radius: 8px;
                margin-top: 14px;
                padding-top: 8px;
                font-weight: bold;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 6px;
                color: {accent};
            }}
            QPushButton {{
                background: {panel};
                color: {fg};
                border: 1px solid {accent};
                border-radius: 6px;
                padding: 8px 10px;
            }}
            QPushButton:hover  {{ background: {accent}; color: #101018; }}
            QPushButton:pressed {{ background: {muted}; }}
            QPushButton:disabled {{ color: {muted}; border-color: {muted}; }}
            QListWidget {{
                background: {panel};
                border: 1px solid {muted};
                border-radius: 6px;
                padding: 4px;
            }}
            QLabel#preview {{
                background: #101018;
                border: 1px solid {muted};
                border-radius: 6px;
            }}
        """.format(bg=BG_COLOR, panel=PANEL_COLOR, fg=TEXT_COLOR,
                   accent=ACCENT_COLOR, muted=MUTED_COLOR,
                   font=FONT_FAMILY, size=FONT_SIZE)

    # ---------------------------------------------------------------
    # 界面搭建
    # TODO: 布局结构可以整个改（现在是「左预览右面板」两栏）
    # ---------------------------------------------------------------
    def build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)

        # 水平主布局：左边图像，右边控制面板
        outer = QHBoxLayout(root)
        outer.setContentsMargins(10, 10, 10, 10)
        outer.setSpacing(10)

        # ===== 左：图像预览 =====
        self.preview_label = QLabel("等待输入")
        self.preview_label.setObjectName("preview")     # 对应样式表里的 QLabel#preview
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setMinimumSize(640, 480)
        outer.addWidget(self.preview_label, stretch=3)

        # ===== 右：控制面板 =====
        side = QVBoxLayout()
        side.setSpacing(10)
        outer.addLayout(side, stretch=1)

        # ---- 操作按钮组 ----
        gb_ctrl = QGroupBox(GROUP_CTRL)
        lay_ctrl = QVBoxLayout(gb_ctrl)

        self.btn_load = QPushButton(BTN_LOAD)
        self.btn_image = QPushButton(BTN_IMAGE)
        self.btn_video = QPushButton(BTN_VIDEO)
        self.btn_camera = QPushButton(BTN_CAMERA)
        self.btn_stop = QPushButton(BTN_STOP)
        self.btn_stop.setEnabled(False)

        for b in (self.btn_load, self.btn_image, self.btn_video,
                  self.btn_camera, self.btn_stop):
            lay_ctrl.addWidget(b)

        self.btn_load.clicked.connect(self.load_model)
        self.btn_image.clicked.connect(self.open_image)
        self.btn_video.clicked.connect(self.open_video)
        self.btn_camera.clicked.connect(self.start_camera)
        self.btn_stop.clicked.connect(self.stop_worker)

        side.addWidget(gb_ctrl)

        # ---- 模型信息 ----
        self.info_label = QLabel("模型未加载")
        self.info_label.setWordWrap(True)
        self.info_label.setStyleSheet("color: %s;" % MUTED_COLOR)
        side.addWidget(self.info_label)

        # ---- 统计区 ----
        gb_stat = QGroupBox(GROUP_STAT)
        lay_stat = QVBoxLayout(gb_stat)
        self.stat_label = QLabel("—")
        self.stat_label.setWordWrap(True)
        lay_stat.addWidget(self.stat_label)
        side.addWidget(gb_stat)

        # ---- 结果列表 ----
        gb_res = QGroupBox(GROUP_RESULT)
        lay_res = QVBoxLayout(gb_res)
        self.result_list = QListWidget()
        lay_res.addWidget(self.result_list)
        side.addWidget(gb_res, stretch=1)

    # ---------------------------------------------------------------
    # 模型加载
    # ---------------------------------------------------------------
    def try_load_default(self):
        """启动时自动加载，失败不打扰用户"""
        if os.path.exists(WEIGHTS):
            try:
                self.model = YOLO(WEIGHTS)
                self.info_label.setText("已加载: %s" % os.path.basename(WEIGHTS))
            except Exception as e:
                self.info_label.setText("默认模型加载失败: %s" % e)

    def load_model(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择模型", "", "PyTorch 模型 (*.pt)")
        if not path:
            return
        try:
            self.model = YOLO(path)
            self.info_label.setText("已加载: %s" % os.path.basename(path))
        except Exception as e:
            QMessageBox.warning(self, "错误", "模型加载失败:\n%s" % e)

    # ---------------------------------------------------------------
    # 图片检测
    # ---------------------------------------------------------------
    def open_image(self):
        if not self._check_model():
            return
        self.stop_worker()

        path, _ = QFileDialog.getOpenFileName(
            self, "选择图片", "",
            "图片 (*.jpg *.jpeg *.png *.bmp);;所有文件 (*)")
        if not path:
            return

        img = cv2.imread(path)
        if img is None:
            QMessageBox.warning(self, "错误", "无法读取图片")
            return

        res = self.model.predict(source=img, imgsz=IN_IMG_SIZE,
                                 conf=0.25, verbose=False)[0]
        drawn = res.plot()

        dets = []
        if res.boxes is not None and len(res.boxes):
            for i in range(len(res.boxes)):
                dets.append((
                    self.model.names[int(res.boxes.cls[i])],
                    float(res.boxes.conf[i]),
                    res.boxes.xyxy[i].tolist(),
                ))

        self.refresh_results(dets)
        self.show_frame(self.cv_to_qimage(drawn))

    # ---------------------------------------------------------------
    # 视频 / 摄像头
    # ---------------------------------------------------------------
    def open_video(self):
        if not self._check_model():
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "选择视频", "", "视频 (*.mp4 *.avi *.mov);;所有文件 (*)")
        if path:
            self._begin_stream(path)

    def start_camera(self):
        if not self._check_model():
            return
        self._begin_stream(0)          # 0 = 默认摄像头

    def _begin_stream(self, source):
        self.stop_worker()
        self.worker = FrameWorker(source, self.model)
        self.worker.frame_ready.connect(self.show_frame)
        self.worker.result_ready.connect(self.refresh_results)
        self.worker.start()

        self.btn_stop.setEnabled(True)
        for b in (self.btn_image, self.btn_video, self.btn_camera):
            b.setEnabled(False)

    def stop_worker(self):
        if self.worker is not None:
            self.worker.stop()
            self.worker.wait(2000)     # 最多等 2 秒让它自然结束
            self.worker = None

        self.btn_stop.setEnabled(False)
        for b in (self.btn_image, self.btn_video, self.btn_camera):
            b.setEnabled(True)

    # ---------------------------------------------------------------
    # 显示与刷新
    # ---------------------------------------------------------------
    @staticmethod
    def cv_to_qimage(bgr):
        """OpenCV 的 BGR 图 -> Qt 的 QImage"""
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        return QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)

    def show_frame(self, qimg):
        # KeepAspectRatio 保证不变形
        pix = QPixmap.fromImage(qimg).scaled(
            self.preview_label.size(), Qt.KeepAspectRatio,
            Qt.SmoothTransformation)
        self.preview_label.setPixmap(pix)

    def refresh_results(self, dets):
        """刷新结果列表和类别统计"""
        self.result_list.clear()
        counts = {}
        for name, conf, _box in dets:
            self.result_list.addItem("%-18s %.2f" % (name, conf))
            counts[name] = counts.get(name, 0) + 1

        if not counts:
            self.stat_label.setText("未检出目标")
            return

        lines = ["共 %d 个目标" % len(dets), ""]
        for name, cnt in sorted(counts.items(), key=lambda x: -x[1]):
            lines.append("%-18s %d" % (name, cnt))
        self.stat_label.setText("\n".join(lines))

    def _check_model(self):
        if self.model is None:
            QMessageBox.information(self, "提示", "请先加载模型")
            return False
        return True

    # ---------------------------------------------------------------
    def closeEvent(self, event):
        """关窗口前先停线程，否则可能崩"""
        self.stop_worker()
        event.accept()


def main():
    app = QApplication(sys.argv)
    win = DroneDetectApp()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
