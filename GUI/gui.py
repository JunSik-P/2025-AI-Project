import sys, os, time, logging, random
from logging.handlers import RotatingFileHandler
from pathlib import Path
import cv2

import json
from dataclasses import dataclass
from collections import Counter, deque
from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QHBoxLayout, QVBoxLayout, QGraphicsDropShadowEffect,
    QLabel, QSplitter, QTableWidget, QTableWidgetItem, QAbstractItemView, QHeaderView,
    QSizePolicy, QMenu, QAction, QActionGroup, QInputDialog, QMessageBox, QDialog, 
    QLineEdit, QSpinBox, QComboBox, QFormLayout, QPlainTextEdit, QCheckBox
)
from PyQt5.QtGui import QIcon, QImage, QPixmap, QIcon, QImage, QPixmap, QColor, QBrush, QFont
from PyQt5.QtCore import Qt, QPoint, QRect, QTimer, QDateTime, QEvent, QSettings, QSize

APP_ORG = "2025-AI-Project"
APP_NAME = "StorageMonitorUI"
LOG_DIR = Path.home() / ".storage_monitor_ui"

# ===== 파일 경로 (필요시 수정/환경변수로 주입) =====
VIDEO_PATH = os.getenv("APP_VIDEO_PATH", r"C:\Users\User\Desktop\GUI\W6\aimodule\KakaoTalk_20251029_133011560.mp4")
RESULT_JSONL = os.getenv("APP_RESULT_JSONL", r"C:\Users\User\Desktop\GUI\W6\aimodule\park_clean_merged_result.jsonl")

# decisions 파일 경로 (= result 프리픽스에 맞춰 자동 생성)
RESULT_PREFIX = Path(RESULT_JSONL).name.replace("_result.jsonl", "").replace(".jsonl", "")
DECISIONS_JSONL = str(Path(RESULT_JSONL).with_name(f"{RESULT_PREFIX}_decisions.jsonl"))

# ===== 동시 인식 트리거 파라미터 =====
HOLD_MIN_S   = 1.5   # 충족 최소 지속
RESET_MIN_S  = 0.5   # 비충족 지속(재무장)
FACE_SIM_THR = 0.40  # 얼굴 유사도 임계

# -------- Logging setup --------
def setup_logging(debug=False):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(APP_NAME)
    logger.setLevel(logging.DEBUG if debug else logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")

    # Console
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.DEBUG if debug else logging.INFO)
    sh.setFormatter(fmt)
    # Avoid duplicate handlers if re-run in same session
    for h in list(logger.handlers):
        logger.removeHandler(h)
    logger.addHandler(sh)

    # Rotating file
    fh = RotatingFileHandler(LOG_DIR / "app.log", maxBytes=1_000_000, backupCount=3, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger

# Global logger (level adjusted later if user toggles)
DEBUG_BOOT = ("--debug" in sys.argv) or (os.getenv("APP_DEBUG") == "1")
logger = setup_logging(DEBUG_BOOT)
logger.info("Starting %s (debug=%s)", APP_NAME, DEBUG_BOOT)


# --- Draggable top bar (only empty area is draggable; buttons stay clickable) ---
class TopBar(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self._drag_active = False
        self._drag_pos = QPoint()
        self.setObjectName("topBar")
        self.setAttribute(Qt.WA_StyledBackground, True)

    def set_theme(self, colors, radius=15):
        self.setStyleSheet(f"""
            #topBar {{
                background-color: {colors['topbar']};
                border-top-left-radius: {radius}px;
                border-top-right-radius: {radius}px;
                border-bottom: 1px solid {colors['divider']}; /* your divider */
            }}
        """)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self.childAt(event.pos()) is None:
            self._drag_active = True
            self._drag_pos = event.globalPos() - self.window().frameGeometry().topLeft()
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._drag_active and (event.buttons() & Qt.LeftButton):
            self.window().move(event.globalPos() - self._drag_pos)
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self._drag_active:
            self._drag_active = False
            event.accept()
        else:
            super().mouseReleaseEvent(event)


# --- Aspect-ratio wrapper that sizes its child to a fixed ratio (e.g., 16:9) ---
class AspectRatioBox(QWidget):
    def __init__(self, ratio=(16, 9), child=None, parent=None):
        super().__init__(parent)
        self._rw, self._rh = ratio
        self._child = None
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        if child:
            self.set_child(child)

    def set_child(self, w):
        if self._child is not None:
            self._child.setParent(None)
        self._child = w
        if self._child is not None:
            self._child.setParent(self)
            self._child.show()
            self._layout_child()

    def set_ratio(self, rw, rh):
        self._rw, self._rh = rw, rh
        self._layout_child()

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._layout_child()

    def _layout_child(self):
        if not self._child:
            return
        W, H = self.width(), self.height()
        if self._rw == 0 or self._rh == 0:
            self._child.setGeometry(0, 0, W, H)
            return
        target_w = W
        target_h = int(target_w * self._rh / self._rw)
        if target_h > H:
            target_h = H
            target_w = int(target_h * self._rw / self._rh)
        x = (W - target_w) // 2
        y = (H - target_h) // 2
        self._child.setGeometry(x, y, target_w, target_h)

@dataclass
class FrameRec:
    t_s: float
    faces: list
    items: list

class ResultStream:
    def __init__(self, path: str):
        self.frames = []
        if not path or not Path(path).exists():
            logger.warning("Result JSONL not found: %s", path)
            self.idx = 0
            return
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    o = json.loads(line)
                except Exception:
                    continue
                t_ms = o.get("timestamp_ms")
                if t_ms is None:
                    continue
                self.frames.append(
                    FrameRec(
                        t_s=float(t_ms) / 1000.0,
                        faces=o.get("faces", []) or [],
                        items=o.get("items", []) or [],
                    )
                )
        self.frames.sort(key=lambda r: r.t_s)
        self.idx = 0
        logger.info("Loaded %d result frames from %s", len(self.frames), path)

    def reset_to_time(self, t: float):
        lo, hi = 0, len(self.frames)
        while lo < hi:
            mid = (lo + hi) // 2
            if self.frames[mid].t_s < t:
                lo = mid + 1
            else:
                hi = mid
        self.idx = max(0, lo - 1)

    def get_state_at(self, t: float):
        # (ok, face_name, item_class)
        while self.idx + 1 < len(self.frames) and self.frames[self.idx + 1].t_s <= t:
            self.idx += 1
        if not self.frames:
            return False, None, None
        fr = self.frames[self.idx]

        # 얼굴 대표(최고 similarity, Unknown 제외, 임계 이상)
        best = None
        for f in fr.faces:
            name = str(f.get("person_id", "") or "").strip()
            sim = float(f.get("similarity", 0.0))
            if name and name.lower() != "unknown" and sim >= FACE_SIM_THR:
                if best is None or sim > best[1]:
                    best = (name, sim)
        face_name = best[0] if best else None

        # 품목 대표(Unknown 제외 첫 후보)
        item_class = None
        for it in fr.items:
            cls = str(it.get("class_name", "") or "").strip()
            if cls and cls.lower() != "unknown":
                item_class = cls
                break

        ok = (face_name is not None) and (item_class is not None)
        return ok, face_name, item_class

class DecisionDialog(QDialog):
    def __init__(self, person: str, item: str, t_s: float, parent=None):
        super().__init__(parent)
        self.setWindowTitle("이벤트 분류")
        self.choice = None
        form = QFormLayout()
        form.addRow("시간(초)", QLabel(f"{t_s:.2f}"))
        form.addRow("이름", QLabel(person or "(미확정)"))
        form.addRow("품목", QLabel(item or "(미확정)"))
        btn_issue  = QPushButton("불출")
        btn_return = QPushButton("반납")
        btn_skip   = QPushButton("건너뛰기")
        btn_issue.clicked.connect(lambda: self._set_choice("불출"))
        btn_return.clicked.connect(lambda: self._set_choice("반납"))
        btn_skip.clicked.connect(self.reject)
        row = QHBoxLayout(); row.addWidget(btn_issue); row.addWidget(btn_return); row.addWidget(btn_skip)
        lay = QVBoxLayout(self); lay.addLayout(form); lay.addLayout(row)
    def _set_choice(self, c):
        self.choice = c; self.accept()

class VideoPlayer(QWidget):
    def __init__(self, video_path: str, title="입력 영상", aspect_ratio=(9, 16)):
        super().__init__()
        self.title = title
        self.aspect_ratio = aspect_ratio
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap or not self.cap.isOpened():
            raise RuntimeError(f"영상 열기 실패: {video_path}")
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 30.0)
        self.nframes = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        self.title_label = QLabel(self.title)
        self.view = QLabel("영상 로딩 중…"); self.view.setAlignment(Qt.AlignCenter); self.view.setMinimumSize(240, 135)
        self.fps_label = QLabel("FPS: --", self.view); self.fps_label.hide()

        self.ar = AspectRatioBox(ratio=aspect_ratio, child=self.view, parent=self)
        lay = QVBoxLayout(self); lay.setContentsMargins(0,0,0,0); lay.setSpacing(6)
        lay.addWidget(self.title_label); lay.addWidget(self.ar, 1)

        self._last_pixmap = None; self._show_fps = False; self._last_times = []
        self.playing = True; self.frame_idx = 0
        self.timer = QTimer(self); self.timer.timeout.connect(self._on_tick)
        self.timer.start(max(1, int(1000 / (self.fps or 30))))

    def sizeHint(self): return QSize(540, 960)
    def time_s(self):   return (self.frame_idx / self.fps) if self.fps > 0 else 0.0
    def set_aspect_ratio(self, rw, rh): self.aspect_ratio = (rw, rh); self.ar.set_ratio(rw, rh); self._render_scaled()
    def set_theme(self, colors):
        self.title_label.setStyleSheet(f"font-weight:600; color:{colors['text']}; padding:6px;")
        self.view.setStyleSheet(f"background:{colors['cam_bg']}; border:1px solid {colors['border']}; border-radius:8px;")
    def set_show_fps(self, enabled): self._show_fps = enabled; self.fps_label.setVisible(enabled); self._last_times.clear()
    def pause(self):  self.playing = False; self.timer.stop()
    def play(self):   self.playing = True;  self.timer.start(max(1, int(1000 / (self.fps or 30))))
    def restart(self): self.timer.stop(); self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0); self.frame_idx = 0; self.play()
    def _on_tick(self):
        if not self.playing: return
        if self.frame_idx >= self.nframes: self.pause(); return
        self.frame_idx += 1; self._read_and_show()
    def _read_and_show(self):
        ok, frame = self.cap.read()
        if not ok: self.pause(); return
        if self.aspect_ratio == (9,16) and frame.shape[1] > frame.shape[0]:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h,w,ch = rgb.shape
        qimg = QImage(rgb.data, w, h, w*ch, QImage.Format_RGB888)
        self._last_pixmap = QPixmap.fromImage(qimg); self._render_scaled()
        if self._show_fps:
            now = time.time(); self._last_times.append(now)
            while self._last_times and now - self._last_times[0] > 1.0: self._last_times.pop(0)
            self.fps_label.setText(f"FPS: {len(self._last_times):02d}")
    def _render_scaled(self):
        if self._last_pixmap is None: return
        self.view.setPixmap(self._last_pixmap.scaled(self.view.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
    def closeEvent(self, e):
        try: self.timer.stop(); self.cap.release()
        finally: super().closeEvent(e)

# --- Camera widget: resizes with window, image keeps aspect ratio; debug FPS overlay ---
class CameraFeed(QWidget):
    def __init__(self, source=0, title="Camera", aspect_ratio=(16, 9)):
        super().__init__()
        self.source = source
        self.title = title
        self.cap = None
        self._last_pixmap = None
        self._show_fps = False
        self._fps_label_visible = False
        self._last_times = []  # simple fps window
        self._log = logging.getLogger(f"{APP_NAME}.CameraFeed[{self.title}]")

        self.title_label = QLabel(title)

        self.view = QLabel()
        self.view.setAlignment(Qt.AlignCenter)
        self.view.setMinimumSize(240, 135)
        self.view.setScaledContents(False)

        # Small FPS overlay
        self.fps_label = QLabel("FPS: --", self.view)
        self.fps_label.setStyleSheet(
            "background: rgba(0,0,0,0.45); color: white; padding: 2px 6px; "
            "border-radius: 4px; font: 10px 'Monospace';"
        )
        self.fps_label.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.fps_label.move(6, 6)
        self.fps_label.hide()

        # wrap label in aspect-ratio box
        self.ar = AspectRatioBox(ratio=aspect_ratio, child=self.view, parent=self)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)
        lay.addWidget(self.title_label)
        lay.addWidget(self.ar, 1)

        self.view.installEventFilter(self)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._grab)
        self.start()

    # ---- Debug helpers ----
    def set_show_fps(self, enabled: bool):
        self._show_fps = enabled
        self.fps_label.setVisible(enabled)
        self._fps_label_visible = enabled
        self._last_times.clear()
        self._log.info("Show FPS overlay: %s", enabled)

    # ---- Theme & ratio ----
    def set_aspect_ratio(self, rw, rh):
        self.ar.set_ratio(rw, rh)
        self._render_scaled()

    def set_theme(self, colors):
        self.title_label.setStyleSheet(f"font-weight:600; color:{colors['text']}; padding:6px;")
        self.view.setStyleSheet(
            f"background:{colors['cam_bg']}; border:1px solid {colors['border']}; border-radius:8px;"
        )

    def eventFilter(self, obj, e):
        if obj is self.view and e.type() == QEvent.Resize:
            self._render_scaled()
            # keep overlay in corner
            self.fps_label.move(6, 6)
        return super().eventFilter(obj, e)

    # ---- Camera control ----
    def start(self):
        try:
            self.cap = cv2.VideoCapture(self.source)
            if not self.cap or not self.cap.isOpened():
                self.title_label.setText(f"{self.title} (응답없음)")
                self._log.error("Cannot open camera source: %r", self.source)
                return
            self._log.info("Camera started: %r", self.source)
            self.timer.start(33)  # ~30 FPS
        except Exception as e:
            self._log.exception("Failed to start camera: %s", e)

    def restart(self):
        self._log.info("Restarting camera...")
        self.stop()
        self.start()

    def stop(self):
        try:
            self.timer.stop()
            if self.cap:
                self.cap.release()
                self.cap = None
            self._log.info("Camera stopped.")
        except Exception as e:
            self._log.exception("Error during camera stop: %s", e)

    # ---- Frame pipeline ----
    def _grab(self):
        if not self.cap:
            return
        ok, frame = self.cap.read()
        if not ok:
            # Avoid log spam — log occasionally
            self._log.debug("Frame grab failed.")
            return

        try:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame.shape
            qimg = QImage(frame.data, w, h, ch * w, QImage.Format_RGB888)
            self._last_pixmap = QPixmap.fromImage(qimg)
            self._render_scaled()

            # update FPS
            if self._show_fps:
                now = time.time()
                self._last_times.append(now)
                # keep last ~1s window
                while self._last_times and now - self._last_times[0] > 1.0:
                    self._last_times.pop(0)
                fps = len(self._last_times)  # approx frames per ~1s
                self.fps_label.setText(f"FPS: {fps:02d}")
        except Exception as e:
            self._log.exception("Error processing frame: %s", e)

    def _render_scaled(self):
        if self._last_pixmap is None:
            return
        target = self.view.size()
        scaled = self._last_pixmap.scaled(target, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.view.setPixmap(scaled)

    def closeEvent(self, e):
        self.stop()
        super().closeEvent(e)


class DebugConsole(QDialog):
    def __init__(self, host_window: 'FramelessWindow'):
        super().__init__(host_window)
        self.host = host_window
        self.setWindowTitle("Debug Console")
        self.setModal(False)
        self.setMinimumWidth(420)

        # --- Inputs ---
        self.user_edit = QLineEdit("사용자")
        self.action_combo = QComboBox()
        self.action_combo.addItems(["반납", "불출", "경고!"])
        self.item_edit = QLineEdit("ㅋ")
        self.count_spin = QSpinBox(); self.count_spin.setRange(1, 999); self.count_spin.setValue(1)

        self.note_edit = QLineEdit("(대충 아주 수상한 재고 불일치가 있다는 내용)")

        self.chk_log_also = QCheckBox("Also add to event log when pushing a notification")
        self.chk_log_also.setChecked(True)

        form = QFormLayout()
        form.addRow("이름:", self.user_edit)
        form.addRow("반납/불출:", self.action_combo)
        form.addRow("품목:", self.item_edit)
        form.addRow("수량:", self.count_spin)
        form.addRow("알림 문구:", self.note_edit)
        form.addRow("", self.chk_log_also)

        # --- Buttons ---
        self.btn_add = QPushButton("로그 추가")
        self.btn_add10 = QPushButton("랜덤으로 10개 추가")
        self.btn_notify = QPushButton("알림 테스트")

        btn_row = QHBoxLayout()
        btn_row.addWidget(self.btn_add)
        btn_row.addWidget(self.btn_add10)
        btn_row.addWidget(self.btn_notify)

        # --- Console output (optional) ---
        self.out = QPlainTextEdit(); self.out.setReadOnly(True); self.out.setMaximumBlockCount(500)

        lay = QVBoxLayout(self)
        lay.addLayout(form)
        lay.addLayout(btn_row)
        lay.addWidget(self.out)

        # Wire
        self.btn_add.clicked.connect(self._add_one)
        self.btn_add10.clicked.connect(self._add_ten)
        self.btn_notify.clicked.connect(self._push_notif)

        # Theme
        self.apply_theme(self.host.themes[self.host.current_theme])

    def apply_theme(self, colors: dict):
        self.setStyleSheet(f"""
            QDialog {{ background: {colors['bg']}; color: {colors['text']}; }}
            QLineEdit, QPlainTextEdit, QComboBox, QSpinBox {{
                background: {colors['cam_bg']}; color: {colors['text']};
                border: 1px solid {colors['border']}; border-radius: 6px; padding: 4px;
            }}
            QPushButton {{
                border: none; background: transparent; padding: 6px 10px; border-radius: 6px;
            }}
            QPushButton:hover {{ background: {colors['min_hover']}; }}
            QPushButton:pressed {{ background: {colors['min_press']}; }}
        """)

    def _add_one(self):
        u = self.user_edit.text().strip() or "사용자"
        a = self.action_combo.currentText()
        i = self.item_edit.text().strip() or "품목"
        c = int(self.count_spin.value())
        self.host.add_log(u, a, i, c)
        self._log_out(f"add_log({u!r}, {a!r}, {i!r}, {c})")

    def _add_ten(self):
        users = ["김춘식", "신창섭", "김아무개", "홍길동", "최범수"]
        items = ["탄알집", "방탄모", "방탄판", "수통", "방독면", "군화(육면)", "군화(은면)"]
        actions = ["반납", "불출"]
        for _ in range(10):
            u = random.choice(users)
            a = random.choice(actions)
            i = random.choice(items)
            c = random.randint(1, 50)
            self.host.add_log(u, a, i, c)
        self._log_out("무작위 로그 10개를 추가했습니다.")

    def _push_notif(self):
        msg = self.note_edit.text().strip() or "(대충 아주 수상한 재고 불일치가 있다는 내용)"
        self.host.push_notification(msg)
        self._log_out(f"push_notification({msg!r})")
        if self.chk_log_also.isChecked():
            self.host.add_log("SYSTEM", "Alert", msg, 1)

    def _log_out(self, text):
        self.out.appendPlainText(text)


class InventoryDialog(QDialog):
    def __init__(self, host_window: 'FramelessWindow'):
        super().__init__(host_window)
        self.host = host_window
        self.setWindowTitle("재고 현황")
        self.setModal(False)
        self.setMinimumWidth(360)

        self.table = QTableWidget(0, 2, self)
        self.table.setHorizontalHeaderLabels(["품목", "수량"])
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setAlternatingRowColors(True)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.addWidget(self.table)

        self.apply_theme(self.host.themes[self.host.current_theme])

    def apply_theme(self, colors: dict):
        # match your table style
        self.setStyleSheet(f"""
            QDialog {{ background: {colors['bg']}; color: {colors['text']}; }}
            QTableWidget {{
                border: 1px solid {colors['border']};
                border-radius: 8px;
                background: {colors['bg']};
                alternate-background-color: {colors['alt_row']};
                gridline-color: {colors['grid']};
                color: {colors['text']};
            }}
            QHeaderView::section {{
                background: {colors['header_bg']};
                color: {colors['text']};
                border: none;
                border-right: 1px solid {colors['border']};
                padding: 6px;
            }}
        """)

    def set_counts(self, counts: dict):
        items = sorted(counts.items(), key=lambda kv: (kv[0] or "").lower())
        self.table.setRowCount(len(items))
        for r, (name, cnt) in enumerate(items):
            name_item = QTableWidgetItem(str(name))
            cnt_item = QTableWidgetItem(str(int(cnt)))
            cnt_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self.table.setItem(r, 0, name_item)
            self.table.setItem(r, 1, cnt_item)


# --- Aspect ratio helpers ---
def parse_ratio_text(s, fallback=(16, 9)):
    try:
        s = str(s).strip().lower().replace('x', ':').replace(',', ':').replace(' ', '')
        a, b = s.split(':')
        a, b = int(a), int(b)
        return (a, b) if a > 0 and b > 0 else fallback
    except Exception:
        return fallback

def ratio_to_text(tup):
    return f"{int(tup[0])}:{int(tup[1])}"


class FramelessWindow(QWidget):
    RESIZE_MARGIN = 8
    MIN_W = 800
    MIN_H = 480

    def __init__(self, face_source=0, ratio=(16, 9)):
        super().__init__()
        self._log = logging.getLogger(f"{APP_NAME}.Window")
        self.setWindowFlags(Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)

        # QSettings
        self.settings = QSettings(APP_ORG, APP_NAME)

        # Defaults
        self.resize(1000, 700)
        self.setMinimumSize(self.MIN_W, self.MIN_H)

        self._normal_geometry = None
        self.current_theme = 'light'

        # Theme palettes
        self.themes = {
            'light': {
                'bg': 'white',
                'text': '#202124',
                'topbar': '#f2f2f2',
                'divider': 'rgba(0,0,0,0.08)',
                'border': '#e7e7e7',
                'cam_bg': '#fafafa',
                'alt_row': '#fafafa',
                'header_bg': '#f7f7f7',
                'grid': '#e7e7e7',
                'min_hover': 'rgba(0,0,0,0.06)',
                'min_press': 'rgba(0,0,0,0.12)',
                'selection': 'rgba(0,0,0,0.12)',
            },
            'dark': {
                'bg': '#1e1e1e',
                'text': '#e6e6e6',
                'topbar': '#2a2a2a',
                'divider': 'rgba(255,255,255,0.12)',
                'border': '#3a3a3a',
                'cam_bg': '#111111',
                'alt_row': '#1a1a1a',
                'header_bg': '#222222',
                'grid': '#3a3a3a',
                'min_hover': 'rgba(255,255,255,0.08)',
                'min_press': 'rgba(255,255,255,0.16)',
                'selection': 'rgba(255,255,255,0.16)',
            }
        }

        # --- Inner container ---
        self.container = QWidget(self)
        self.container.setObjectName("container")
        self.container.setAttribute(Qt.WA_StyledBackground, True)

        self.outer_layout = QVBoxLayout(self)
        self.outer_layout.setContentsMargins(10, 10, 10, 10)
        self.outer_layout.addWidget(self.container)

        self.shadow = QGraphicsDropShadowEffect(self)
        self.shadow.setBlurRadius(24)
        self.shadow.setOffset(0, 6)
        self.shadow.setColor(Qt.black)
        self.container.setGraphicsEffect(self.shadow)

        # --- Top bar buttons ---
        # Left: Settings (menu)
        self.settings_button = QPushButton("")
        self.settings_button.setObjectName("settingsBtn")
        self.settings_button.setIcon(QIcon("settings_button.png"))
        self.settings_button.setCursor(Qt.PointingHandCursor)

        # --- Settings menu (top-level) ---
        self.settings_menu = QMenu(self.settings_button)

        # Theme submenu (radio list like Debug items)
        self.theme_menu = QMenu("테마", self.settings_menu)
        self.action_light = QAction("밝은 테마", self, checkable=True)
        self.action_dark = QAction("어두운 테마", self, checkable=True)
        g_theme = QActionGroup(self); g_theme.setExclusive(True)
        g_theme.addAction(self.action_light); g_theme.addAction(self.action_dark)
        self.theme_menu.addAction(self.action_light)
        self.theme_menu.addAction(self.action_dark)

        # Aspect Ratio submenu (presets + Custom…)
        self.ratio_menu = QMenu("카메라 비율", self.settings_menu)
        self.action_ratio_169 = QAction("16:9", self, checkable=True)
        self.action_ratio_43  = QAction("4:3",  self, checkable=True)
        self.action_ratio_11  = QAction("1:1",  self, checkable=True)
        self.action_ratio_219 = QAction("21:9", self, checkable=True)
        self.action_ratio_custom = QAction("상세 비율…", self)
        g_ratio = QActionGroup(self); g_ratio.setExclusive(True)
        for a in (self.action_ratio_169, self.action_ratio_43, self.action_ratio_11, self.action_ratio_219):
            g_ratio.addAction(a); self.ratio_menu.addAction(a)
        self.ratio_menu.addSeparator()
        self.ratio_menu.addAction(self.action_ratio_custom)

        # Debug submenu
        self.debug_menu = QMenu("Debug", self.settings_menu)
        self.action_debug_logs = QAction("Enable debug logging", self, checkable=True)
        self.action_show_fps  = QAction("Show FPS overlay", self, checkable=True)
        self.action_restart_cams = QAction("Restart video", self)  # text updated (single camera)
        self.action_dump_settings = QAction("Dump settings to log", self)
        self.action_open_debug_console = QAction("Open debug console…", self)
        # Inventory debug tools
        self.action_rebuild_counts = QAction("Rebuild inventory counts from log", self)
        self.action_clear_counts   = QAction("Clear all inventory counts", self)
        self.action_seed_counts    = QAction("Seed counts manually…", self)

        self.debug_menu.addAction(self.action_debug_logs)
        self.debug_menu.addAction(self.action_show_fps)
        self.debug_menu.addSeparator()
        self.debug_menu.addAction(self.action_restart_cams)
        self.debug_menu.addAction(self.action_dump_settings)
        self.debug_menu.addSeparator()
        self.debug_menu.addAction(self.action_open_debug_console)
        self.debug_menu.addSeparator()
        self.debug_menu.addAction(self.action_rebuild_counts)
        self.debug_menu.addAction(self.action_clear_counts)
        self.debug_menu.addAction(self.action_seed_counts)

        # Assemble the Settings menu on the button
        self.settings_menu.addMenu(self.theme_menu)
        self.settings_menu.addMenu(self.ratio_menu)
        self.settings_menu.addMenu(self.debug_menu)
        self.settings_button.setMenu(self.settings_menu)

        # Wire actions
        self.action_light.triggered.connect(lambda: self.apply_theme('light'))
        self.action_dark.triggered.connect(lambda: self.apply_theme('dark'))

        # Aspect-ratio actions
        self.action_ratio_169.triggered.connect(lambda: self._apply_ratio((16, 9)))
        self.action_ratio_43.triggered.connect( lambda: self._apply_ratio((4, 3)))
        self.action_ratio_11.triggered.connect( lambda: self._apply_ratio((1, 1)))
        self.action_ratio_219.triggered.connect(lambda: self._apply_ratio((21, 9)))
        self.action_ratio_custom.triggered.connect(self._on_custom_ratio)

        # Debug actions (reuse your existing handlers)
        self.action_debug_logs.toggled.connect(self._on_toggle_debug)
        self.action_show_fps.toggled.connect(self._on_toggle_fps)
        self.action_restart_cams.setText("Restart video")
        self.action_restart_cams.triggered.disconnect()
        self.action_restart_cams.triggered.connect(self._on_restart_video)
        self.action_dump_settings.triggered.connect(self._on_dump_settings)
        self.action_open_debug_console.triggered.connect(self._open_debug_console)
        self._debug_console = None

        self.action_rebuild_counts.triggered.connect(self._rebuild_counts_from_log)
        self.action_clear_counts.triggered.connect(self._clear_counts)
        self.action_seed_counts.triggered.connect(self._seed_counts_prompt)

        # Right: Min/Max/Close
        self.min_button = QPushButton("")
        self.min_button.setObjectName("minBtn")
        self.min_button.setIcon(QIcon("minimize_button.png"))
        self.min_button.clicked.connect(self.showMinimized)
        self.min_button.setCursor(Qt.PointingHandCursor)

        self.fullsize_button = QPushButton("")
        self.fullsize_button.setObjectName("fullBtn")
        self.fullsize_button.setIcon(QIcon("fullsize_button.png"))
        self.fullsize_button.clicked.connect(self.toggle_fullscreen)
        self.fullsize_button.setCursor(Qt.PointingHandCursor)

        self.close_button = QPushButton("")
        self.close_button.setObjectName("closeBtn")
        self.close_button.setIcon(QIcon("close_button.png"))
        self.close_button.clicked.connect(self.close)
        self.close_button.setCursor(Qt.PointingHandCursor)
        
        # Notifications button (left side, next to Settings)
        self.notifications_button = QPushButton("")
        self.notifications_button.setObjectName("notifBtn")
        self.notifications_button.setIcon(QIcon("notifications_button.png"))  # provide an icon file
        self.notifications_button.setCursor(Qt.PointingHandCursor)

        # Notifications menu
        self.notifications_menu = QMenu(self.notifications_button)
        self._notif_title = QAction("알림 목록", self)
        self._notif_title.setEnabled(False)
        self._notif_separator = self.notifications_menu.addSeparator()  # anchor to insert above
        self.notifications_menu.insertAction(self._notif_separator, self._notif_title)

        self.action_notif_mark_read = QAction("전부 읽음처리", self)
        self.action_notif_clear = QAction("비우기", self)
        self.notifications_menu.addSeparator()
        self.notifications_menu.addAction(self.action_notif_mark_read)
        self.notifications_menu.addAction(self.action_notif_clear)

        self.notifications_button.setMenu(self.notifications_menu)

        # Badge (small red dot in bottom-right of the button)
        self._notif_badge = QLabel(self.notifications_button)
        self._notif_badge.setFixedSize(10, 10)
        self._notif_badge.setStyleSheet(
            "background:#e81123; border:1px solid white; border-radius:5px;"
        )
        self._notif_badge.hide()
        self.unread_notif_count = 0

        # Wire notification handlers
        self.notifications_menu.aboutToShow.connect(self._on_notif_menu_opened)
        self.action_notif_mark_read.triggered.connect(self._on_mark_all_read)
        self.action_notif_clear.triggered.connect(self._on_clear_notifications)

        # Inventory table button (next to Notifications)
        self.inventory_button = QPushButton("")
        self.inventory_button.setObjectName("invBtn")
        self.inventory_button.setIcon(QIcon("table_button.png"))   # supply an icon file
        self.inventory_button.setCursor(Qt.PointingHandCursor)
        self.inventory_button.clicked.connect(self._open_inventory)

        self._inventory_dialog = None
        self.item_counts = {}  # item -> count (int)

        # --- Top bar (draggable in empty area) ---
        self.top_bar_widget = TopBar(self)
        self.top_bar_layout = QHBoxLayout(self.top_bar_widget)
        self.top_bar_layout.setContentsMargins(8, 6, 6, 6)
        self.top_bar_layout.addWidget(self.settings_button)
        self.top_bar_layout.addWidget(self.notifications_button)
        self.top_bar_layout.addWidget(self.inventory_button)
        self.top_bar_layout.addStretch()
        self.top_bar_layout.addWidget(self.min_button)
        self.top_bar_layout.addWidget(self.fullsize_button)
        self.top_bar_layout.addWidget(self.close_button)

        # --- Content: Single Camera + Log ---
        self.player = VideoPlayer(VIDEO_PATH, title="입력 영상", aspect_ratio=ratio)

        cameras_col = QWidget()
        cam_layout = QVBoxLayout(cameras_col)
        cam_layout.setContentsMargins(8, 8, 8, 8)
        cam_layout.setSpacing(8)
        cam_layout.addWidget(self.player, 1)
        cameras_col.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.log_table = QTableWidget(0, 5)
        self.log_table.setObjectName("logBox")
        self.log_table.setHorizontalHeaderLabels(["시간", "이름", "반납/불출", "품목", "수량"])
        self.log_table.verticalHeader().setVisible(False)
        self.log_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.log_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.log_table.setAlternatingRowColors(True)
        self.log_table.setShowGrid(True)
        self.log_table.setGridStyle(Qt.SolidLine)

        header = self.log_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Stretch)
        header.setHighlightSections(False)
        header.setSectionResizeMode(4, QHeaderView.ResizeToContents)

        # Splitter
        self.splitter = QSplitter(Qt.Horizontal)
        self.splitter.addWidget(cameras_col)
        self.splitter.addWidget(self.log_table)
        self.splitter.setStretchFactor(0, 2)
        self.splitter.setStretchFactor(1, 1)
        self.splitter.splitterMoved.connect(self._save_splitter_sizes)

        main_layout = QVBoxLayout(self.container)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        main_layout.addWidget(self.top_bar_widget)
        main_layout.addWidget(self.splitter, 1)

        # Resize/drag state
        self._resizing = False
        self._resize_region = None
        self._resize_start_geo = QRect()
        self._resize_start_mouse = QPoint()

        self.setMouseTracking(True)
        self.container.setMouseTracking(True)
        self.top_bar_widget.setMouseTracking(True)

        self._update_button_sizes()
        self._update_topbar_height()
        self._update_topbar_spacing()

        # Load theme & debug prefs
        saved_theme = self.settings.value("theme", "light")
        self.apply_theme(saved_theme)

        debug_on = self.settings.value("debugEnabled", DEBUG_BOOT, type=bool)
        self.action_debug_logs.setChecked(bool(debug_on))
        self._on_toggle_debug(bool(debug_on))  # apply handler levels

        show_fps = self.settings.value("showFps", False, type=bool)
        self.action_show_fps.setChecked(bool(show_fps))
        self._on_toggle_fps(bool(show_fps))

        # Restore geometry/splitter
        self._restore_geometry_and_splitter()

        # Load saved aspect ratio (defaults to 16:9) and update menu checks
        saved_ratio_text = self.settings.value("ratio", "16:9")
        self._apply_ratio(parse_ratio_text(saved_ratio_text), update_menu_checks=True)

        # Log environment info
        try:
            import PyQt5
            self._log.info("Environment: PyQt5=%s, OpenCV=%s", PyQt5.QtCore.QT_VERSION_STR, cv2.__version__)
        except Exception:
            pass
    
        # JSONL 스트림 준비
        self.stream = ResultStream(RESULT_JSONL)

        # 상태 머신 변수
        self.hold_acc_s = 0.0
        self.reset_acc_s = 0.0
        self.armed = True
        self.last_t = 0.0
        self.window_names = deque(maxlen=90)
        self.window_items = deque(maxlen=90)

        # 30ms 주기 UI 타이머
        self.ui_timer = QTimer(self)
        self.ui_timer.timeout.connect(self._on_tick)
        self.ui_timer.start(30)

    def _on_tick(self):
        t = self.player.time_s()
        dt = max(0.0, t - self.last_t)
        self.last_t = t
        ok, name, item = self.stream.get_state_at(t)

        if ok:
            self.hold_acc_s += dt
            self.reset_acc_s = 0.0
            if name: self.window_names.append(name)
            if item: self.window_items.append(item)
            if self.armed and self.hold_acc_s >= HOLD_MIN_S:
                rep_name = self._mode(self.window_names) or (name or "Unknown")
                rep_item = self._mode(self.window_items) or (item or "Unknown")
                logger.info("Trigger @ %.2fs  name=%s  item=%s", t, rep_name, rep_item)
                self._trigger_modal(t, rep_name, rep_item)
        else:
            self.reset_acc_s += dt
            if self.reset_acc_s >= RESET_MIN_S:
                self.armed = True
                self.hold_acc_s = 0.0
                self.window_names.clear(); self.window_items.clear()

    def _mode(self, dq):
        if not dq: return None
        return Counter(dq).most_common(1)[0][0]

    def _trigger_modal(self, t_s: float, person: str, item: str):
        self.player.pause()
        self.armed = False
        dlg = DecisionDialog(person, item, t_s, parent=self)
        if dlg.exec_() == QDialog.Accepted and dlg.choice:
            self._append_decision(t_s, person, item, dlg.choice)
        self.player.play()

    def _format_ts_text(self, t_s: float) -> str:
        # (요구사항) 로그에 영상 기준 타임스탬프 반영
        ms = int(round(t_s * 1000))
        s, ms = divmod(ms, 1000)
        m, s = divmod(s, 60)
        h, m = divmod(m, 60)
        return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"

    def _append_decision(self, t_s: float, person: str, item: str, action: str):
        # decisions.jsonl에 기록
        rec = {
            "timestamp_s": round(t_s, 3),
            "person_name": person,
            "item_class": item,
            "action": action,
        }
        try:
            with open(DECISIONS_JSONL, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except Exception as e:
            QMessageBox.critical(self, "쓰기 오류", f"decisions 저장 실패: {e}")
            return

        # 로그 테이블: 현재 시각으로 표시
        self.add_log(person, action, item, count=1)

    def _on_restart_video(self):
        self._log.info("Restarting video by user request.")
        self.player.restart()
        # 상태 머신 초기화
        self.hold_acc_s = 0.0; self.reset_acc_s = 0.0; self.armed = True; self.last_t = 0.0
        self.window_names.clear(); self.window_items.clear()
        if hasattr(self, "stream"): self.stream.reset_to_time(0.0)

    def _apply_edge_to_edge(self):
        edge = self.isMaximized()  # treat maximized as fullscreen
        radius = 0 if edge else 15

        # margins
        if edge:
            self.outer_layout.setContentsMargins(0, 0, 0, 0)
        else:
            self.outer_layout.setContentsMargins(10, 10, 10, 10)

        # shadow: DO NOT detach/attach; just enable/disable + adjust params
        if edge:
            self.shadow.setEnabled(False)
            self.shadow.setBlurRadius(0)
            self.shadow.setOffset(0, 0)
        else:
            # enable shadow a tick later to avoid DWM hiccups when restoring
            def _enable_shadow():
                self.shadow.setBlurRadius(24)
                self.shadow.setOffset(0, 6)
                self.shadow.setEnabled(True)
            QTimer.singleShot(0, _enable_shadow)

        # rounded corners via stylesheet
        c = self.themes[self.current_theme]
        self.container.setStyleSheet(self._container_styles(c, radius))
        self.top_bar_widget.set_theme(c, radius)

    def _position_notif_badge(self):
        """Place the red badge at the bottom-right of the notifications button."""
        if not hasattr(self, "_notif_badge"):
            return
        btn = self.notifications_button
        bsz = self._notif_badge.size()
        # Slight positive offsets so it sits just inside the corner
        x = max(0, btn.width() - bsz.width() + 2)
        y = max(0, btn.height() - bsz.height() + 2)
        self._notif_badge.move(x, y)

    def _set_unread_badge(self, count: int):
        self.unread_notif_count = max(0, int(count))
        self._notif_badge.setVisible(self.unread_notif_count > 0)
        self._position_notif_badge()

    def push_notification(self, text: str):
        """Public API to add a notification and show badge."""
        ts = QDateTime.currentDateTime().toString("yyyy-MM-dd HH:mm:ss")
        act = QAction(f"[{ts}] {text}", self)
        act.setEnabled(False)
        # Insert above the anchor separator (so newest appear on top)
        self.notifications_menu.insertAction(self._notif_separator, act)
        self._set_unread_badge(self.unread_notif_count + 1)
        self._log.warning("Notification: %s", text)

    def _on_notif_menu_opened(self):
        # Opening the menu marks everything as 'seen'
        if self.unread_notif_count:
            self._set_unread_badge(0)

    def _on_mark_all_read(self):
        self._set_unread_badge(0)

    def _on_clear_notifications(self):
        # Remove all dynamic actions above the separator (keep title + separator + footer actions)
        for a in list(self.notifications_menu.actions()):
            if a in (self._notif_title, self._notif_separator,
                     self.action_notif_mark_read, self.action_notif_clear):
                continue
            self.notifications_menu.removeAction(a)
        self._set_unread_badge(0)

    def _open_debug_console(self):
        if self._debug_console is None:
            self._debug_console = DebugConsole(self)
        # sync theme each time before showing
        self._debug_console.apply_theme(self.themes[self.current_theme])
        self._debug_console.show()
        self._debug_console.raise_()
        self._debug_console.activateWindow()

    def _open_inventory(self):
        if self._inventory_dialog is None:
            self._inventory_dialog = InventoryDialog(self)
        # theme + data
        self._inventory_dialog.apply_theme(self.themes[self.current_theme])
        self._inventory_dialog.set_counts(self.item_counts)
        self._inventory_dialog.show()
        self._inventory_dialog.raise_()
        self._inventory_dialog.activateWindow()

    def _update_item_count(self, item_name: str, delta: int):
        name = (str(item_name) or "").strip()
        if not name:
            return
        self.item_counts[name] = int(self.item_counts.get(name, 0)) + int(delta)
        # live-refresh if dialog is open
        if self._inventory_dialog and self._inventory_dialog.isVisible():
            self._inventory_dialog.set_counts(self.item_counts)

    def _delta_for_action(self, action_text: str, count: int) -> int:
        a = (str(action_text) or "").strip()
        if a == "반납":      # deposit/return
            return +int(count)
        if a == "불출":      # withdraw
            return -int(count)
        # '경고!' or any other action does not change stock
        return 0
    
    def _refresh_inventory_dialog(self):
        if hasattr(self, "_inventory_dialog") and self._inventory_dialog and self._inventory_dialog.isVisible():
            self._inventory_dialog.set_counts(self.item_counts)
    
    def _rebuild_counts_from_log(self):
        counts = {}
        rows = self.log_table.rowCount()
        for r in range(rows):
            a_item = self.log_table.item(r, 2)
            i_item = self.log_table.item(r, 3)
            c_item = self.log_table.item(r, 4)
            action_txt = a_item.text() if a_item else ""
            item_txt   = i_item.text() if i_item else ""
            try:
                cnt = int(c_item.text()) if c_item else 0
            except Exception:
                cnt = 0
            delta = self._delta_for_action(action_txt, cnt)  # uses your existing helper
            if item_txt and delta != 0:
                counts[item_txt] = int(counts.get(item_txt, 0)) + int(delta)
    
        self.item_counts = counts
        self._refresh_inventory_dialog()
        QMessageBox.information(self, "Inventory", f"Rebuilt counts from log ({len(counts)} items).")
    
    def _clear_counts(self):
        self.item_counts.clear()
        self._refresh_inventory_dialog()
        QMessageBox.information(self, "Inventory", "All inventory counts cleared.")
    
    def _seed_counts_prompt(self):
        # Let user paste something like:
        #   Box-A=10, Box-B: 5
        #   Crate-12 7
        text, ok = QInputDialog.getMultiLineText(
            self, "Seed counts manually…",
            "Enter items and counts (examples):\n"
            "  Box-A=10, Box-B:5\n"
            "  Crate-12 7\n"
            "Lines or commas are fine. Only integers are accepted.",
            ""
        )
        if not ok or not text.strip():
            return
    
        updated = 0
        for chunk in [p.strip() for p in text.replace("\n", ",").split(",")]:
            if not chunk:
                continue
            name, val = None, None
            # try separators in order: '=', ':', whitespace
            if "=" in chunk:
                name, val = chunk.split("=", 1)
            elif ":" in chunk:
                name, val = chunk.split(":", 1)
            else:
                parts = chunk.split()
                if len(parts) >= 2:
                    name, val = " ".join(parts[:-1]), parts[-1]
            if name is None or val is None:
                continue
            name = name.strip()
            try:
                cnt = int(val.strip())
            except Exception:
                continue
            if name:
                self.item_counts[name] = cnt
                updated += 1
    
        self._refresh_inventory_dialog()
        QMessageBox.information(self, "Inventory", f"Seeded {updated} item(s).")



    # ===== Persistence helpers =====
    def _restore_geometry_and_splitter(self):
        geo = self.settings.value("geometry")
        if geo is not None:
            ok = self.restoreGeometry(geo)
            self._log.info("Restore geometry: %s", ok)

        sizes = self.settings.value("splitterSizes")
        if sizes:
            try:
                sizes = [int(x) for x in list(sizes)]
                if len(sizes) == 2 and sum(sizes) > 0:
                    self.splitter.setSizes(sizes)
                    self._log.info("Restore splitter sizes: %s", sizes)
            except Exception:
                self._log.exception("Failed to restore splitter sizes.")

    def _save_splitter_sizes(self, *_):
        sizes = self.splitter.sizes()
        self.settings.setValue("splitterSizes", sizes)
        self._log.debug("Splitter moved, sizes=%s", sizes)

    # ===== Theme application with persistence =====
    def _container_styles(self, c, radius=15):
        return f"""
            #container {{
                background-color: {c['bg']};
                border-radius: {radius}px;
                color: {c['text']};
            }}

            QPushButton {{
                border: none;
                background: transparent;
                border-radius: 6px;
                padding: 0 2px;
            }}

            #settingsBtn:hover, #minBtn:hover, #fullBtn:hover, #notifBtn:hover, #invBtn:hover {{
                background-color: {c['min_hover']};
            }}
            #settingsBtn:pressed, #minBtn:pressed, #fullBtn:pressed, #notifBtn:pressed, #invBtn:pressed {{
                background-color: {c['min_press']};
            }}

            #closeBtn:hover {{ background-color: #e81123; }}
            #closeBtn:pressed {{ background-color: #c50f1f; }}

            #logBox {{
                border: 1px solid {c['border']};
                border-radius: 8px;
                background: {c['bg']};
                alternate-background-color: {c['alt_row']};
                gridline-color: {c['grid']};
                color: {c['text']};
            }}
            #logBox::item:selected {{
                background: {c['selection']};
            }}
            #logBox QHeaderView::section {{
                background: {c['header_bg']};
                color: {c['text']};
                border: none;
                border-right: 1px solid {c['border']};
                padding: 6px;
            }}
            #logBox QTableCornerButton::section {{
                background: {c['header_bg']};
                border: none;
            }}
        """

    def apply_theme(self, name):
        if name not in self.themes:
            return
        self.current_theme = name
        c = self.themes[name]

        # Persist & checkmarks
        self.settings.setValue("theme", name)
        self.action_light.setChecked(name == 'light')
        self.action_dark.setChecked(name == 'dark')

        # Apply styles
        radius = 0 if self.isMaximized() else 15
        self.container.setStyleSheet(self._container_styles(c, radius))
        self.top_bar_widget.set_theme(c, radius)
        self.player.set_theme(c)
        self._log.info("Applied theme: %s", name)

        # Keep debug console and inventory dialog themed too
        if hasattr(self, "_debug_console") and self._debug_console:
            self._debug_console.apply_theme(c)
        if hasattr(self, "_inventory_dialog") and self._inventory_dialog:
            self._inventory_dialog.apply_theme(c)

        # Repaint existing alert rows to match theme
        for r in range(self.log_table.rowCount()):
            a = self.log_table.item(r, 2)
            i = self.log_table.item(r, 3)
            action_txt = a.text() if a else ""
            item_txt   = i.text() if i else ""
            self._paint_log_row(r, self._is_alert_entry("", action_txt, item_txt))


    def _apply_ratio(self, ratio_tuple, update_menu_checks=False):
        rw, rh = ratio_tuple
        # apply to single camera
        self.player.set_aspect_ratio(rw, rh)
        # persist
        self.settings.setValue("ratio", ratio_to_text((rw, rh)))
        self._log.info("Aspect ratio set to %s", ratio_to_text((rw, rh)))

        if update_menu_checks:
            txt = ratio_to_text((rw, rh))
            self.action_ratio_169.setChecked(txt == "16:9")
            self.action_ratio_43.setChecked( txt == "4:3")
            self.action_ratio_11.setChecked( txt == "1:1")
            self.action_ratio_219.setChecked(txt == "21:9")

    def _on_custom_ratio(self):
        current = self.settings.value("ratio", "16:9")
        text, ok = QInputDialog.getText(self, "Custom Aspect Ratio",
                                        "Enter ratio (e.g., 16:9):", text=current)
        if ok and text:
            r = parse_ratio_text(text, fallback=parse_ratio_text(current))
            self._apply_ratio(r, update_menu_checks=True)

    # ===== Debug toggles =====
    def _on_toggle_debug(self, enabled: bool):
        self.settings.setValue("debugEnabled", enabled)
        # Adjust console handler level
        root = logging.getLogger(APP_NAME)
        for h in root.handlers:
            # console handler is StreamHandler; we keep file at DEBUG always
            if isinstance(h, RotatingFileHandler):
                continue
            h.setLevel(logging.DEBUG if enabled else logging.INFO)
        root.setLevel(logging.DEBUG if enabled else logging.INFO)
        self._log.info("Debug logging: %s", enabled)

    def _on_toggle_fps(self, enabled: bool):
        self.settings.setValue("showFps", enabled)
        self.player.set_show_fps(enabled)

    def _on_restart_cameras(self):
        self._log.info("Restarting camera by user request.")
        self.player.restart()

    def _on_dump_settings(self):
        keys = ["theme", "debugEnabled", "showFps", "geometry", "splitterSizes", "maximized"]
        for k in keys:
            self._log.info("Settings[%s] = %r", k, self.settings.value(k))

    # ===== Public API: add a log row (with count) =====
    def add_log(self, user, action, item, count=1):
        ts = QDateTime.currentDateTime().toString("yyyy-MM-dd HH:mm:ss")
        row = self.log_table.rowCount()
        self.log_table.insertRow(row)
        for col, val in enumerate([ts, user, action, item]):
            self.log_table.setItem(row, col, QTableWidgetItem(str(val)))
        count_item = QTableWidgetItem(str(int(count)))
        count_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.log_table.setItem(row, 4, count_item)
        self.log_table.scrollToBottom()
        # 이하 기존 강조/인벤토리 갱신 로직 그대로 유지
        self._log.debug("Log row: %s | %s | %s | %s", ts, user, action, item)

        # highlight alerts (Action or Item equals "경고!" or action == "Alert")
        self._paint_log_row(row, self._is_alert_entry(user, action, item))

        # --- NEW: inventory count update ---
        delta = self._delta_for_action(action, count)
        if delta != 0:
            self._update_item_count(item, delta)

        # --- NEW: push notification for '경고!' (action or item) ---
        if str(action).strip() == "경고!" or str(item).strip() == "경고!":
            self.push_notification(f"[{user}] {item} - 경고 발생")

    def _is_alert_entry(self, user, action, item) -> bool:
        a = (str(action) or "").strip()
        i = (str(item) or "").strip()
        return a == "경고!" or i == "경고!" or a.lower() == "alert"

    def _paint_log_row(self, row: int, alert: bool):
        # Choose colors per theme
        if alert:
            if self.current_theme == "dark":
                bg = QColor("#4a1214")   # dark red-ish
                fg = QColor("#ffcdd2")   # soft pink text
            else:
                bg = QColor("#ffebee")   # light red-ish
                fg = QColor("#b71c1c")   # deep red text
            bold = True
        else:
            bg = QColor(0, 0, 0, 0)      # transparent -> use normal alt row colors
            fg = None
            bold = False

        for col in range(self.log_table.columnCount()):
            it = self.log_table.item(row, col)
            if not it:
                continue
            it.setBackground(QBrush(bg))
            if fg:
                it.setForeground(QBrush(fg))
            f = it.font()
            f.setBold(bold)
            it.setFont(f)


    # ===== Fullscreen toggle =====
    def toggle_fullscreen(self):
        if self.isMaximized():
            self.showNormal()
            if self._normal_geometry:
                self.setGeometry(self._normal_geometry)
            self._log.info("Exit maximized.")
        else:
            self._normal_geometry = self.geometry()
            self.showMaximized()
            self._log.info("Enter maximized.")
        QTimer.singleShot(0, self._post_layout_fix)
        QTimer.singleShot(0, self._apply_edge_to_edge)

    def changeEvent(self, e):
        if e.type() == QEvent.WindowStateChange:
            QTimer.singleShot(0, self._apply_edge_to_edge)
        super().changeEvent(e)

    def _post_layout_fix(self):
        self._update_button_sizes()
        self._update_topbar_height()
        self._update_topbar_spacing()

    # ===== Resize border detection =====
    def _hit_test(self, pos):
        x, y = pos.x(), pos.y()
        w, h = self.width(), self.height()
        m = self.RESIZE_MARGIN
        left   = x <= m
        right  = x >= w - m
        top    = y <= m
        bottom = y >= h - m
        if top and left: return 'topleft'
        if top and right: return 'topright'
        if bottom and left: return 'bottomleft'
        if bottom and right: return 'bottomright'
        if left: return 'left'
        if right: return 'right'
        if top: return 'top'
        if bottom: return 'bottom'
        return None

    def _cursor_for_region(self, region):
        mapping = {
            'left': Qt.SizeHorCursor, 'right': Qt.SizeHorCursor,
            'top': Qt.SizeVerCursor, 'bottom': Qt.SizeVerCursor,
            'topleft': Qt.SizeFDiagCursor, 'bottomright': Qt.SizeFDiagCursor,
            'topright': Qt.SizeBDiagCursor, 'bottomleft': Qt.SizeBDiagCursor,
        }
        return mapping.get(region, Qt.ArrowCursor)

    # ===== Mouse events on the window (resize-only; dragging handled by TopBar) =====
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            region = self._hit_test(event.pos())
            if region and not self.isFullScreen():
                self._resizing = True
                self._resize_region = region
                self._resize_start_geo = self.geometry()
                self._resize_start_mouse = event.globalPos()
                self.setCursor(self._cursor_for_region(region))
                event.accept()
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._resizing and (event.buttons() & Qt.LeftButton):
            self._perform_resize(event.globalPos())
            event.accept()
            return
        if not self.isFullScreen():
            region = self._hit_test(event.pos())
            self.setCursor(self._cursor_for_region(region))
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self._resizing:
            self._resizing = False
            self._resize_region = None
            self.setCursor(Qt.ArrowCursor)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def _perform_resize(self, global_pos):
        dx = global_pos.x() - self._resize_start_mouse.x()
        dy = global_pos.y() - self._resize_start_mouse.y()
        g = QRect(self._resize_start_geo)
        if 'left' in self._resize_region:
            new_x = g.x() + dx
            new_w = g.width() - dx
            if new_w >= self.MIN_W:
                g.setX(new_x); g.setWidth(new_w)
        if 'right' in self._resize_region:
            new_w = g.width() + dx
            if new_w >= self.MIN_W:
                g.setWidth(new_w)
        if 'top' in self._resize_region:
            new_y = g.y() + dy
            new_h = g.height() - dy
            if new_h >= self.MIN_H:
                g.setY(new_y); g.setHeight(new_h)
        if 'bottom' in self._resize_region:
            new_h = g.height() + dy
            if new_h >= self.MIN_H:
                g.setHeight(new_h)
        self.setGeometry(g)

    # ===== Responsive titlebar buttons =====
    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_button_sizes()
        self._update_topbar_height()
        self._update_topbar_spacing()
        self._position_notif_badge()

    def _update_button_sizes(self):
        base = int(min(self.width(), self.height()) * 0.05)
        base = max(12, min(32, base))
        for btn in (self.settings_button, self.notifications_button, self.inventory_button, self.min_button, self.fullsize_button, self.close_button):
            btn.setFixedSize(base, base)
            btn.setIconSize(btn.size())
        self._position_notif_badge()  # keep badge in the bottom-right corner

    def _update_topbar_height(self):
        btn_h = self.min_button.height()
        self.top_bar_widget.setMinimumHeight(int(btn_h + 12))
        self.top_bar_widget.setMaximumHeight(int(btn_h + 14))

    def _update_topbar_spacing(self):
        gap = max(4, min(12, int(self.min_button.width() * 0.25)))
        self.top_bar_layout.setSpacing(gap)

    def closeEvent(self, event):
        # Save geometry & maximized state & splitter sizes
        self.settings.setValue("geometry", self.saveGeometry())
        self.settings.setValue("maximized", self.isMaximized())
        self.settings.setValue("splitterSizes", self.splitter.sizes())
        self._log.info("Saved geometry/maximized/splitter. Closing.")
        # Stop camera
        self.player.close()
        super().closeEvent(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)

    default_ratio = (16, 9)
    w = FramelessWindow(face_source=0, ratio=default_ratio)

    # Restore maximized state AFTER creating the window
    if w.settings.value("maximized", False, type=bool):
        w.showMaximized()
    else:
        w.show()

    sys.exit(app.exec_())

 # Settings are stored under org="2025-AI-Project", app="StorageMonitorUI". Change those two strings if you want a different storage key. (At the top of the code lol)

 # <a target="_blank" href="https://icons8.com/icon/43725/cancel">Cancel</a> icon by <a target="_blank" href="https://icons8.com">Icons8</a>

