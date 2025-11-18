# -*- coding: utf-8 -*-
"""
StorageMonitorUI – v4.1
- CLI: --video, --result-jsonl, [--decisions-jsonl], [--debug]
- 동시 인식(이벤트) 트리거 + 에러 트리거(사람만/물건만 2초)
- ui_cues.jsonl 폴링하여 NO_FACE/NO_ITEM 즉시 반영 (상태머신으로 중복 팝업 억제)
- 에러 팝업은 상황이 바뀔 때까지 1회만. 상단 배너로 지속 안내.
- 수동 정정(DecisionDialog) + 로그 테이블 우클릭 수정 메뉴
- 아이콘: ./assets/icons/*.png  (스크립트 기준)
"""

import sys, os, time, json, logging, random, argparse
from logging.handlers import RotatingFileHandler
from pathlib import Path
import cv2

from dataclasses import dataclass
from collections import Counter, deque, defaultdict

from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QHBoxLayout, QVBoxLayout, QGraphicsDropShadowEffect,
    QLabel, QSplitter, QTableWidget, QTableWidgetItem, QAbstractItemView, QHeaderView,
    QSizePolicy, QMenu, QAction, QActionGroup, QInputDialog, QMessageBox, QDialog,
    QLineEdit, QSpinBox, QComboBox, QFormLayout, QPlainTextEdit, QCheckBox
)
from PyQt5.QtGui import QIcon, QImage, QPixmap, QColor, QBrush, QCursor
from PyQt5.QtCore import Qt, QPoint, QRect, QTimer, QDateTime, QEvent, QSettings, QSize

# -------------------- App constants --------------------
APP_ORG = "2025-AI-Project"
APP_NAME = "StorageMonitorUI"
LOG_DIR = Path.home() / ".storage_monitor_ui"

ROOT = Path(__file__).resolve().parent
ICON_DIR = ROOT / "assets" / "icons"

# ===== 동시 인식 트리거 =====
HOLD_MIN_S   = 1.5
RESET_MIN_S  = 0.5
FACE_SIM_THR = 0.40

# ===== 에러 트리거 =====
ONLY_FACE_ERR_S   = 2.0   # 사람만
ONLY_ITEM_ERR_S   = 2.0   # 물건만
NORMAL_CLEAR_S    = 0.5   # 얼굴+물건 동시로 유지 시 해제
NEUTRAL_CLEAR_S   = 1.0   # 둘 다 없음 유지 시 해제
BANNER_MIN_SHOW_S = 1.5   # 해제 직전 깜빡임 방지용 최소 노출 시간

# -------------------- CLI --------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="재생할 분석된 mp4(오버레이) 경로")
    ap.add_argument("--result-jsonl", required=True, help="프레임 로그 JSONL 경로(*_result.jsonl)")
    ap.add_argument("--decisions-jsonl", help="결정 로그 JSONL(없으면 *_decisions.jsonl로 자동 생성)")
    ap.add_argument("--debug", action="store_true", help="디버그 로그 활성화")
    return ap.parse_args()

# -------------------- Logging --------------------
def setup_logging(debug=False):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(APP_NAME)
    logger.setLevel(logging.DEBUG if debug else logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")

    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.DEBUG if debug else logging.INFO)
    sh.setFormatter(fmt)
    for h in list(logger.handlers):
        logger.removeHandler(h)
    logger.addHandler(sh)

    fh = RotatingFileHandler(LOG_DIR / "app.log", maxBytes=1_000_000, backupCount=3, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger

# -------------------- 아이콘 로더 --------------------
def load_icon(filename: str, fallback_text_button: QPushButton = None) -> QIcon:
    p = ICON_DIR / filename
    if p.exists():
        return QIcon(str(p))
    logging.getLogger(APP_NAME).warning("Icon missing: %s", p)
    if fallback_text_button is not None and not fallback_text_button.text():
        name = Path(filename).stem
        fallback_text_button.setText(name.replace("_", " "))
    return QIcon()

# -------------------- Result reader --------------------
@dataclass
class FrameRec:
    t_s: float
    faces: list
    items: list

class ResultStream:
    def __init__(self, path: str):
        self.frames = []
        self.idx = 0
        self._log = logging.getLogger(f"{APP_NAME}.ResultStream")
        if not path or not Path(path).exists():
            self._log.warning("Result JSONL not found: %s", path)
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
        self._log.info("Loaded %d frames.", len(self.frames))

    def reset_to_time(self, t: float):
        lo, hi = 0, len(self.frames)
        while lo < hi:
            mid = (lo + hi) // 2
            if self.frames[mid].t_s < t:
                lo = mid + 1
            else:
                hi = mid
        self.idx = max(0, lo - 1)

    def _current(self, t):
        while self.idx + 1 < len(self.frames) and self.frames[self.idx + 1].t_s <= t:
            self.idx += 1
        if not self.frames:
            return None
        return self.frames[self.idx]

    def state_flags_at(self, t: float):
        fr = self._current(t)
        if fr is None:
            return False, False, None, None
        # 얼굴 대표
        best = None
        for f in fr.faces:
            name = str(f.get("person_id", "") or "").strip()
            sim = float(f.get("similarity", 0.0))
            if name and name.lower() != "unknown" and sim >= FACE_SIM_THR:
                if best is None or sim > best[1]:
                    best = (name, sim)
        face_name = best[0] if best else None
        has_face = len(fr.faces) > 0

        # 품목 대표
        item_class = None
        for it in fr.items:
            cls = str(it.get("class_name", "") or "").strip()
            if cls:
                item_class = cls
                break
        has_item = len(fr.items) > 0

        return has_face, has_item, face_name, item_class

# -------------------- UI 위젯 --------------------
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
                border-bottom: 1px solid {colors['divider']};
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

class DecisionDialog(QDialog):
    """이벤트 수동 분류/정정 대화상자 (편집 가능)"""
    def __init__(self, person: str, item: str, t_s: float, item_candidates=None, person_candidates=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("이벤트 분류/정정")
        self.choice = None
        self.edited_person = person or ""
        self.edited_item = item or ""
        item_candidates = item_candidates or []
        person_candidates = person_candidates or []

        self.person_edit = QComboBox()
        self.person_edit.setEditable(True)
        self.person_edit.addItems(sorted(set([p for p in person_candidates if p] + ([person] if person else []))))
        if person: self.person_edit.setCurrentText(person)

        self.item_edit = QComboBox()
        self.item_edit.setEditable(True)
        self.item_edit.addItems(sorted(set([c for c in item_candidates if c] + ([item] if item else []))))
        if item: self.item_edit.setCurrentText(item)

        t_label = QLabel(f"{t_s:.2f} s")
        self.btn_issue  = QPushButton("불출")
        self.btn_return = QPushButton("반납")
        self.btn_skip   = QPushButton("건너뛰기")

        self.btn_issue.clicked.connect(lambda: self._set_choice("불출"))
        self.btn_return.clicked.connect(lambda: self._set_choice("반납"))
        self.btn_skip.clicked.connect(self.reject)

        form = QFormLayout()
        form.addRow("시간", t_label)
        form.addRow("이름", self.person_edit)
        form.addRow("품목", self.item_edit)

        row = QHBoxLayout()
        row.addWidget(self.btn_issue)
        row.addWidget(self.btn_return)
        row.addWidget(self.btn_skip)

        lay = QVBoxLayout(self)
        lay.addLayout(form)
        lay.addLayout(row)
        self.setMinimumWidth(360)

    def _set_choice(self, c):
        self.choice = c
        self.edited_person = (self.person_edit.currentText() or "").strip()
        self.edited_item = (self.item_edit.currentText() or "").strip()
        self.accept()

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
        self.view = QLabel("영상 로딩 중…")
        self.view.setAlignment(Qt.AlignCenter)
        self.view.setMinimumSize(240, 135)
        self.fps_label = QLabel("FPS: --", self.view)
        self.fps_label.hide()

        self.ar = AspectRatioBox(ratio=aspect_ratio, child=self.view, parent=self)
        lay = QVBoxLayout(self); lay.setContentsMargins(0, 0, 0, 0); lay.setSpacing(6)
        lay.addWidget(self.title_label); lay.addWidget(self.ar, 1)

        self._last_pixmap = None
        self._show_fps = False
        self._last_times = []
        self.playing = True
        self.frame_idx = 0
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._on_tick)
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
        if self.aspect_ratio == (9, 16) and frame.shape[1] > frame.shape[0]:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, w * ch, QImage.Format_RGB888)
        self._last_pixmap = QPixmap.fromImage(qimg)
        self._render_scaled()
        if self._show_fps:
            now = time.time()
            self._last_times.append(now)
            while self._last_times and now - self._last_times[0] > 1.0:
                self._last_times.pop(0)
            self.fps_label.setText(f"FPS: {len(self._last_times):02d}")
    def _render_scaled(self):
        if self._last_pixmap is None: return
        self.view.setPixmap(self._last_pixmap.scaled(self.view.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
    def closeEvent(self, e):
        try:
            self.timer.stop()
            self.cap.release()
        finally:
            super().closeEvent(e)

class DebugConsole(QDialog):
    def __init__(self, host_window: 'FramelessWindow'):
        super().__init__(host_window)
        self.host = host_window
        self.setWindowTitle("Debug Console")
        self.setModal(False)
        self.setMinimumWidth(420)

        self.user_edit = QLineEdit("사용자")
        self.action_combo = QComboBox()
        self.action_combo.addItems(["반납", "불출", "경고!"])
        self.item_edit = QLineEdit("품목")
        self.count_spin = QSpinBox(); self.count_spin.setRange(1, 999); self.count_spin.setValue(1)

        self.note_edit = QLineEdit("(알림 메시지)")

        self.chk_log_also = QCheckBox("알림을 로그에도 남기기")
        self.chk_log_also.setChecked(True)

        form = QFormLayout()
        form.addRow("이름:", self.user_edit)
        form.addRow("반납/불출:", self.action_combo)
        form.addRow("품목:", self.item_edit)
        form.addRow("수량:", self.count_spin)
        form.addRow("알림:", self.note_edit)
        form.addRow("", self.chk_log_also)

        self.btn_add = QPushButton("로그 추가")
        self.btn_add10 = QPushButton("랜덤 10개 추가")
        self.btn_notify = QPushButton("알림 테스트")

        btn_row = QHBoxLayout()
        btn_row.addWidget(self.btn_add)
        btn_row.addWidget(self.btn_add10)
        btn_row.addWidget(self.btn_notify)

        self.out = QPlainTextEdit(); self.out.setReadOnly(True); self.out.setMaximumBlockCount(500)

        lay = QVBoxLayout(self)
        lay.addLayout(form)
        lay.addLayout(btn_row)
        lay.addWidget(self.out)

        self.btn_add.clicked.connect(self._add_one)
        self.btn_add10.clicked.connect(self._add_ten)
        self.btn_notify.clicked.connect(self._push_notif)

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
        msg = self.note_edit.text().strip() or "(알림 메시지)"
        self.host.push_notification(msg)
        self._log_out(f"push_notification({msg!r})")
        if self.chk_log_also.isChecked():
            self.host.add_log("SYSTEM", "경고!", msg, 1)

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
        # ---------- 인벤토리 다이얼로그 ----------
    def _open_inventory(self):
        # 다이얼로그가 없으면 생성
        if not hasattr(self, "_inventory_dialog") or self._inventory_dialog is None:
            self._inventory_dialog = InventoryDialog(self)

        # 테마/데이터 반영
        try:
            self._inventory_dialog.apply_theme(self.themes[self.current_theme])
        except Exception:
            pass
        self._inventory_dialog.set_counts(self.item_counts)

        # 표시
        self._inventory_dialog.show()
        self._inventory_dialog.raise_()
        self._inventory_dialog.activateWindow()

    def __init__(self, video_path=None, result_jsonl=None, ratio=(16, 9), decisions_jsonl=None):
        super().__init__()
        self._log = logging.getLogger(f"{APP_NAME}.Window")
        self.video_path = video_path
        self.result_jsonl = result_jsonl
        self.decisions_jsonl = decisions_jsonl  # may be None; we will set default below

        # 파일/경로 유도
        res_path = Path(self.result_jsonl)
        base_dir = res_path.parent
        base_stem = res_path.stem.replace("_result", "")
        if not self.decisions_jsonl:
            self.decisions_jsonl = str(base_dir / f"{base_stem}_decisions.jsonl")
        self.corrections_jsonl = str(base_dir / f"{base_stem}_corrections.jsonl")
        self.ui_cues_jsonl = base_dir / f"{base_stem}_ui_cues.jsonl"
        self._ui_cues_offset = 0  # tail offset

        # 상태(이벤트/에러)
        self.hold_acc_s = 0.0
        self.reset_acc_s = 0.0
        self.armed = True
        self.last_t = 0.0
        self.window_names = deque(maxlen=90)
        self.window_items = deque(maxlen=90)

        # 에러 상태머신(중복 억제)
        self.only_face_acc_s = 0.0
        self.only_item_acc_s = 0.0
        self.no_item_active  = False  # 사람만
        self.no_face_active  = False  # 물건만
        self.normal_acc_s    = 0.0    # 얼굴+물건 동시 유지 누적
        self.neutral_acc_s   = 0.0    # 둘 다 없음 유지 누적
        self.banner_started_at_s = None

        # 후보 캐시
        self.recent_person_names = deque(maxlen=200)
        self.recent_item_names = deque(maxlen=200)

        # 창/테마
        self.setWindowFlags(Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.settings = QSettings(APP_ORG, APP_NAME)
        self.resize(1100, 740)
        self.setMinimumSize(self.MIN_W, self.MIN_H)
        self._normal_geometry = None
        self.current_theme = 'light'
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

        # ----- Outer container -----
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

        # ----- Top buttons -----
        self.settings_button = QPushButton("")
        self.settings_button.setObjectName("settingsBtn")
        self.settings_button.setIcon(load_icon("settings_button.png", self.settings_button))
        self.settings_button.setCursor(Qt.PointingHandCursor)

        self.settings_menu = QMenu(self.settings_button)

        self.theme_menu = QMenu("테마", self.settings_menu)
        self.action_light = QAction("밝은 테마", self, checkable=True)
        self.action_dark = QAction("어두운 테마", self, checkable=True)
        g_theme = QActionGroup(self); g_theme.setExclusive(True)
        g_theme.addAction(self.action_light); g_theme.addAction(self.action_dark)
        self.theme_menu.addAction(self.action_light)
        self.theme_menu.addAction(self.action_dark)

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

        self.debug_menu = QMenu("Debug", self.settings_menu)
        self.action_debug_logs = QAction("Enable debug logging", self, checkable=True)
        self.action_show_fps  = QAction("Show FPS overlay", self, checkable=True)
        self.action_restart_cams = QAction("Restart video", self)
        self.action_dump_settings = QAction("Dump settings to log", self)
        self.action_open_debug_console = QAction("Open debug console…", self)
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

        self.settings_menu.addMenu(self.theme_menu)
        self.settings_menu.addMenu(self.ratio_menu)
        self.settings_menu.addMenu(self.debug_menu)
        self.settings_button.setMenu(self.settings_menu)

        self.action_light.triggered.connect(lambda: self.apply_theme('light'))
        self.action_dark.triggered.connect(lambda: self.apply_theme('dark'))

        self.action_ratio_169.triggered.connect(lambda: self._apply_ratio((16, 9)))
        self.action_ratio_43.triggered.connect(lambda: self._apply_ratio((4, 3)))
        self.action_ratio_11.triggered.connect(lambda: self._apply_ratio((1, 1)))
        self.action_ratio_219.triggered.connect(lambda: self._apply_ratio((21, 9)))
        self.action_ratio_custom.triggered.connect(self._on_custom_ratio)

        self.action_debug_logs.toggled.connect(self._on_toggle_debug)
        self.action_show_fps.toggled.connect(self._on_toggle_fps)
        self.action_restart_cams.setText("Restart video")
        self.action_restart_cams.triggered.connect(self._on_restart_video)
        self.action_dump_settings.triggered.connect(self._on_dump_settings)
        self.action_open_debug_console.triggered.connect(self._open_debug_console)
        self._debug_console = None

        self.action_rebuild_counts.triggered.connect(self._rebuild_counts_from_log)
        self.action_clear_counts.triggered.connect(self._clear_counts)
        self.action_seed_counts.triggered.connect(self._seed_counts_prompt)

        self.min_button = QPushButton("")
        self.min_button.setObjectName("minBtn")
        self.min_button.setIcon(load_icon("minimize_button.png", self.min_button))
        self.min_button.clicked.connect(self.showMinimized)
        self.min_button.setCursor(Qt.PointingHandCursor)

        self.fullsize_button = QPushButton("")
        self.fullsize_button.setObjectName("fullBtn")
        self.fullsize_button.setIcon(load_icon("fullsize_button.png", self.fullsize_button))
        self.fullsize_button.clicked.connect(self.toggle_fullscreen)
        self.fullsize_button.setCursor(Qt.PointingHandCursor)

        self.close_button = QPushButton("")
        self.close_button.setObjectName("closeBtn")
        self.close_button.setIcon(load_icon("close_button.png", self.close_button))
        self.close_button.clicked.connect(self.close)
        self.close_button.setCursor(Qt.PointingHandCursor)

        self.notifications_button = QPushButton("")
        self.notifications_button.setObjectName("notifBtn")
        self.notifications_button.setIcon(load_icon("notifications_button.png", self.notifications_button))
        self.notifications_button.setCursor(Qt.PointingHandCursor)

        self.notifications_menu = QMenu(self.notifications_button)
        self._notif_title = QAction("알림 목록", self); self._notif_title.setEnabled(False)
        self._notif_separator = self.notifications_menu.addSeparator()
        self.notifications_menu.insertAction(self._notif_separator, self._notif_title)

        self.action_notif_mark_read = QAction("전부 읽음처리", self)
        self.action_notif_clear = QAction("비우기", self)
        self.notifications_menu.addSeparator()
        self.notifications_menu.addAction(self.action_notif_mark_read)
        self.notifications_menu.addAction(self.action_notif_clear)

        self.notifications_button.setMenu(self.notifications_menu)

        self._notif_badge = QLabel(self.notifications_button)
        self._notif_badge.setFixedSize(10, 10)
        self._notif_badge.setStyleSheet("background:#e81123; border:1px solid white; border-radius:5px;")
        self._notif_badge.hide()
        self.unread_notif_count = 0

        self.notifications_menu.aboutToShow.connect(self._on_notif_menu_opened)
        self.action_notif_mark_read.triggered.connect(self._on_mark_all_read)
        self.action_notif_clear.triggered.connect(self._on_clear_notifications)

        self.inventory_button = QPushButton("")
        self.inventory_button.setObjectName("invBtn")
        self.inventory_button.setIcon(load_icon("table_button.png", self.inventory_button))
        self.inventory_button.setCursor(Qt.PointingHandCursor)
        self.inventory_button.clicked.connect(self._open_inventory)

        self._inventory_dialog = None
        self.item_counts = {}

        # Top bar
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

        # Alert banner (상단 고정)
        self.alert_banner = QLabel("")
        self.alert_banner.setObjectName("alertBanner")
        self.alert_banner.setWordWrap(True)
        self.alert_banner.setVisible(False)
        self.alert_banner.setAlignment(Qt.AlignCenter)
        self.alert_banner.setStyleSheet("""
            #alertBanner {
                background: rgba(255, 215, 0, 0.18);
                color: #8a6d00;
                border: 1px solid rgba(255, 193, 7, 0.6);
                border-radius: 8px;
                padding: 8px 10px;
                margin: 6px 8px 0 8px;
                font-weight: 600;
            }
        """)

        # Player column
        self.player = VideoPlayer(self.video_path, title="입력 영상", aspect_ratio=ratio)

        cameras_col = QWidget()
        cam_layout = QVBoxLayout(cameras_col)
        cam_layout.setContentsMargins(8, 8, 8, 8)
        cam_layout.setSpacing(8)
        cam_layout.addWidget(self.alert_banner)  # 배너를 영상 위쪽에
        cam_layout.addWidget(self.player, 1)
        cameras_col.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # Log table
        self.log_table = QTableWidget(0, 5)
        self.log_table.setObjectName("logBox")
        self.log_table.setHorizontalHeaderLabels(["시간", "이름", "반납/불출", "품목", "수량"])
        self.log_table.verticalHeader().setVisible(False)
        self.log_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.log_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.log_table.setAlternatingRowColors(True)
        self.log_table.setShowGrid(True)
        self.log_table.setGridStyle(Qt.SolidLine)
        self.log_table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.log_table.customContextMenuRequested.connect(self._open_log_context_menu)

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

        # 테마/옵션 복원
        saved_theme = self.settings.value("theme", "light")
        self.apply_theme(saved_theme)

        debug_on = self.settings.value("debugEnabled", False, type=bool)
        self.action_debug_logs.setChecked(bool(debug_on))
        self._on_toggle_debug(bool(debug_on))

        show_fps = self.settings.value("showFps", False, type=bool)
        self.action_show_fps.setChecked(bool(show_fps))
        self._on_toggle_fps(bool(show_fps))

        self._restore_geometry_and_splitter()

        saved_ratio_text = self.settings.value("ratio", "16:9")
        self._apply_ratio(parse_ratio_text(saved_ratio_text), update_menu_checks=True)

        try:
            import PyQt5
            self._log.info("Environment: PyQt5=%s, OpenCV=%s", PyQt5.QtCore.QT_VERSION_STR, cv2.__version__)
        except Exception:
            pass

        # 데이터 스트림
        self.stream = ResultStream(self.result_jsonl)

        # 타이머: UI 프레임 + UI cues 폴링
        self.ui_timer = QTimer(self)
        self.ui_timer.timeout.connect(self._on_tick)
        self.ui_timer.start(30)

        self.cues_timer = QTimer(self)
        self.cues_timer.timeout.connect(self._poll_ui_cues)
        self.cues_timer.start(150)

    # ---------- Debug console open ----------
    def _open_debug_console(self):
        if self._debug_console is None:
            self._debug_console = DebugConsole(self)
        self._debug_console.apply_theme(self.themes[self.current_theme])
        self._debug_console.show()
        self._debug_console.raise_()
        self._debug_console.activateWindow()

    # ---------- 컨텍스트 메뉴(로그 수정) ----------
    def _open_log_context_menu(self, pos):
        idx = self.log_table.indexAt(pos)
        if not idx.isValid():
            return
        row = idx.row()
        menu = QMenu(self)

        act_edit_name = QAction("이름 수정…", self)
        act_edit_item = QAction("품목 수정…", self)
        act_edit_count = QAction("수량 수정…", self)

        act_edit_name.triggered.connect(lambda: self._edit_log_cell(row, 1, "이름"))
        act_edit_item.triggered.connect(lambda: self._edit_log_cell(row, 3, "품목"))
        act_edit_count.triggered.connect(lambda: self._edit_log_cell(row, 4, "수량", numeric=True))

        menu.addAction(act_edit_name)
        menu.addAction(act_edit_item)
        menu.addAction(act_edit_count)

        menu.exec_(QCursor.pos())

    def _edit_log_cell(self, row, col, label, numeric=False):
        it = self.log_table.item(row, col)
        before = it.text() if it else ""
        text, ok = QInputDialog.getText(self, f"{label} 수정", f"새 {label}:", text=before)
        if not ok:
            return
        new = (text or "").strip()
        if numeric:
            try:
                int(new)
            except Exception:
                QMessageBox.warning(self, "입력 오류", "정수만 입력할 수 있습니다.")
                return
        self.log_table.setItem(row, col, QTableWidgetItem(new))
        self._paint_log_row(row, self._is_alert_entry(
            self.log_table.item(row, 1).text() if self.log_table.item(row,1) else "",
            self.log_table.item(row, 2).text() if self.log_table.item(row,2) else "",
            self.log_table.item(row, 3).text() if self.log_table.item(row,3) else ""
        ))
        # corrections.jsonl 기록
        corr = {
            "ts": QDateTime.currentDateTime().toString("yyyy-MM-dd HH:mm:ss"),
            "row": int(row),
            "field": {1: "name", 3: "item", 4: "count"}.get(col, f"col{col}"),
            "before": before,
            "after": new
        }
        try:
            with open(self.corrections_jsonl, "a", encoding="utf-8") as f:
                f.write(json.dumps(corr, ensure_ascii=False) + "\n")
        except Exception as e:
            QMessageBox.critical(self, "쓰기 오류", f"corrections 저장 실패: {e}")

        # 재고 자동 갱신은 간단히 전체 재계산
        self._rebuild_counts_from_log()

    # ---------- UI cues 폴링 ----------
    def _poll_ui_cues(self):
        path = self.ui_cues_jsonl
        if not path.exists():
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                f.seek(self._ui_cues_offset)
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        o = json.loads(line)
                    except Exception:
                        continue
                    if o.get("type") == "ERROR":
                        reason = (o.get("reason", "") or "").upper()
                        # 파일은 트리거 신호로만 사용, 실제 억제/재등장은 GUI 상태머신이 관리
                        if reason == "NO_FACE":
                            self._handle_error_kind("NO_FACE")
                        elif reason == "NO_ITEM":
                            self._handle_error_kind("NO_ITEM")
                self._ui_cues_offset = f.tell()
        except Exception as e:
            self._log.exception("UI cues read error: %s", e)

    # ---------- 에러 진입 공통 처리(1회 팝업 + 배너) ----------
    def _handle_error_kind(self, kind: str):
        now_t = self.player.time_s()
        if kind == "NO_FACE":
            # 물건만: 배너 문구
            if self.no_face_active:
                return
            self.no_face_active = True
            self.banner_started_at_s = now_t
            self._show_banner("⚠ 물건과 얼굴이 함께 화면에 잘 보이도록 정면을 바라봐 주세요.\n(팁) 얼굴은 카메라 중앙, 물건은 가슴 위쪽 위치에서 1~2초 유지")
            self._emit_error_once(kind, f"물건만 {ONLY_ITEM_ERR_S:.1f}s 이상 감지됨")
        elif kind == "NO_ITEM":
            if self.no_item_active:
                return
            self.no_item_active = True
            self.banner_started_at_s = now_t
            self._show_banner("⚠ 인식할 물품을 가슴 위쪽 위치에서 1~2초간 명확히 보여 주세요.")
            self._emit_error_once(kind, f"사람만 {ONLY_FACE_ERR_S:.1f}s 이상 감지됨")

    def _emit_error_once(self, kind: str, message: str):
        # 알림 + 로그 + 팝업(1회)
        self.push_notification(f"[{kind}] {message}")
        self.add_log("SYSTEM", "경고!", kind, 1)
        self.player.pause()
        QMessageBox.warning(self, f"에러: {kind}", message)
        self.player.play()

    # ---------- 메인 Tick ----------
    def _on_tick(self):
        t = self.player.time_s()
        dt = max(0.0, t - self.last_t)
        self.last_t = t

        has_face, has_item, name, item = self.stream.state_flags_at(t)

        # 후보 캐시
        if name: self.recent_person_names.append(name)
        if item: self.recent_item_names.append(item)

        # 1) 정상 이벤트(동시 인식) 모달
        if has_face and has_item:
            self.hold_acc_s += dt
            self.reset_acc_s = 0.0
            if name: self.window_names.append(name)
            if item: self.window_items.append(item)
            if self.armed and self.hold_acc_s >= HOLD_MIN_S:
                rep_name = self._mode(self.window_names) or (name or "Unknown")
                rep_item = self._mode(self.window_items) or (item or "Unknown")
                logging.getLogger(APP_NAME).info("Trigger @ %.2fs  name=%s  item=%s", t, rep_name, rep_item)
                self._trigger_modal(t, rep_name, rep_item)
        else:
            self.reset_acc_s += dt
            if self.reset_acc_s >= RESET_MIN_S:
                self.armed = True
                self.hold_acc_s = 0.0
                self.window_names.clear(); self.window_items.clear()

        # 2) 에러 상태머신
        # 진입 누적
        if has_face and not has_item:
            self.only_face_acc_s += dt
            self.only_item_acc_s = 0.0
            if self.only_face_acc_s >= ONLY_FACE_ERR_S:
                self._handle_error_kind("NO_ITEM")   # 사람만
                self.only_face_acc_s = 0.0
        elif has_item and not has_face:
            self.only_item_acc_s += dt
            self.only_face_acc_s = 0.0
            if self.only_item_acc_s >= ONLY_ITEM_ERR_S:
                self._handle_error_kind("NO_FACE")   # 물건만
                self.only_item_acc_s = 0.0
        else:
            self.only_face_acc_s = 0.0
            self.only_item_acc_s = 0.0

        # 해제 누적
        if has_face and has_item:
            self.normal_acc_s += dt
            self.neutral_acc_s = 0.0
        elif (not has_face) and (not has_item):
            self.neutral_acc_s += dt
            self.normal_acc_s = 0.0
        else:
            # 중립/정상 모두 아님 → 둘 다 리셋
            self.normal_acc_s = 0.0
            self.neutral_acc_s = 0.0

        # 해제 조건 충족 시 배너/상태 해제 (최소 노출 보장)
        if (self.no_face_active or self.no_item_active):
            banner_elapsed = (t - (self.banner_started_at_s or t))
            can_hide_by_time = (banner_elapsed >= BANNER_MIN_SHOW_S)
            if can_hide_by_time and (self.normal_acc_s >= NORMAL_CLEAR_S or self.neutral_acc_s >= NEUTRAL_CLEAR_S):
                self.no_face_active = False
                self.no_item_active = False
                self._hide_banner()
                self.normal_acc_s = 0.0
                self.neutral_acc_s = 0.0

    def _mode(self, dq):
        if not dq: return None
        return Counter(dq).most_common(1)[0][0]

    def _trigger_modal(self, t_s: float, person: str, item: str):
        self.player.pause()
        self.armed = False
        dlg = DecisionDialog(
            person, item, t_s,
            item_candidates=list(self.recent_item_names),
            person_candidates=list(self.recent_person_names),
            parent=self
        )
        if dlg.exec_() == QDialog.Accepted and dlg.choice:
            p2, i2, action = dlg.edited_person or person, dlg.edited_item or item, dlg.choice
            self._append_decision(t_s, p2, i2, action)
        self.player.play()

    # ---------- 배너 ----------
    def _show_banner(self, text: str):
        self.alert_banner.setText(text)
        self.alert_banner.setVisible(True)

    def _hide_banner(self):
        self.alert_banner.setVisible(False)
        self.alert_banner.setText("")

    def _format_ts_text(self, t_s: float) -> str:
        ms = int(round(t_s * 1000))
        s, ms = divmod(ms, 1000)
        m, s = divmod(s, 60)
        h, m = divmod(m, 60)
        return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"

    def _append_decision(self, t_s: float, person: str, item: str, action: str):
        rec = {
            "timestamp_s": QDateTime.currentDateTime().toString("yyyy-MM-dd HH:mm:ss"),
            "person_name": person,
            "item_class": item,
            "action": action,
        }
        try:
            with open(self.decisions_jsonl, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except Exception as e:
            QMessageBox.critical(self, "쓰기 오류", f"decisions 저장 실패: {e}")
            return

        self.add_log(person, action, item, count=1)

    # ---------- 알림/로그/재고 ----------
    def push_notification(self, text: str):
        ts = QDateTime.currentDateTime().toString("yyyy-MM-dd HH:mm:ss")
        act = QAction(f"[{ts}] {text}", self)
        act.setEnabled(False)
        self.notifications_menu.insertAction(self._notif_separator, act)
        self._set_unread_badge(self.unread_notif_count + 1)
        self._log.warning("Notification: %s", text)

    def _on_notif_menu_opened(self):
        if self.unread_notif_count:
            self._set_unread_badge(0)

    def _on_mark_all_read(self):
        self._set_unread_badge(0)

    def _on_clear_notifications(self):
        for a in list(self.notifications_menu.actions()):
            if a in (self._notif_title, self._notif_separator,
                     self.action_notif_mark_read, self.action_notif_clear):
                continue
            self.notifications_menu.removeAction(a)
        self._set_unread_badge(0)

    def _set_unread_badge(self, count: int):
        self.unread_notif_count = max(0, int(count))
        self._notif_badge.setVisible(self.unread_notif_count > 0)
        self._position_notif_badge()

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
        self._log.debug("Log row: %s | %s | %s | %s", ts, user, action, item)

        self._paint_log_row(row, self._is_alert_entry(user, action, item))

        delta = self._delta_for_action(action, count)
        if delta != 0:
            self._update_item_count(item, delta)

        if str(action).strip() == "경고!" or str(item).strip() == "경고!":
            self.push_notification(f"[{user}] {item} - 경고 발생")

    def _is_alert_entry(self, user, action, item) -> bool:
        a = (str(action) or "").strip()
        i = (str(item) or "").strip()
        return a == "경고!" or i == "경고!" or a.lower() == "alert"

    def _paint_log_row(self, row: int, alert: bool):
        if alert:
            if self.current_theme == "dark":
                bg = QColor("#4a1214")
                fg = QColor("#ffcdd2")
            else:
                bg = QColor("#ffebee")
                fg = QColor("#b71c1c")
            bold = True
        else:
            bg = QColor(0, 0, 0, 0)
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

    def _delta_for_action(self, action_text: str, count: int) -> int:
        a = (str(action_text) or "").strip()
        if a == "반납":
            return +int(count)
        if a == "불출":
            return -int(count)
        return 0

    def _update_item_count(self, item_name: str, delta: int):
        name = (str(item_name) or "").strip()
        if not name:
            return
        self.item_counts[name] = int(self.item_counts.get(name, 0)) + int(delta)
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
            delta = self._delta_for_action(action_txt, cnt)
            if item_txt and delta != 0:
                counts[item_txt] = int(counts.get(item_txt, 0)) + int(delta)
        self.item_counts = counts
        if hasattr(self, "_inventory_dialog") and self._inventory_dialog and self._inventory_dialog.isVisible():
            self._inventory_dialog.set_counts(self.item_counts)
        QMessageBox.information(self, "Inventory", f"Rebuilt counts from log ({len(counts)} items).")

    def _clear_counts(self):
        self.item_counts.clear()
        if hasattr(self, "_inventory_dialog") and self._inventory_dialog and self._inventory_dialog.isVisible():
            self._inventory_dialog.set_counts(self.item_counts)
        QMessageBox.information(self, "Inventory", "All inventory counts cleared.")

    def _seed_counts_prompt(self):
        text, ok = QInputDialog.getMultiLineText(
            self, "Seed counts manually…",
            "예시:\nBox-A=10, Box-B:5\nCrate-12 7\n(콤마/줄바꿈 구분, 마지막 값은 정수)",
            ""
        )
        if not ok or not text.strip():
            return
        updated = 0
        for chunk in [p.strip() for p in text.replace("\n", ",").split(",")]:
            if not chunk:
                continue
            name, val = None, None
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
        if hasattr(self, "_inventory_dialog") and self._inventory_dialog and self._inventory_dialog.isVisible():
            self._inventory_dialog.set_counts(self.item_counts)
        QMessageBox.information(self, "Inventory", f"Seeded {updated} item(s).")

    # ---------- 비주얼/UI ----------
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

        self.settings.setValue("theme", name)
        self.action_light.setChecked(name == 'light')
        self.action_dark.setChecked(name == 'dark')

        radius = 0 if self.isMaximized() else 15
        self.container.setStyleSheet(self._container_styles(c, radius))
        self.top_bar_widget.set_theme(c, radius)
        self.player.set_theme(c)
        self._log.info("Applied theme: %s", name)

        if hasattr(self, "_debug_console") and self._debug_console:
            self._debug_console.apply_theme(c)
        if hasattr(self, "_inventory_dialog") and self._inventory_dialog:
            self._inventory_dialog.apply_theme(c)

        for r in range(self.log_table.rowCount()):
            a = self.log_table.item(r, 2)
            i = self.log_table.item(r, 3)
            action_txt = a.text() if a else ""
            item_txt   = i.text() if i else ""
            self._paint_log_row(r, self._is_alert_entry("", action_txt, item_txt))

    def _apply_ratio(self, ratio_tuple, update_menu_checks=False):
        rw, rh = ratio_tuple
        self.player.set_aspect_ratio(rw, rh)
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

    def _on_toggle_debug(self, enabled: bool):
        self.settings.setValue("debugEnabled", enabled)
        root = logging.getLogger(APP_NAME)
        for h in root.handlers:
            if isinstance(h, RotatingFileHandler):
                continue
            h.setLevel(logging.DEBUG if enabled else logging.INFO)
        root.setLevel(logging.DEBUG if enabled else logging.INFO)
        self._log.info("Debug logging: %s", enabled)

    def _on_toggle_fps(self, enabled: bool):
        self.settings.setValue("showFps", enabled)
        self.player.set_show_fps(enabled)

    def _on_restart_video(self):
        self._log.info("Restarting video by user request.")
        self.player.restart()
        self.hold_acc_s = 0.0; self.reset_acc_s = 0.0; self.armed = True; self.last_t = 0.0
        self.window_names.clear(); self.window_items.clear()
        self.only_face_acc_s = self.only_item_acc_s = 0.0
        self.normal_acc_s = self.neutral_acc_s = 0.0
        self.no_face_active = self.no_item_active = False
        self._hide_banner()
        if hasattr(self, "stream"): self.stream.reset_to_time(0.0)

    def _on_dump_settings(self):
        keys = ["theme", "debugEnabled", "showFps", "geometry", "splitterSizes", "maximized"]
        for k in keys:
            self._log.info("Settings[%s] = %r", k, self.settings.value(k))

    # ---------- 창/리사이즈 ----------
    def _position_notif_badge(self):
        btn = self.notifications_button
        bsz = self._notif_badge.size()
        x = max(0, btn.width() - bsz.width() + 2)
        y = max(0, btn.height() - bsz.height() + 2)
        self._notif_badge.move(x, y)

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

    def _apply_edge_to_edge(self):
        edge = self.isMaximized()
        radius = 0 if edge else 15
        if edge:
            self.outer_layout.setContentsMargins(0, 0, 0, 0)
            self.shadow.setEnabled(False)
            self.shadow.setBlurRadius(0)
            self.shadow.setOffset(0, 0)
        else:
            self.outer_layout.setContentsMargins(10, 10, 10, 10)
            def _enable_shadow():
                self.shadow.setBlurRadius(24)
                self.shadow.setOffset(0, 6)
                self.shadow.setEnabled(True)
            QTimer.singleShot(0, _enable_shadow)

        c = self.themes[self.current_theme]
        self.container.setStyleSheet(self._container_styles(c, radius))
        self.top_bar_widget.set_theme(c, radius)

    def _update_button_sizes(self):
        base = int(min(self.width(), self.height()) * 0.05)
        base = max(12, min(32, base))
        for btn in (self.settings_button, self.notifications_button, self.inventory_button, self.min_button, self.fullsize_button, self.close_button):
            btn.setFixedSize(base, base)
            btn.setIconSize(btn.size())
        self._position_notif_badge()

    def _update_topbar_height(self):
        btn_h = self.min_button.height()
        self.top_bar_widget.setMinimumHeight(int(btn_h + 12))
        self.top_bar_widget.setMaximumHeight(int(btn_h + 14))

    def _update_topbar_spacing(self):
        gap = max(4, min(12, int(self.min_button.width() * 0.25)))
        self.top_bar_layout.setSpacing(gap)

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

    def closeEvent(self, event):
        self.settings.setValue("geometry", self.saveGeometry())
        self.settings.setValue("maximized", self.isMaximized())
        self.settings.setValue("splitterSizes", self.splitter.sizes())
        self._log.info("Saved geometry/maximized/splitter. Closing.")
        self.player.close()
        super().closeEvent(event)

# -------------------- 메인 엔트리 --------------------
if __name__ == "__main__":
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    args = parse_args()
    logger = setup_logging(args.debug or (os.getenv("APP_DEBUG") == "1"))
    logger.info("Starting %s (debug=%s)", APP_NAME, args.debug)

    VIDEO_PATH = args.video
    RESULT_JSONL = args.result_jsonl

    missing = []
    if not VIDEO_PATH or not Path(VIDEO_PATH).is_file():
        missing.append(f"영상(mp4): {VIDEO_PATH or '(비어 있음)'}")
    if not RESULT_JSONL or not Path(RESULT_JSONL).is_file():
        missing.append(f"JSONL: {RESULT_JSONL or '(비어 있음)'}")

    app = QApplication(sys.argv)
    if missing:
        QMessageBox.critical(None, "경로 오류", "다음 파일을 찾지 못했습니다:\n\n" + "\n".join(missing))
        sys.exit(2)

    logger.info("Using video=%s", VIDEO_PATH)
    logger.info("Using results=%s", RESULT_JSONL)

    w = FramelessWindow(video_path=VIDEO_PATH, result_jsonl=RESULT_JSONL, ratio=(16,9), decisions_jsonl=args.decisions_jsonl)
    if w.settings.value("maximized", False, type=bool):
        w.showMaximized()
    else:
        w.show()
    sys.exit(app.exec_())
