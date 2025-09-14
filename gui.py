import sys
import cv2
from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QHBoxLayout, QVBoxLayout, QGraphicsDropShadowEffect,
    QLabel, QSplitter, QTableWidget, QTableWidgetItem, QAbstractItemView, QHeaderView,
    QSizePolicy
)
from PyQt5.QtGui import QIcon, QImage, QPixmap
from PyQt5.QtCore import Qt, QPoint, QRect, QTimer, QDateTime, QEvent

# Shared border color for camera boxes & log box
BORDER_COLOR = "#e7e7e7"


# --- Draggable top bar (only empty area is draggable; buttons stay clickable) ---
class TopBar(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self._drag_active = False
        self._drag_pos = QPoint()
        self.setObjectName("topBar")
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setStyleSheet("""
            #topBar {
                background-color: #f2f2f2;        /* light grey */
                border-top-left-radius: 15px;
                border-top-right-radius: 15px;
                border-bottom: 1px solid rgba(0,0,0,0.08); /* divider */
            }
        """)

    def mousePressEvent(self, event):
        # start window drag only if clicking empty area (not on a child)
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


# --- Camera widget: resizes with window, image keeps aspect ratio, inside AspectRatioBox ---
class CameraFeed(QWidget):
    def __init__(self, source=0, title="Camera", aspect_ratio=(16, 9)):
        super().__init__()
        self.source = source
        self.title = title
        self.cap = None
        self._last_pixmap = None

        self.title_label = QLabel(title)
        self.title_label.setStyleSheet("font-weight: 600; color: #333; padding: 6px;")

        self.view = QLabel()
        self.view.setAlignment(Qt.AlignCenter)
        self.view.setMinimumSize(240, 135)  # small 16:9-ish minimum
        self.view.setStyleSheet(
            f"background:#fafafa; border:1px solid {BORDER_COLOR}; border-radius:8px;"
        )
        self.view.setScaledContents(False)  # we scale manually with KeepAspectRatio

        # wrap the label in an aspect-ratio box
        self.ar = AspectRatioBox(ratio=aspect_ratio, child=self.view, parent=self)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)
        lay.addWidget(self.title_label)
        lay.addWidget(self.ar, 1)

        # re-render when the inner label changes size
        self.view.installEventFilter(self)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._grab)
        self.start()

    def set_aspect_ratio(self, rw, rh):
        self.ar.set_ratio(rw, rh)
        self._render_scaled()

    def eventFilter(self, obj, e):
        if obj is self.view and e.type() == QEvent.Resize:
            self._render_scaled()
        return super().eventFilter(obj, e)

    def start(self):
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap or not self.cap.isOpened():
            self.title_label.setText(f"{self.title} (not found)")
            return
        self.timer.start(33)  # ~30 FPS

    def stop(self):
        self.timer.stop()
        if self.cap:
            self.cap.release()
            self.cap = None

    def _grab(self):
        if not self.cap:
            return
        ok, frame = self.cap.read()
        if not ok:
            return
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame.shape
        qimg = QImage(frame.data, w, h, ch * w, QImage.Format_RGB888)
        self._last_pixmap = QPixmap.fromImage(qimg)
        self._render_scaled()

    def _render_scaled(self):
        if self._last_pixmap is None:
            return
        target = self.view.size()
        scaled = self._last_pixmap.scaled(target, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.view.setPixmap(scaled)

    def closeEvent(self, e):
        self.stop()
        super().closeEvent(e)


class FramelessWindow(QWidget):
    RESIZE_MARGIN = 8
    MIN_W = 800
    MIN_H = 480

    def __init__(self, face_source=0, item_source=1, ratio=(16, 9)):
        super().__init__()
        self.setWindowFlags(Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)

        # Default size
        self.resize(1000, 700)
        self.setMinimumSize(self.MIN_W, self.MIN_H)

        self._normal_geometry = None  # remember size/pos before fullscreen

        # --- Inner container (white rounded background) ---
        self.container = QWidget(self)
        self.container.setObjectName("container")
        self.container.setAttribute(Qt.WA_StyledBackground, True)
        self.container.setStyleSheet(f"""
            #container {{
                background-color: white;
                border-radius: 15px;
            }}

            QPushButton {{
                border: none;
                background: transparent;
                border-radius: 6px;      /* rounded hover/press background */
                padding: 0 2px;          /* tiny inner padding */
            }}

            #minBtn:hover, #fullBtn:hover {{
                background-color: rgba(0, 0, 0, 0.06);
            }}
            #minBtn:pressed, #fullBtn:pressed {{
                background-color: rgba(0, 0, 0, 0.12);
            }}

            #closeBtn:hover {{ background-color: #e81123; }}
            #closeBtn:pressed {{ background-color: #c50f1f; }}

            /* Log box styling to match camera box border */
            #logBox {{
                border: 1px solid {BORDER_COLOR};
                border-radius: 8px;
                background: #ffffff;
                alternate-background-color: #fafafa;
                gridline-color: {BORDER_COLOR};   /* grid lines color */
            }}
            #logBox::item:selected {{
                background: rgba(0,0,0,0.12);
            }}
            #logBox QHeaderView::section {{
                background: #f7f7f7;
                border: none;
                border-right: 1px solid {BORDER_COLOR};  /* vertical separators in header */
                padding: 6px;
            }}
            #logBox QTableCornerButton::section {{
                background: #f7f7f7;
                border: none;
            }}
        """)

        # Outer margin so the drop shadow isn't clipped
        outer = QVBoxLayout(self)
        outer.setContentsMargins(10, 10, 10, 10)
        outer.addWidget(self.container)

        # Drop shadow
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(24)
        shadow.setOffset(0, 6)
        shadow.setColor(Qt.black)
        self.container.setGraphicsEffect(shadow)

        # --- Buttons ---
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

        # --- Top bar widget (light grey, draggable only in empty area) ---
        self.top_bar_widget = TopBar(self)
        self.top_bar_layout = QHBoxLayout(self.top_bar_widget)
        self.top_bar_layout.setContentsMargins(8, 6, 6, 6)  # L T R B
        self.top_bar_layout.addStretch()
        self.top_bar_layout.addWidget(self.min_button)
        self.top_bar_layout.addWidget(self.fullsize_button)
        self.top_bar_layout.addWidget(self.close_button)

        # --- CONTENT: Left cameras (resizable, ratio-locked) | Right log (resizable) ---
        self.face_cam = CameraFeed(face_source, "Face Camera", aspect_ratio=ratio)
        self.item_cam = CameraFeed(item_source, "Item Camera", aspect_ratio=ratio)

        cameras_col = QWidget()
        cam_layout = QVBoxLayout(cameras_col)
        cam_layout.setContentsMargins(8, 8, 8, 8)
        cam_layout.setSpacing(8)
        cam_layout.addWidget(self.face_cam, 1)
        cam_layout.addWidget(self.item_cam, 1)
        cameras_col.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # Right: log table with Count column
        self.log_table = QTableWidget(0, 5)
        self.log_table.setObjectName("logBox")
        self.log_table.setHorizontalHeaderLabels(["Time", "User", "Action", "Item", "Count"])
        self.log_table.verticalHeader().setVisible(False)
        self.log_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.log_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.log_table.setAlternatingRowColors(True)
        self.log_table.setShowGrid(True)                          # <- show grid
        self.log_table.setGridStyle(Qt.SolidLine)                 # <- solid lines

        header = self.log_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Stretch)
        header.setHighlightSections(False)
        header.setSectionResizeMode(4, QHeaderView.ResizeToContents)  # Count compact

        # Splitter: both sides can grow, left a bit heavier
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(cameras_col)
        splitter.addWidget(self.log_table)
        splitter.setStretchFactor(0, 2)  # left grows more
        splitter.setStretchFactor(1, 1)

        # --- Main layout inside the white container ---
        main_layout = QVBoxLayout(self.container)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        main_layout.addWidget(self.top_bar_widget)
        main_layout.addWidget(splitter, 1)

        # --- Resize/drag state (for window frame edges) ---
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

    # ===== Fullscreen toggle =====
    def toggle_fullscreen(self):
        if self.isFullScreen():
            self.showNormal()
            if self._normal_geometry:
                self.setGeometry(self._normal_geometry)
        else:
            self._normal_geometry = self.geometry()
            self.showFullScreen()
        QTimer.singleShot(0, self._post_layout_fix)

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
            'left': Qt.SizeHorCursor,
            'right': Qt.SizeHorCursor,
            'top': Qt.SizeVerCursor,
            'bottom': Qt.SizeVerCursor,
            'topleft': Qt.SizeFDiagCursor,
            'bottomright': Qt.SizeFDiagCursor,
            'topright': Qt.SizeBDiagCursor,
            'bottomleft': Qt.SizeBDiagCursor,
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
                g.setX(new_x)
                g.setWidth(new_w)
        if 'right' in self._resize_region:
            new_w = g.width() + dx
            if new_w >= self.MIN_W:
                g.setWidth(new_w)
        if 'top' in self._resize_region:
            new_y = g.y() + dy
            new_h = g.height() - dy
            if new_h >= self.MIN_H:
                g.setY(new_y)
                g.setHeight(new_h)
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

    def _update_button_sizes(self):
        base = int(min(self.width(), self.height()) * 0.05)
        base = max(12, min(32, base))
        for btn in (self.min_button, self.fullsize_button, self.close_button):
            btn.setFixedSize(base, base)
            btn.setIconSize(btn.size())

    def _update_topbar_height(self):
        btn_h = self.min_button.height()
        self.top_bar_widget.setMinimumHeight(int(btn_h + 12))
        self.top_bar_widget.setMaximumHeight(int(btn_h + 14))

    def _update_topbar_spacing(self):
        gap = max(4, min(12, int(self.min_button.width() * 0.25)))
        self.top_bar_layout.setSpacing(gap)

    # ===== Cleanup =====
    def closeEvent(self, event):
        self.face_cam.stop()
        self.item_cam.stop()
        super().closeEvent(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)

    # Choose default aspect ratio here (e.g., (16,9), (4,3), (1,1), (21,9), ...)
    default_ratio = (16, 9)

    w = FramelessWindow(face_source=0, item_source=1, ratio=default_ratio)

    # Example: append a test log row
    # w.add_log("Alice", "Deposit", "Box-A", 3)

    w.show()
    sys.exit(app.exec_())

 # <a target="_blank" href="https://icons8.com/icon/43725/cancel">Cancel</a> icon by <a target="_blank" href="https://icons8.com">Icons8</a>