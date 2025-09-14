import sys
import cv2
from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QHBoxLayout, QVBoxLayout, QGraphicsDropShadowEffect,
    QLabel, QSplitter, QTableWidget, QTableWidgetItem, QAbstractItemView, QHeaderView,
    QSizePolicy, QMenu, QAction, QActionGroup
)
from PyQt5.QtGui import QIcon, QImage, QPixmap
from PyQt5.QtCore import Qt, QPoint, QRect, QTimer, QDateTime, QEvent, QSettings

# --- Draggable top bar (only empty area is draggable; buttons stay clickable) ---
class TopBar(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self._drag_active = False
        self._drag_pos = QPoint()
        self.setObjectName("topBar")
        self.setAttribute(Qt.WA_StyledBackground, True)

    def set_theme(self, colors):
        # Keeps your bottom divider line
        self.setStyleSheet(f"""
            #topBar {{
                background-color: {colors['topbar']};
                border-top-left-radius: 15px;
                border-top-right-radius: 15px;
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

        self.view = QLabel()
        self.view.setAlignment(Qt.AlignCenter)
        self.view.setMinimumSize(240, 135)
        self.view.setScaledContents(False)

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

    def set_theme(self, colors):
        self.title_label.setStyleSheet(f"font-weight:600; color:{colors['text']}; padding:6px;")
        self.view.setStyleSheet(
            f"background:{colors['cam_bg']}; border:1px solid {colors['border']}; border-radius:8px;"
        )

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

        # QSettings: org/app names — change if you want different keys/paths
        self.settings = QSettings("2025-AI-Project", "StorageMonitorUI")

        # Default size
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

        outer = QVBoxLayout(self)
        outer.setContentsMargins(10, 10, 10, 10)
        outer.addWidget(self.container)

        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(24)
        shadow.setOffset(0, 6)
        shadow.setColor(Qt.black)
        self.container.setGraphicsEffect(shadow)

        # --- Top bar buttons ---
        # Left: Settings (with menu)
        self.settings_button = QPushButton("")
        self.settings_button.setObjectName("settingsBtn")
        self.settings_button.setIcon(QIcon("settings_button.png"))
        self.settings_button.setCursor(Qt.PointingHandCursor)

        self.theme_menu = QMenu(self.settings_button)
        self.action_light = QAction("Light theme", self, checkable=True)
        self.action_dark = QAction("Dark theme", self, checkable=True)
        group = QActionGroup(self)
        group.setExclusive(True)
        group.addAction(self.action_light)
        group.addAction(self.action_dark)
        self.theme_menu.addAction(self.action_light)
        self.theme_menu.addAction(self.action_dark)
        self.settings_button.setMenu(self.theme_menu)

        self.action_light.triggered.connect(lambda: self.apply_theme('light'))
        self.action_dark.triggered.connect(lambda: self.apply_theme('dark'))

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

        # --- Top bar (draggable in empty area) ---
        self.top_bar_widget = TopBar(self)
        self.top_bar_layout = QHBoxLayout(self.top_bar_widget)
        self.top_bar_layout.setContentsMargins(8, 6, 6, 6)
        self.top_bar_layout.addWidget(self.settings_button)  # left
        self.top_bar_layout.addStretch()
        self.top_bar_layout.addWidget(self.min_button)
        self.top_bar_layout.addWidget(self.fullsize_button)
        self.top_bar_layout.addWidget(self.close_button)

        # --- Content: Cameras + Log ---
        self.face_cam = CameraFeed(face_source, "Face Camera", aspect_ratio=ratio)
        self.item_cam = CameraFeed(item_source, "Item Camera", aspect_ratio=ratio)

        cameras_col = QWidget()
        cam_layout = QVBoxLayout(cameras_col)
        cam_layout.setContentsMargins(8, 8, 8, 8)
        cam_layout.setSpacing(8)
        cam_layout.addWidget(self.face_cam, 1)
        cam_layout.addWidget(self.item_cam, 1)
        cameras_col.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.log_table = QTableWidget(0, 5)
        self.log_table.setObjectName("logBox")
        self.log_table.setHorizontalHeaderLabels(["Time", "User", "Action", "Item", "Count"])
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

        # Splitter: both sides can grow, left a bit heavier
        self.splitter = QSplitter(Qt.Horizontal)
        self.splitter.addWidget(cameras_col)
        self.splitter.addWidget(self.log_table)
        self.splitter.setStretchFactor(0, 2)
        self.splitter.setStretchFactor(1, 1)
        # Persist sizes when user drags the divider
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

        # Load last theme from settings (defaults to 'light')
        saved_theme = self.settings.value("theme", "light")
        self.apply_theme(saved_theme)

        # ---- Restore geometry and splitter sizes ----
        self._restore_geometry_and_splitter()

    # ===== Persistence helpers =====
    def _restore_geometry_and_splitter(self):
        geo = self.settings.value("geometry")
        if geo is not None:
            self.restoreGeometry(geo)

        sizes = self.settings.value("splitterSizes")
        if sizes:
            try:
                sizes = [int(x) for x in list(sizes)]
                if len(sizes) == 2 and sum(sizes) > 0:
                    self.splitter.setSizes(sizes)
            except Exception:
                pass

    def _save_splitter_sizes(self, *_):
        self.settings.setValue("splitterSizes", self.splitter.sizes())

    # ===== Theme application with persistence =====
    def _container_styles(self, c):
        return f"""
            #container {{
                background-color: {c['bg']};
                border-radius: 15px;
                color: {c['text']};
            }}

            QPushButton {{
                border: none;
                background: transparent;
                border-radius: 6px;
                padding: 0 2px;
            }}

            #settingsBtn:hover, #minBtn:hover, #fullBtn:hover {{
                background-color: {c['min_hover']};
            }}
            #settingsBtn:pressed, #minBtn:pressed, #fullBtn:pressed {{
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

        # Persist selection
        self.settings.setValue("theme", name)

        # Update menu check marks
        self.action_light.setChecked(name == 'light')
        self.action_dark.setChecked(name == 'dark')

        # Apply styles
        self.container.setStyleSheet(self._container_styles(c))
        self.top_bar_widget.set_theme(c)
        self.face_cam.set_theme(c)
        self.item_cam.set_theme(c)

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
        for btn in (self.settings_button, self.min_button, self.fullsize_button, self.close_button):
            btn.setFixedSize(base, base)
            btn.setIconSize(btn.size())

    def _update_topbar_height(self):
        btn_h = self.min_button.height()
        self.top_bar_widget.setMinimumHeight(int(btn_h + 12))
        self.top_bar_widget.setMaximumHeight(int(btn_h + 14))

    def _update_topbar_spacing(self):
        gap = max(4, min(12, int(self.min_button.width() * 0.25)))
        self.top_bar_layout.setSpacing(gap)

    def closeEvent(self, event):
        # Save geometry & maximized state
        self.settings.setValue("geometry", self.saveGeometry())
        self.settings.setValue("maximized", self.isMaximized())
        # Save splitter sizes (again, just in case)
        self.settings.setValue("splitterSizes", self.splitter.sizes())
        # Stop cameras cleanly
        self.face_cam.stop()
        self.item_cam.stop()
        super().closeEvent(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)

    default_ratio = (16, 9)
    w = FramelessWindow(face_source=0, item_source=1, ratio=default_ratio)

    # Restore maximized state AFTER creating the window
    if w.settings.value("maximized", False, type=bool):
        w.showMaximized()
    else:
        w.show()

    # Example:
    # w.add_log("Alice", "Deposit", "Box-A", 3)

    sys.exit(app.exec_())

 # Settings are stored under org="2025-AI-Project", app="StorageMonitorUI". Change those two strings if you want a different storage key.

 # <a target="_blank" href="https://icons8.com/icon/43725/cancel">Cancel</a> icon by <a target="_blank" href="https://icons8.com">Icons8</a>blank" href="https://icons8.com">Icons8</a>
