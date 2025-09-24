import sys, os, time, logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
import cv2

from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QHBoxLayout, QVBoxLayout, QGraphicsDropShadowEffect,
    QLabel, QSplitter, QTableWidget, QTableWidgetItem, QAbstractItemView, QHeaderView,
    QSizePolicy, QMenu, QAction, QActionGroup, QInputDialog, QMessageBox
)
from PyQt5.QtGui import QIcon, QImage, QPixmap
from PyQt5.QtCore import Qt, QPoint, QRect, QTimer, QDateTime, QEvent, QSettings

APP_ORG = "2025-AI-Project"
APP_NAME = "StorageMonitorUI"
LOG_DIR = Path.home() / ".storage_monitor_ui"

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
                self.title_label.setText(f"{self.title} (not found)")
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

    def __init__(self, face_source=0, item_source=1, ratio=(16, 9)):
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
        self.theme_menu = QMenu("Theme", self.settings_menu)
        self.action_light = QAction("Light", self, checkable=True)
        self.action_dark = QAction("Dark", self, checkable=True)
        g_theme = QActionGroup(self); g_theme.setExclusive(True)
        g_theme.addAction(self.action_light); g_theme.addAction(self.action_dark)
        self.theme_menu.addAction(self.action_light)
        self.theme_menu.addAction(self.action_dark)

        # Aspect Ratio submenu (presets + Custom…)
        self.ratio_menu = QMenu("Aspect Ratio", self.settings_menu)
        self.action_ratio_169 = QAction("16:9", self, checkable=True)
        self.action_ratio_43  = QAction("4:3",  self, checkable=True)
        self.action_ratio_11  = QAction("1:1",  self, checkable=True)
        self.action_ratio_219 = QAction("21:9", self, checkable=True)
        self.action_ratio_custom = QAction("Custom…", self)
        g_ratio = QActionGroup(self); g_ratio.setExclusive(True)
        for a in (self.action_ratio_169, self.action_ratio_43, self.action_ratio_11, self.action_ratio_219):
            g_ratio.addAction(a); self.ratio_menu.addAction(a)
        self.ratio_menu.addSeparator()
        self.ratio_menu.addAction(self.action_ratio_custom)

        # Debug submenu (unchanged items, just not nested under Theme anymore)
        self.debug_menu = QMenu("Debug", self.settings_menu)
        self.action_debug_logs = QAction("Enable debug logging", self, checkable=True)
        self.action_show_fps  = QAction("Show FPS overlay", self, checkable=True)
        self.action_restart_cams = QAction("Restart cameras", self)
        self.action_dump_settings = QAction("Dump settings to log", self)
        self.action_notif_test = QAction("Send test notification", self)
        self.debug_menu.addAction(self.action_notif_test)
        self.action_notif_test.triggered.connect(
            lambda: self.push_notification("Test alert: suspicious activity detected")
        )
        self.debug_menu.addAction(self.action_debug_logs)
        self.debug_menu.addAction(self.action_show_fps)
        self.debug_menu.addSeparator()
        self.debug_menu.addAction(self.action_restart_cams)
        self.debug_menu.addAction(self.action_dump_settings)

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
        self.action_restart_cams.triggered.connect(self._on_restart_cameras)
        self.action_dump_settings.triggered.connect(self._on_dump_settings)


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
        self._notif_title = QAction("Notifications", self)
        self._notif_title.setEnabled(False)
        self._notif_separator = self.notifications_menu.addSeparator()  # anchor to insert above
        self.notifications_menu.insertAction(self._notif_separator, self._notif_title)

        self.action_notif_mark_read = QAction("Mark all as read", self)
        self.action_notif_clear = QAction("Clear all", self)
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


        # --- Top bar (draggable in empty area) ---
        self.top_bar_widget = TopBar(self)
        self.top_bar_layout = QHBoxLayout(self.top_bar_widget)
        self.top_bar_layout.setContentsMargins(8, 6, 6, 6)
        self.top_bar_layout.addWidget(self.settings_button)
        self.top_bar_layout.addWidget(self.notifications_button)
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

            #settingsBtn:hover, #minBtn:hover, #fullBtn:hover, #notifBtn:hover {{
                background-color: {c['min_hover']};
            }}
            #settingsBtn:pressed, #minBtn:pressed, #fullBtn:pressed, #notifBtn:pressed {{
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
        self.face_cam.set_theme(c)
        self.item_cam.set_theme(c)
        self._log.info("Applied theme: %s", name)

    def _apply_ratio(self, ratio_tuple, update_menu_checks=False):
        rw, rh = ratio_tuple
        # apply to both cameras
        self.face_cam.set_aspect_ratio(rw, rh)
        self.item_cam.set_aspect_ratio(rw, rh)
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
        self.face_cam.set_show_fps(enabled)
        self.item_cam.set_show_fps(enabled)

    def _on_restart_cameras(self):
        self._log.info("Restarting both cameras by user request.")
        self.face_cam.restart()
        self.item_cam.restart()

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
        self._log.debug("Log row: %s | %s | %s | %s", ts, user, action, item)

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
        for btn in (self.settings_button, self.notifications_button, self.min_button, self.fullsize_button, self.close_button):
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
        # Stop cameras
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

 # Settings are stored under org="2025-AI-Project", app="StorageMonitorUI". Change those two strings if you want a different storage key. (At the top of the code lol)

 # <a target="_blank" href="https://icons8.com/icon/43725/cancel">Cancel</a> icon by <a target="_blank" href="https://icons8.com">Icons8</a>
