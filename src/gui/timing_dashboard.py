"""
F1 Live Timing Dashboard — A standalone PySide6 window that connects to the
telemetry stream (localhost:9999) and displays a professional F1-style live
timing tower during race replay.

Can be launched standalone:  python -m src.gui.timing_dashboard
Or via the main app:        python main.py --viewer --year 2025 --round 12 --timing
"""

import sys
from collections import deque
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QTableWidget, QTableWidgetItem, QStatusBar, QHeaderView,
    QAbstractItemView, QSizePolicy
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFont, QColor, QBrush

from src.services.stream import TelemetryStreamClient
from src.lib.tyres import get_tyre_compound_str


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Track status code → (display name, hex colour)
TRACK_STATUS_MAP = {
    "1": ("GREEN FLAG", "#00C853"),
    "2": ("YELLOW FLAG", "#FFD600"),
    "4": ("SAFETY CAR", "#FF6D00"),
    "5": ("RED FLAG", "#FF1744"),
    "6": ("VSC", "#FF9100"),
    "7": ("VSC ENDING", "#FF9100"),
    "GREEN": ("GREEN FLAG", "#00C853"),
    "YELLOW": ("YELLOW FLAG", "#FFD600"),
    "SC": ("SAFETY CAR", "#FF6D00"),
    "VSC": ("VSC", "#FF9100"),
    "RED": ("RED FLAG", "#FF1744"),
}
DEFAULT_TRACK_STATUS = ("GREEN FLAG", "#00C853")

# Tyre compound int → (hex colour, single-letter label)
TYRE_COLORS = {
    0: ("#FF3333", "S"),   # SOFT
    1: ("#FFD700", "M"),   # MEDIUM
    2: ("#FFFFFF", "H"),   # HARD
    3: ("#00CC00", "I"),   # INTERMEDIATE
    4: ("#0088FF", "W"),   # WET
}

# Static team colours keyed by 3-letter driver code (2024/2025 grid).
# Falls back to DEFAULT_TEAM_COLOR for unknown codes.
TEAM_COLORS = {
    # Red Bull
    "VER": "#3671C6", "PER": "#3671C6",
    # Ferrari
    "LEC": "#E80020", "SAI": "#E80020", "HAM": "#E80020",
    # McLaren
    "NOR": "#FF8000", "PIA": "#FF8000",
    # Mercedes
    "RUS": "#27F4D2", "ANT": "#27F4D2",
    # Aston Martin
    "ALO": "#229971", "STR": "#229971",
    # Alpine
    "GAS": "#00A0DE", "DOO": "#00A0DE", "OCO": "#00A0DE",
    # RB / VCARB
    "TSU": "#6692FF", "HAD": "#6692FF", "LAW": "#6692FF", "RIC": "#6692FF",
    # Kick Sauber
    "BOT": "#52E252", "ZHO": "#52E252", "BEA": "#52E252",
    # Williams
    "ALB": "#1868DB", "SAR": "#1868DB", "COL": "#1868DB",
    # Haas
    "MAG": "#B6BABD", "HUL": "#B6BABD", "BEA": "#52E252",
}
DEFAULT_TEAM_COLOR = "#888888"

# Gap calculation tuning
MIN_SPEED_FOR_GAP = 10.0      # km/h – below this the driver is likely in pits
GAP_SMOOTHING_WINDOW = 5      # number of recent values to average
MAX_REASONABLE_GAP = 120.0    # seconds – clamp above this
REF_SPEED_MS = 55.56          # 200 km/h in m/s – constant reference speed (matches main window)
DEFAULT_TRACK_LENGTH = 5000.0 # metres – fallback until estimated from data
SECONDS_BEFORE_RETIRED = 30.0 # seconds at near-zero speed before marking driver as retired (OUT)

# UI refresh rate (Hz) – we receive data at ~25 FPS but only repaint this often
UI_REFRESH_HZ = 5

# How many frames a driver can be absent before marking as OUT
FRAMES_BEFORE_OUT = 50

# Column indices
COL_POS = 0
COL_TEAM = 1
COL_DRIVER = 2
COL_GAP = 3
COL_INT = 4
COL_LAST = 5
COL_TYRE = 6
COL_AGE = 7
COL_SPEED = 8
COL_DRS = 9
COL_PIT = 10

COLUMN_HEADERS = ["P", "", "Driver", "Gap", "Int", "Last", "Tyre", "Age", "Spd", "DRS", "Pit"]
COLUMN_WIDTHS = [30, 6, 55, 75, 75, 90, 40, 35, 50, 35, 30]

def _mono_font(size: int, bold: bool = False) -> QFont:
    """Create a cross-platform monospace font.

    Tries Consolas (Windows) first, then Menlo (macOS); Qt's StyleHint
    guarantees a monospace fallback on any platform.
    """
    font = QFont("Consolas", size)
    font.setStyleHint(QFont.Monospace)
    if bold:
        font.setBold(True)
    return font


DARK_STYLESHEET = """
    QMainWindow {
        background-color: #1a1a2e;
    }
    QWidget#header {
        background-color: #0f0f1a;
    }
    QTableWidget {
        background-color: #16213e;
        alternate-background-color: #1a2744;
        color: #e0e0e0;
        border: none;
        font-size: 13px;
        gridline-color: transparent;
    }
    QTableWidget::item {
        padding: 2px 6px;
        border-bottom: 1px solid #2a2a4a;
    }
    QHeaderView::section {
        background-color: #0f3460;
        color: #ffffff;
        font-weight: bold;
        font-size: 11px;
        border: none;
        padding: 4px;
    }
    QLabel {
        color: #e0e0e0;
    }
    QStatusBar {
        background-color: #0f0f1a;
        color: #aaaaaa;
    }
"""


# ---------------------------------------------------------------------------
# DriverState – tracks per-driver derived state across frames
# ---------------------------------------------------------------------------

class DriverState:
    """Tracks derived state for a single driver across telemetry frames."""

    def __init__(self, code: str):
        self.code = code

        # Lap tracking
        self.current_lap = 0
        self.last_lap_time = None        # seconds
        self.best_lap_time = None        # seconds
        self.lap_start_time = None       # elapsed seconds when current lap started
        self.prev_lap_start_time = None  # elapsed seconds when previous lap started

        # Tyre / pit tracking
        self.last_tyre_compound = None   # int compound id
        self.pit_stop_count = 0

        # Status flags
        self.is_in_pit = False
        self.is_out = False
        self.last_seen_frame = 0
        self.slow_since_time = None  # elapsed time when speed first dropped below threshold

        # Smoothing buffers
        self._gap_buf = deque(maxlen=GAP_SMOOTHING_WINDOW)
        self._int_buf = deque(maxlen=GAP_SMOOTHING_WINDOW)

    # ---- lap detection ----

    def update_lap(self, new_lap: int, elapsed_time: float,
                   dist: float = 0.0, speed: float = 0.0):
        """Call every frame. Detects lap transitions and computes lap time."""
        if self.current_lap == 0:
            # First time we see this driver — estimate when their current lap
            # started based on how far they are into it. This lets us produce
            # a reasonable lap time at the very first finish-line crossing
            # (e.g. after the user fast-forwards into the middle of a race).
            self.current_lap = new_lap
            speed_ms = speed / 3.6
            if dist > 100 and speed_ms > 10:
                self.lap_start_time = elapsed_time - (dist / speed_ms)
            else:
                self.lap_start_time = elapsed_time
            return

        if new_lap > self.current_lap:
            self.prev_lap_start_time = self.lap_start_time
            self.lap_start_time = elapsed_time

            if self.prev_lap_start_time is not None:
                lap_time = elapsed_time - self.prev_lap_start_time
                # Sanity filter: realistic F1 lap times are 30s–300s
                if 30.0 < lap_time < 300.0:
                    self.last_lap_time = lap_time
                    if self.best_lap_time is None or lap_time < self.best_lap_time:
                        self.best_lap_time = lap_time

            self.current_lap = new_lap

    # ---- tyre / pit detection ----

    def update_tyre(self, compound: int):
        """Detect compound changes to count pit stops."""
        if compound < 0:
            return
        if self.last_tyre_compound is not None and compound != self.last_tyre_compound:
            self.pit_stop_count += 1
        self.last_tyre_compound = compound

    # ---- gap smoothing ----

    def smoothed_gap(self, raw: float) -> float:
        self._gap_buf.append(raw)
        return sum(self._gap_buf) / len(self._gap_buf)

    def smoothed_interval(self, raw: float) -> float:
        self._int_buf.append(raw)
        return sum(self._int_buf) / len(self._int_buf)


# ---------------------------------------------------------------------------
# TimingDashboard – main window
# ---------------------------------------------------------------------------

class TimingDashboard(QMainWindow):
    """Professional F1-style live timing tower consuming the telemetry stream."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("F1 Race Replay – Live Timing")
        self.setGeometry(100, 100, 900, 750)
        self.setMinimumSize(700, 400)

        # Telemetry client (reuses existing infrastructure)
        self.client = TelemetryStreamClient()
        self.client.data_received.connect(self._on_data_received)
        self.client.connection_status.connect(self._on_connection_status)
        self.client.error_occurred.connect(self._on_error)

        # State
        self.driver_states: dict[str, DriverState] = {}
        self.message_count = 0
        self._latest_data = None          # most recent frame payload
        self._overall_best_lap = None     # float seconds
        self._estimated_track_length = DEFAULT_TRACK_LENGTH

        # Build UI
        self._setup_ui()
        self.setStyleSheet(DARK_STYLESHEET)

        # UI refresh timer (decouple from 25 FPS stream)
        self._refresh_timer = QTimer(self)
        self._refresh_timer.setInterval(int(1000 / UI_REFRESH_HZ))
        self._refresh_timer.timeout.connect(self._refresh_ui)
        self._refresh_timer.start()

        # Start client
        self.client.start()

    # ------------------------------------------------------------------ UI
    # ------------------------------------------------------------------

    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        # --- Header ---
        root_layout.addWidget(self._create_header())

        # --- Timing table ---
        self.table = self._create_timing_table()
        root_layout.addWidget(self.table)

        # --- Status bar ---
        self._create_status_bar()

    def _create_header(self) -> QWidget:
        header = QWidget()
        header.setObjectName("header")
        header.setFixedHeight(70)
        layout = QHBoxLayout(header)
        layout.setContentsMargins(16, 8, 16, 8)

        mono = _mono_font(11)
        mono_large = _mono_font(20, bold=True)

        # Left: session time
        left = QVBoxLayout()
        self.session_time_label = QLabel("--:--:--")
        self.session_time_label.setFont(mono_large)
        left.addWidget(self.session_time_label)

        self.lap_label = QLabel("LAP -- / --")
        self.lap_label.setFont(mono)
        left.addWidget(self.lap_label)
        layout.addLayout(left)

        layout.addStretch()

        # Centre: track status
        self.track_status_label = QLabel("GREEN FLAG")
        self.track_status_label.setAlignment(Qt.AlignCenter)
        self.track_status_label.setFont(_mono_font(12, bold=True))
        self.track_status_label.setFixedHeight(36)
        self.track_status_label.setMinimumWidth(180)
        self._set_track_status("GREEN")
        layout.addWidget(self.track_status_label)

        layout.addStretch()

        # Right: playback info
        right = QVBoxLayout()
        right.setAlignment(Qt.AlignRight)
        self.speed_label = QLabel("1.0x")
        self.speed_label.setFont(mono)
        self.speed_label.setAlignment(Qt.AlignRight)
        right.addWidget(self.speed_label)

        self.state_label = QLabel("PLAYING")
        self.state_label.setFont(_mono_font(11, bold=True))
        self.state_label.setAlignment(Qt.AlignRight)
        right.addWidget(self.state_label)
        layout.addLayout(right)

        return header

    def _create_timing_table(self) -> QTableWidget:
        table = QTableWidget(0, len(COLUMN_HEADERS))
        table.setHorizontalHeaderLabels(COLUMN_HEADERS)
        table.setAlternatingRowColors(True)

        # Sizing
        h_header = table.horizontalHeader()
        for i, w in enumerate(COLUMN_WIDTHS):
            if i == COL_DRIVER:
                h_header.setSectionResizeMode(i, QHeaderView.Stretch)
            else:
                table.setColumnWidth(i, w)
                h_header.setSectionResizeMode(i, QHeaderView.Fixed)

        table.verticalHeader().setVisible(False)
        table.verticalHeader().setDefaultSectionSize(28)
        table.setShowGrid(False)
        table.setSelectionMode(QAbstractItemView.NoSelection)
        table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        table.setFocusPolicy(Qt.NoFocus)

        # Monospace for data readability
        table.setFont(_mono_font(12))

        return table

    def _create_status_bar(self):
        bar = QStatusBar()
        self.setStatusBar(bar)

        self.connection_label = QLabel("Disconnected")
        self.frame_label = QLabel("Frame: --")
        self.messages_label = QLabel("Messages: 0")

        bar.addPermanentWidget(self.connection_label)
        bar.addPermanentWidget(self.frame_label)
        bar.addPermanentWidget(self.messages_label)

    # --------------------------------------------------------- data slots
    # ---------------------------------------------------------

    def _on_data_received(self, data: dict):
        """Process driver state on EVERY frame, store latest for UI refresh."""
        self.message_count += 1
        self._latest_data = data

        # Update driver state machine on every incoming frame so we never
        # miss a lap transition — even during fast-forward.  The heavy UI
        # repaint still happens at UI_REFRESH_HZ via the timer.
        frame = data.get("frame")
        if frame:
            elapsed = frame.get("t", 0.0)
            frame_index = data.get("frame_index", 0)
            self._update_driver_states(frame, elapsed, frame_index)

        # Seed lap times from broadcast's session data (handles fast-forward
        # where we never saw the lap transition ourselves).
        broadcast_lap_times = data.get("driver_lap_times", {})
        for code, lt in broadcast_lap_times.items():
            st = self.driver_states.get(code)
            if st is not None and lt is not None and 30.0 < lt < 300.0:
                st.last_lap_time = lt
                if st.best_lap_time is None or lt < st.best_lap_time:
                    st.best_lap_time = lt

    def _on_connection_status(self, status: str):
        self.connection_label.setText(f"Status: {status}")
        if status == "Connected":
            self.connection_label.setStyleSheet("color: #00C853; font-weight: bold;")
        elif status == "Connecting...":
            self.connection_label.setStyleSheet("color: #FF9100; font-weight: bold;")
        else:
            self.connection_label.setStyleSheet("color: #FF1744; font-weight: bold;")

    def _on_error(self, msg: str):
        self.statusBar().showMessage(f"Error: {msg}", 5000)

    # -------------------------------------------------------- refresh logic
    # --------------------------------------------------------

    def _refresh_ui(self):
        """Called by QTimer at UI_REFRESH_HZ to repaint from latest data."""
        data = self._latest_data
        if data is None:
            return

        frame = data.get("frame")

        # Driver state is already updated per-frame in _on_data_received;
        # here we just refresh the visual table at the slower UI rate.

        # Calculate gaps and refresh table
        drivers_data = frame.get("drivers", {}) if frame else {}
        sorted_entries = self._calculate_gaps(drivers_data)

        # Compute leader lap from our progress-based ordering
        # (the broadcast's session_data.lap is unreliable — see race_replay.py line 245)
        leader_lap = int(round(sorted_entries[0]["lap"])) if sorted_entries else None

        # Update header with corrected leader lap
        self._update_header(data, leader_lap)

        self._refresh_timing_tower(sorted_entries)

        # Status bar
        frame_index = data.get("frame_index", 0)
        total = data.get("total_frames", "?")
        self.frame_label.setText(f"Frame: {frame_index} / {total}")
        self.messages_label.setText(f"Messages: {self.message_count}")

    # ---------------------------------------------------- state management
    # ----------------------------------------------------

    def _update_driver_states(self, frame: dict, elapsed: float, frame_index: int):
        if not frame or "drivers" not in frame:
            return

        active_codes = set(frame["drivers"].keys())

        for code, d in frame["drivers"].items():
            if code not in self.driver_states:
                self.driver_states[code] = DriverState(code)
            st = self.driver_states[code]
            st.last_seen_frame = frame_index

            # Lap detection
            new_lap = int(round(d.get("lap", 0)))
            st.update_lap(new_lap, elapsed,
                          dist=d.get("dist", 0.0), speed=d.get("speed", 0.0))

            # Tyre / pit
            compound = int(round(d.get("tyre", -1)))
            st.update_tyre(compound)

            # Pit / retired detection based on speed duration
            speed = d.get("speed", 0.0)
            if st.is_out:
                # Once retired, only recover if they reach real racing speed
                if speed > 50.0:
                    st.is_out = False
                    st.slow_since_time = None
                    st.is_in_pit = False
            elif speed < MIN_SPEED_FOR_GAP:
                if st.slow_since_time is None:
                    st.slow_since_time = elapsed
                # Stationary for too long → retired, not pitting
                if elapsed - st.slow_since_time > SECONDS_BEFORE_RETIRED:
                    st.is_out = True
                    st.is_in_pit = False
                else:
                    st.is_in_pit = True
            else:
                st.slow_since_time = None
                st.is_in_pit = False

        # Mark drivers missing for too long as OUT
        for code, st in self.driver_states.items():
            if code not in active_codes and (frame_index - st.last_seen_frame) > FRAMES_BEFORE_OUT:
                st.is_out = True

    # --------------------------------------------------- gap calculation
    # ---------------------------------------------------

    def _calculate_gaps(self, drivers_data: dict) -> list[dict]:
        if not drivers_data:
            return []

        track_len = self._estimated_track_length

        # --- Estimate track length from dist / rel_dist ---
        # Use drivers mid-lap (0.1 < rel_dist < 0.9) for a reliable estimate.
        track_len_samples = []
        for d in drivers_data.values():
            rel = d.get("rel_dist", 0.0)
            dist = d.get("dist", 0.0)
            if 0.1 < rel < 0.9 and dist > 100:
                track_len_samples.append(dist / rel)
        if track_len_samples:
            self._estimated_track_length = sum(track_len_samples) / len(track_len_samples)
            track_len = self._estimated_track_length

        # --- Build entries with lap-aware progress ---
        entries = []
        for code, d in drivers_data.items():
            st = self.driver_states.get(code)
            lap = d.get("lap", 0)
            dist = d.get("dist", 0.0)

            # Total progress since race start (metres).
            # dist is per-lap (resets each lap), so we add completed laps.
            progress = (max(int(round(lap)), 1) - 1) * track_len + dist

            entries.append({
                "code": code,
                "position": d.get("position", 99),
                "progress": progress,
                "speed": d.get("speed", 0.0),
                "lap": lap,
                "tyre": int(round(d.get("tyre", -1))),
                "tyre_life": d.get("tyre_life", 0),
                "drs": d.get("drs", 0),
                "is_out": st.is_out if st else False,
                "is_in_pit": st.is_in_pit if st else False,
            })

        # Sort by progress descending — leader has the most progress.
        # This naturally handles lapped cars (they have less progress).
        entries.sort(key=lambda e: e["progress"], reverse=True)
        if not entries:
            return entries

        leader_progress = entries[0]["progress"]

        for i, entry in enumerate(entries):
            st = self.driver_states.get(entry["code"])
            # Assign position from progress-based ordering
            entry["position"] = i + 1

            if i == 0:
                entry["gap_display"] = "LEADER"
                entry["int_display"] = ""
                continue

            if entry["is_out"]:
                entry["gap_display"] = "OUT"
                entry["int_display"] = "OUT"
                continue

            # Gap to leader using constant reference speed (matches main window)
            progress_diff = max(0.0, leader_progress - entry["progress"])
            raw_gap = min(progress_diff / REF_SPEED_MS, MAX_REASONABLE_GAP)
            gap = st.smoothed_gap(raw_gap) if st else raw_gap

            # How many full laps behind the leader
            laps_behind = int(progress_diff / track_len) if track_len > 0 else 0

            # Interval to car directly ahead
            ahead_progress = entries[i - 1]["progress"]
            int_diff = max(0.0, ahead_progress - entry["progress"])
            raw_int = min(int_diff / REF_SPEED_MS, MAX_REASONABLE_GAP)
            interval = st.smoothed_interval(raw_int) if st else raw_int

            if entry["is_in_pit"]:
                entry["gap_display"] = "PIT"
                entry["int_display"] = "PIT"
            else:
                # Show "+NL" for lapped cars' gap to leader
                entry["gap_display"] = self._format_gap(gap, laps_behind)
                entry["int_display"] = self._format_gap(interval)

        return entries

    # ------------------------------------------------- table rendering
    # -------------------------------------------------

    def _refresh_timing_tower(self, entries: list[dict]):
        row_count = len(entries)

        # Resize table if driver count changed
        if self.table.rowCount() != row_count:
            self.table.setRowCount(row_count)

        # Track overall best lap
        self._overall_best_lap = None
        for st in self.driver_states.values():
            if st.best_lap_time is not None:
                if self._overall_best_lap is None or st.best_lap_time < self._overall_best_lap:
                    self._overall_best_lap = st.best_lap_time

        white = QBrush(QColor("#e0e0e0"))
        dim = QBrush(QColor("#666666"))

        for row, entry in enumerate(entries):
            code = entry["code"]
            st = self.driver_states.get(code)
            is_out = entry.get("is_out", False)
            text_brush = dim if is_out else white

            # Position
            self._set_cell(row, COL_POS, str(entry["position"]), brush=text_brush,
                           align=Qt.AlignCenter)

            # Team colour bar (thin coloured cell)
            team_col = TEAM_COLORS.get(code, DEFAULT_TEAM_COLOR)
            item = self._set_cell(row, COL_TEAM, "", brush=text_brush)
            item.setBackground(QBrush(QColor(team_col)))

            # Driver code
            self._set_cell(row, COL_DRIVER, code, brush=text_brush)

            # Gap to leader
            self._set_cell(row, COL_GAP, entry.get("gap_display", "--"),
                           brush=text_brush, align=Qt.AlignRight | Qt.AlignVCenter)

            # Interval
            self._set_cell(row, COL_INT, entry.get("int_display", "--"),
                           brush=text_brush, align=Qt.AlignRight | Qt.AlignVCenter)

            # Last lap time
            lap_text, lap_brush = self._lap_time_display(st)
            self._set_cell(row, COL_LAST, lap_text,
                           brush=lap_brush if not is_out else dim,
                           align=Qt.AlignRight | Qt.AlignVCenter)

            # Tyre compound
            tyre_int = entry.get("tyre", -1)
            tyre_colour, tyre_letter = TYRE_COLORS.get(tyre_int, ("#888888", "?"))
            self._set_cell(row, COL_TYRE, tyre_letter,
                           brush=QBrush(QColor(tyre_colour)) if not is_out else dim,
                           align=Qt.AlignCenter)

            # Tyre age
            age = entry.get("tyre_life", 0)
            age_text = str(int(round(age))) if not is_out else "--"
            self._set_cell(row, COL_AGE, age_text, brush=text_brush, align=Qt.AlignCenter)

            # Speed
            speed = entry.get("speed", 0)
            speed_text = str(int(round(speed))) if not is_out else "--"
            self._set_cell(row, COL_SPEED, speed_text, brush=text_brush,
                           align=Qt.AlignRight | Qt.AlignVCenter)

            # DRS indicator
            drs_val = entry.get("drs", 0)
            drs_active = drs_val >= 10
            drs_text = "\u25CF" if drs_active else ""  # filled circle
            drs_brush = QBrush(QColor("#00E676")) if drs_active else text_brush
            self._set_cell(row, COL_DRS, drs_text, brush=drs_brush, align=Qt.AlignCenter)

            # Pit stop count
            pits = st.pit_stop_count if st else 0
            self._set_cell(row, COL_PIT, str(pits) if pits > 0 else "",
                           brush=text_brush, align=Qt.AlignCenter)

    def _set_cell(self, row: int, col: int, text: str, *,
                  brush: QBrush = None, align: int = Qt.AlignLeft | Qt.AlignVCenter) -> QTableWidgetItem:
        """Set or reuse a table cell item."""
        item = self.table.item(row, col)
        if item is None:
            item = QTableWidgetItem()
            self.table.setItem(row, col, item)
        item.setText(text)
        item.setTextAlignment(align)
        if brush is not None:
            item.setForeground(brush)
        return item

    def _lap_time_display(self, st: DriverState | None) -> tuple[str, QBrush]:
        """Return (text, colour brush) for a driver's last lap time."""
        if st is None or st.last_lap_time is None:
            return ("--", QBrush(QColor("#e0e0e0")))

        text = self._format_lap_time(st.last_lap_time)

        # Purple for overall best, green for personal best, white otherwise
        if self._overall_best_lap is not None and abs(st.last_lap_time - self._overall_best_lap) < 0.001:
            return (text, QBrush(QColor("#BB00FF")))  # purple
        if st.best_lap_time is not None and abs(st.last_lap_time - st.best_lap_time) < 0.001:
            return (text, QBrush(QColor("#00E676")))   # green
        return (text, QBrush(QColor("#e0e0e0")))

    # ----------------------------------------------- header update
    # -----------------------------------------------

    def _update_header(self, data: dict, leader_lap: int = None):
        session = data.get("session_data", {})

        # Session time
        self.session_time_label.setText(session.get("time", "--:--:--"))

        # Lap counter — use our progress-based leader lap, not the stream's
        lap = leader_lap if leader_lap is not None else session.get("lap", "--")
        total = session.get("total_laps", "--")
        self.lap_label.setText(f"LAP {lap} / {total}")

        # Track status
        self._set_track_status(data.get("track_status", "GREEN"))

        # Playback speed & state
        speed = data.get("playback_speed", 1.0)
        self.speed_label.setText(f"{speed}x")

        paused = data.get("is_paused", False)
        if paused:
            self.state_label.setText("PAUSED")
            self.state_label.setStyleSheet("color: #FF1744; font-weight: bold;")
        else:
            self.state_label.setText("PLAYING")
            self.state_label.setStyleSheet("color: #00C853; font-weight: bold;")

    def _set_track_status(self, code: str):
        name, colour = TRACK_STATUS_MAP.get(str(code), DEFAULT_TRACK_STATUS)
        self.track_status_label.setText(name)
        self.track_status_label.setStyleSheet(
            f"background-color: {colour}; color: #000000; font-weight: bold; "
            f"border-radius: 4px; padding: 4px 12px;"
        )

    # -------------------------------------------- formatting helpers
    # --------------------------------------------

    @staticmethod
    def _format_gap(seconds: float, laps_behind: int = 0) -> str:
        if seconds <= 0.001:
            return "--"
        if laps_behind >= 1:
            return f"+{laps_behind}L"
        if seconds >= 60.0:
            mins = int(seconds // 60)
            secs = seconds % 60
            return f"+{mins}:{secs:04.1f}"
        return f"+{seconds:.1f}"

    @staticmethod
    def _format_lap_time(seconds: float) -> str:
        if seconds is None or seconds <= 0:
            return "--"
        mins = int(seconds // 60)
        secs = seconds % 60
        if mins > 0:
            return f"{mins}:{secs:06.3f}"
        return f"{secs:.3f}"

    # ------------------------------------------------- lifecycle
    # -------------------------------------------------

    def closeEvent(self, event):
        if self.client.isRunning():
            self.client.stop()
            self.client.wait()
        event.accept()


# ---------------------------------------------------------------------------
# Module entry point
# ---------------------------------------------------------------------------

def main():
    app = QApplication(sys.argv)
    app.setApplicationName("F1 Live Timing")
    window = TimingDashboard()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
