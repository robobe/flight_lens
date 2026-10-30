"""One desktop overview backed by immutable, independently testable flight data."""
import argparse
import csv
from collections import OrderedDict
from collections.abc import Mapping
from datetime import datetime, timezone
import json
from pathlib import Path
from threading import Event as CancelEvent
import time

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg
import yaml

from .reader import Cancelled, load_flight, read_frame
from .model import series
from .video import export_mp4
from .assistant import LlamaCppBackend, McapAnalysisTools, McapAssistant
from .ttc_analysis import (TtcParameters, butterworth_filter, compare_to_hit_time,
                           low_pass_filter, recalculate)

COLORS = ['#58b8f5', '#eab85a', '#70d6ad', '#c299f0']
ATTITUDE_MEASURED_COLOR = '#58b8f5'
ATTITUDE_ESTIMATED_COLOR = '#ff8282'
MSP_MEASURED_COLOR = '#70d6ad'
MSP_ESTIMATED_COLOR = '#c299f0'
VISUAL_RAW_COLOR = '#eab85a'
VISUAL_FILTERED_COLOR = '#70d6ad'
TTC_DISPLAY_PERCENTILE = 95.
TTC_Y_AXIS_RANGES_S = (10., 20., 30.)
SEVERITY = {'info': '#9bb0c8', 'warning': '#eab85a', 'error': '#ff8282'}
FRAME_CACHE_SIZE = 3
TRACKING_TIMER_INTERVAL_MS = 30
TRACKING_SAMPLE_TOLERANCE_S = 0.0001
BALL_RADIUS_PX = 112.0
BALL_CENTER_X_PX = 132.0
BALL_CENTER_Y_PX = 132.0
BALL_PITCH_SCALE_PX_PER_DEG = 3.0
BALL_MAX_PITCH_DEG = 35.0
DEFAULT_CONFIG_PATH = Path('config/flight_lens.yaml')
TRACKING_CSV_COLUMNS = (
    'time_s', 'phase', 'reason', 'bbox_x', 'bbox_y', 'bbox_width', 'bbox_height',
    'bbox_age_s', 'dx_px', 'dy_px',
)
SQL_EXAMPLES = (
    'Show the first 20 mission events with their times and reasons.',
    'Plot altitude and vertical speed around the first fault.',
    'List telemetry fields with their maximum values.',
    'Show requested roll, output roll, and target X error over time.',
)
ERD_TABLES = {
    'selected_episode': ('id PK', 'physics_id', 'episode_id', 'duration_s', 'outcome'),
    'selected_telemetry': ('episode_id FK', 'topic', 'path', 'time_s', 'value', 'source'),
    'selected_events': ('episode_id FK', 'time_s', 'end_time_s', 'severity', 'component', 'reason'),
}


def mark_export_button(button):
    """Apply the shared visual/accessibility marker used by export actions."""
    button.setObjectName('exportButton')
    button.setProperty('buttonRole', 'export')
    button.setToolTip(button.toolTip() or 'Export data to a local file')
    return button


def tracking_csv_rows(samples):
    """Return CSV-ready tracking rows while preserving samples with missing data."""
    rows = []
    for sample in samples:
        bbox = sample.bbox
        rows.append({
            'time_s': sample.time_s,
            'phase': sample.phase,
            'reason': sample.reason,
            'bbox_x': None if bbox is None else bbox[0],
            'bbox_y': None if bbox is None else bbox[1],
            'bbox_width': None if bbox is None else bbox[2],
            'bbox_height': None if bbox is None else bbox[3],
            'bbox_age_s': sample.bbox_age_s,
            'dx_px': sample.dx_px,
            'dy_px': sample.dy_px,
        })
    return rows


def plain(value):
    if isinstance(value, Mapping):
        return {k: plain(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [plain(v) for v in value]
    return value


def sql_chart(result):
    columns = result.get('columns', []) if isinstance(result, Mapping) else []
    rows = result.get('rows', []) if isinstance(result, Mapping) else []
    if len(columns) < 2 or not rows:
        return None
    numeric = lambda value: isinstance(value, (int, float)) and not isinstance(value, bool) and np.isfinite(value)
    x_index = 0 if all(row and numeric(row[0]) for row in rows) else None
    if x_index is None:
        numeric_columns = [index for index in range(len(columns)) if all(len(row) > index and numeric(row[index]) for row in rows)]
        if not numeric_columns:
            return None
        x_index = numeric_columns[0]
        x_values = list(range(len(rows)))
    else:
        x_values = [float(row[x_index]) for row in rows]
    series = []
    for index, column in enumerate(columns):
        if index == x_index or not all(len(row) > index and numeric(row[index]) for row in rows):
            continue
        series.append({'name': column, 'unit': '', 'values': [float(row[index]) for row in rows]})
    if not series:
        return None
    return dict(type='line_chart', title='SQL query result', x_label=columns[x_index], x_unit='',
                x=x_values, series=series[:8])


def load_config(path):
    path = Path(path)
    if not path.exists():
        return {}
    try:
        data = yaml.safe_load(path.read_text())
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise ValueError(f'invalid FlightLens config: {path}') from error
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError('FlightLens config must be a YAML mapping')
    model = data.get('assistant-model', '')
    if model is not None and not isinstance(model, str):
        raise ValueError('assistant-model must be a string')
    return {'assistant_model': model or None}


class Task(QtCore.QThread):
    result = QtCore.Signal(object)
    failed = QtCore.Signal(str)
    progress = QtCore.Signal(str)

    def __init__(self, kind, token, operation):
        super().__init__()
        self.kind, self.token, self.operation = kind, token, operation
        self.cancelled = CancelEvent()

    def run(self):
        try:
            result = self.operation(self.progress.emit, self.cancelled.is_set)
            if not self.cancelled.is_set():
                self.result.emit(result)
        except Cancelled:
            pass
        except Exception as error:
            self.failed.emit(str(error))


class FrameCanvas(QtWidgets.QLabel):
    def __init__(self):
        super().__init__('Select an event to inspect its exact camera frame.')
        self.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.setWordWrap(True)
        self.setMinimumSize(220, 140)
        self.setStyleSheet('background:#101c2a; color:#9bb0c8; border-radius:5px; padding:6px')
        self.source_image = None

    def clear_frame(self, text):
        self.source_image = None
        self.clear()
        self.setText(text)

    def show_frame(self, frame, bbox):
        image = QtGui.QImage(frame.rgb, frame.width, frame.height, frame.width * 3,
                             QtGui.QImage.Format.Format_RGB888).copy()
        if bbox is not None:
            painter = QtGui.QPainter(image)
            painter.setPen(QtGui.QPen(QtGui.QColor('#ffb454'), 2))
            painter.drawRect(QtCore.QRectF(*bbox))
            painter.end()
        self.source_image = image
        self._fit()

    def _fit(self):
        if self.source_image is not None:
            self.setPixmap(QtGui.QPixmap.fromImage(self.source_image).scaled(
                self.size() - QtCore.QSize(12, 12), QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                QtCore.Qt.TransformationMode.SmoothTransformation))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._fit()


class ExportDialog(QtWidgets.QDialog):
    def __init__(self, destination, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Export MP4')
        self.resize(580, 180)
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(QtWidgets.QLabel('Selected episode · simulation-time playback · H.264 · 30 FPS'))
        row = QtWidgets.QHBoxLayout()
        self.destination = QtWidgets.QLineEdit(str(destination))
        self.destination.setAccessibleName('MP4 output file')
        browse = QtWidgets.QPushButton('Browse…')
        browse.clicked.connect(self.browse)
        row.addWidget(self.destination, 1)
        row.addWidget(browse)
        layout.addLayout(row)
        self.include_boxes = QtWidgets.QCheckBox('Include bounding boxes')
        layout.addWidget(self.include_boxes)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Save |
                                             QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def browse(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, 'Save MP4', self.destination.text(), 'MP4 video (*.mp4)',
            options=QtWidgets.QFileDialog.Option.DontConfirmOverwrite)
        if path:
            self.destination.setText(path)


class ErdDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('FlightLens database ERD')
        self.resize(900, 560)
        layout = QtWidgets.QVBoxLayout(self)
        help_text = QtWidgets.QLabel(
            'SQL is scoped to the selected episode. Join selected_telemetry or selected_events '
            'to selected_episode using episode_id. Use time_s for time-series graphs.')
        help_text.setWordWrap(True)
        layout.addWidget(help_text)
        view = QtWidgets.QGraphicsView()
        view.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        view.setDragMode(QtWidgets.QGraphicsView.DragMode.ScrollHandDrag)
        scene = QtWidgets.QGraphicsScene(view)
        view.setScene(scene)
        layout.addWidget(view, 1)
        positions = {'selected_episode': (30, 50), 'selected_telemetry': (360, 20), 'selected_events': (360, 300)}
        boxes = {}
        for name, columns in ERD_TABLES.items():
            x, y = positions[name]
            width, height = 270, 42 + 22 * len(columns)
            rect = scene.addRect(x, y, width, height, QtGui.QPen(QtGui.QColor('#58b8f5')), QtGui.QBrush(QtGui.QColor('#203950')))
            boxes[name] = (rect, x, y, width, height)
            title = scene.addText(name)
            title.setDefaultTextColor(QtGui.QColor('#70d6ad'))
            title.setFont(QtGui.QFont('Sans', 11, QtGui.QFont.Weight.Bold))
            title.setPos(x + 10, y + 7)
            for index, column in enumerate(columns):
                item = scene.addText(column)
                item.setDefaultTextColor(QtGui.QColor('#d7e5f2'))
                item.setPos(x + 12, y + 36 + index * 22)
        episode = boxes['selected_episode']
        for target_name, target in (('selected_telemetry', boxes['selected_telemetry']), ('selected_events', boxes['selected_events'])):
            scene.addLine(episode[1] + episode[3], episode[2] + episode[4] / 2,
                          target[1], target[2] + 50, QtGui.QPen(QtGui.QColor('#eab85a'), 2))
        scene.setSceneRect(scene.itemsBoundingRect().adjusted(-30, -30, 30, 30))
        view.fitInView(scene.sceneRect(), QtCore.Qt.AspectRatioMode.KeepAspectRatio)
        close = QtWidgets.QPushButton('Close')
        close.clicked.connect(self.accept)
        layout.addWidget(close, 0, QtCore.Qt.AlignmentFlag.AlignRight)


class TrackingReplay(QtWidgets.QWidget):
    SPEEDS = (0.1, 0.25, 0.5, 1.0, 2.0, 4.0)
    issue_selected = QtCore.Signal(object)
    frame_requested = QtCore.Signal(object, object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.samples = ()
        self.issue_events = ()
        self.issue_times = ()
        self.issue_index = -1
        self.index = -1
        self.playing = False
        self.last_wall = None
        self.playback_elapsed_s = 0.0
        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(TRACKING_TIMER_INTERVAL_MS)
        self.timer.timeout.connect(self._tick)
        root = QtWidgets.QVBoxLayout(self)
        controls = QtWidgets.QHBoxLayout()
        self.previous = QtWidgets.QPushButton('◀ Step')
        self.previous.clicked.connect(lambda: self.step(-1))
        self.next = QtWidgets.QPushButton('Step ▶')
        self.next.clicked.connect(lambda: self.step(1))
        self.play = QtWidgets.QPushButton('Play')
        self.play.clicked.connect(self.toggle_play)
        self.reset = QtWidgets.QPushButton('Reset')
        self.reset.clicked.connect(self.reset_replay)
        self.previous_issue = QtWidgets.QPushButton('◀ Issue')
        self.previous_issue.setToolTip('Jump to the previous warning or error')
        self.previous_issue.clicked.connect(lambda: self.jump_issue(-1))
        self.next_issue = QtWidgets.QPushButton('Issue ▶')
        self.next_issue.setToolTip('Jump to the next warning or error')
        self.next_issue.clicked.connect(lambda: self.jump_issue(1))
        self.issue_count = QtWidgets.QLabel('Issues: 0')
        self.export_button = mark_export_button(QtWidgets.QPushButton('Export tracking CSV'))
        self.export_button.clicked.connect(self._export_csv)
        self.export_button.setToolTip('Export bbox geometry and dx/dy tracking errors over time')
        self.export_status = QtWidgets.QLabel('')
        self.export_status.setStyleSheet('color:#9bb0c8')
        self.show_camera_frame = QtWidgets.QCheckBox('Show recorded image')
        self.show_camera_frame.setChecked(True)
        self.show_camera_frame.setToolTip('Use the exact recorded MCAP image in the camera view')
        self.show_camera_frame.toggled.connect(self._toggle_camera_frame)
        self.speed = QtWidgets.QComboBox()
        for value in self.SPEEDS:
            self.speed.addItem(f'{value:g}×', value)
        for widget in (self.previous, self.next, self.play, self.reset, self.previous_issue, self.next_issue):
            controls.addWidget(widget)
        controls.addWidget(self.issue_count)
        controls.addWidget(self.export_button)
        controls.addWidget(self.export_status)
        controls.addWidget(self.show_camera_frame)
        controls.addWidget(QtWidgets.QLabel('Simulation rate'))
        controls.addWidget(self.speed)
        controls.addStretch(1)
        root.addLayout(controls)
        body = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        camera_panel = QtWidgets.QWidget()
        camera_layout = QtWidgets.QVBoxLayout(camera_panel)
        camera_layout.setContentsMargins(0, 0, 0, 0)
        camera_layout.setSpacing(8)
        timeline_group = QtWidgets.QGroupBox('Replay timeline')
        timeline_group.setStyleSheet(
            'QGroupBox { border:1px solid #426983; border-radius:5px; margin-top:8px; padding:8px; }'
            'QGroupBox::title { subcontrol-origin: margin; left:10px; padding:0 4px; color:#70d6ad; }'
        )
        timeline_layout = QtWidgets.QVBoxLayout(timeline_group)
        timeline_layout.setContentsMargins(8, 8, 8, 8)
        timeline_layout.setSpacing(5)
        camera_header = QtWidgets.QHBoxLayout()
        self.progress = QtWidgets.QLabel('No LOS tracking samples')
        camera_header.addWidget(self.progress, 1)
        self.camera_frame_counter = QtWidgets.QLabel('Frame —/—')
        camera_header.addWidget(self.camera_frame_counter)
        timeline_layout.addLayout(camera_header)
        self.camera_timeline = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.camera_timeline.setRange(0, 0)
        self.camera_timeline.setEnabled(False)
        self.camera_timeline.setToolTip('Scrub through tracking samples')
        self.camera_timeline.valueChanged.connect(self._seek_from_timeline)
        timeline_layout.addWidget(self.camera_timeline)
        camera_layout.addWidget(timeline_group)
        self.camera_scene = QtWidgets.QGraphicsScene(self)
        self.camera_view = QtWidgets.QGraphicsView(self.camera_scene)
        self.camera_view.setMinimumSize(520, 390)
        self.camera_view.setBackgroundBrush(QtGui.QColor('#101c2a'))
        self.camera_image = FrameCanvas()
        self.camera_image.setMinimumSize(520, 390)
        self.camera_image.clear_frame('Recorded image will appear when a timeline frame is available.')
        self.camera_stack = QtWidgets.QStackedWidget()
        self.camera_stack.addWidget(self.camera_image)
        self.camera_stack.addWidget(self.camera_view)
        camera_layout.addWidget(self.camera_stack, 1)
        body.addWidget(camera_panel)
        self.ball_scene = QtWidgets.QGraphicsScene(self)
        self.ball_view = QtWidgets.QGraphicsView(self.ball_scene)
        self.ball_view.setMinimumSize(250, 220)
        self.ball_view.setBackgroundBrush(QtGui.QColor('#101c2a'))
        ball_panel = QtWidgets.QWidget()
        ball_layout = QtWidgets.QVBoxLayout(ball_panel)
        ball_layout.setContentsMargins(0, 0, 0, 0)
        ball_layout.addWidget(self.ball_view, 1)
        body.addWidget(ball_panel)
        panel = QtWidgets.QWidget()
        panel_layout = QtWidgets.QVBoxLayout(panel)
        panel_layout.addWidget(QtWidgets.QLabel('Tracking sample'))
        self.category_layout = QtWidgets.QVBoxLayout()
        self.category_layout.setSpacing(8)
        self.category_values = {}
        for title, fields in (
            ('Tracking', ('phase', 'reason', 'navigator')),
            ('Camera / target', ('dx', 'dy', 'bbox age')),
            ('Time to contact', ('ttc raw', 'ttc filtered', 'ttc valid')),
            ('Attitude', ('attitude age', 'measured R/P/Y')),
            ('Command', ('roll', 'pitch', 'yaw rate', 'thrust')),
        ):
            group = QtWidgets.QGroupBox(title)
            form = QtWidgets.QFormLayout(group)
            form.setContentsMargins(8, 6, 8, 6)
            form.setHorizontalSpacing(12)
            for field in fields:
                value = QtWidgets.QLabel('—')
                value.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
                value.setStyleSheet('font:11px monospace; color:#d9e6f4')
                form.addRow(QtWidgets.QLabel(field + ':'), value)
                self.category_values[field] = value
            self.category_layout.addWidget(group)
        self.category_layout.addStretch(1)
        panel_layout.addLayout(self.category_layout, 1)
        issue_group = QtWidgets.QGroupBox('Issue explanation · recorded evidence')
        issue_layout = QtWidgets.QVBoxLayout(issue_group)
        self.issue_explanation = QtWidgets.QPlainTextEdit()
        self.issue_explanation.setReadOnly(True)
        self.issue_explanation.setMinimumHeight(250)
        self.issue_explanation.setMaximumHeight(360)
        self.issue_explanation.setLineWrapMode(QtWidgets.QPlainTextEdit.LineWrapMode.WidgetWidth)
        self.issue_explanation.setStyleSheet('font:12px sans-serif; background:#101c2a; padding:6px')
        issue_layout.addWidget(self.issue_explanation)
        panel_layout.addWidget(issue_group)
        body.addWidget(panel)
        body.setStretchFactor(0, 1)
        body.setStretchFactor(1, 1)
        body.setStretchFactor(2, 1)
        body.setSizes([560, 560, 560])
        root.addWidget(body, 1)
        self.setEnabled(False)

    def _seek_from_timeline(self, index):
        if not self.samples or index == self.index:
            return
        self.playing = False
        self.timer.stop()
        self.play.setText('Play')
        self.playback_elapsed_s = 0.0
        self.index = max(0, min(len(self.samples) - 1, int(index)))
        self._render()

    def set_episode(self, episode):
        self.samples = tuple(episode.tracking_samples)
        self.issue_events = tuple(sorted(
            (event for event in episode.events if event.severity in ('warning', 'error') and event.time_s is not None),
            key=lambda event: (event.time_s, event.arrival_ns),
        ))
        self.issue_times = tuple(event.time_s for event in self.issue_events)
        self.issue_index = -1
        self.index = 0 if self.samples else -1
        self.playing = False
        self.timer.stop()
        self.playback_elapsed_s = 0.0
        self.play.setText('Play')
        self.setEnabled(bool(self.samples))
        self.camera_timeline.setRange(0, max(0, len(self.samples) - 1))
        self.camera_timeline.setEnabled(bool(self.samples))
        self.previous_issue.setEnabled(bool(self.issue_times))
        self.next_issue.setEnabled(bool(self.issue_times))
        self.issue_count.setText(f'Issues: {len(self.issue_events)}')
        self.issue_explanation.setPlainText('Use Issue ▶ / ◀ Issue to inspect recorded warnings and errors.')
        self.export_status.setText('')
        self.camera_image.clear_frame('Move the timeline to load the matching recorded camera frame.')
        self._render()

    def reset_replay(self):
        self.playing = False
        self.timer.stop()
        self.playback_elapsed_s = 0.0
        self.play.setText('Play')
        self.index = 0 if self.samples else -1
        self.camera_timeline.setValue(max(0, self.index))
        self.issue_index = -1
        self.issue_count.setText(f'Issues: {len(self.issue_events)}')
        self.issue_explanation.setPlainText('Use Issue ▶ / ◀ Issue to inspect recorded warnings and errors.')
        self.export_status.setText('')
        self.camera_image.clear_frame('Move the timeline to load the matching recorded camera frame.')
        self._render()

    def step(self, delta):
        if self.samples:
            self.index = max(0, min(len(self.samples) - 1, self.index + delta))
            self._render()

    def jump_issue(self, delta):
        if not self.samples or not self.issue_times:
            return
        self.issue_index = max(0, min(len(self.issue_times) - 1, self.issue_index + delta))
        target_time = self.issue_times[self.issue_index]
        self.index = min(range(len(self.samples)), key=lambda idx: abs(self.samples[idx].time_s - target_time))
        self.playing = False
        self.timer.stop()
        self.play.setText('Play')
        event = self.issue_events[self.issue_index]
        self.issue_count.setText(f'Issue {self.issue_index + 1}/{len(self.issue_events)}')
        self.issue_explanation.setPlainText(self._explain_issue(event))
        self.issue_selected.emit(event)
        self._render()

    def show_frame(self, frame, bbox):
        self.camera_image.show_frame(frame, bbox)

    def clear_timeline_frame(self, text):
        self.camera_image.clear_frame(text)

    def _export_csv(self):
        if not self.samples:
            self.export_status.setText('No tracking samples available.')
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, 'Export tracking CSV', 'tracking-replay.csv', 'CSV files (*.csv)')
        if not path:
            return
        try:
            with open(path, 'w', newline='', encoding='utf-8') as output:
                writer = csv.DictWriter(output, fieldnames=TRACKING_CSV_COLUMNS)
                writer.writeheader()
                writer.writerows(tracking_csv_rows(self.samples))
        except OSError as error:
            self.export_status.setText('Tracking CSV export failed: ' + str(error))
            return
        self.export_status.setText(f'Exported tracking CSV: {path}')

    def _toggle_camera_frame(self, enabled):
        self.camera_stack.setCurrentIndex(0 if enabled else 1)

    @staticmethod
    def _explain_issue(event):
        details = plain(event.details)
        if event.component == 'vision':
            trigger = (
                f'diagnostics.vision_reason == {details.get("vision_reason")!r}; '
                'during LOS, vision_reason == "no_target" is also treated as a warning.'
            )
            interpretation = (
                'The vision pipeline reported a degraded target measurement. Tracking may hold the last '
                'usable bearing, limit the command, or enter recovery until a fresh target is available.'
            )
        elif event.component == 'navigation':
            trigger = f'navigator_state == "RECOVER" with mission reason {event.reason!r}.'
            interpretation = (
                'The navigator entered RECOVER. The controller is prioritizing a safe fallback over normal '
                'target-centering commands.'
            )
        elif event.component == 'phase' and event.phase == 'FAULT':
            trigger = f'mission.phase == "FAULT" and mission.reason == {event.reason!r}.'
            interpretation = (
                'The mission declared a terminal fault. Inspect the preceding tracking samples and limits '
                'to find the control condition that led to this transition.'
            )
        elif event.component == 'physics':
            trigger = f'simulation/state.state == "error" and state.reason == {event.reason!r}.'
            interpretation = 'The physics/backend reported an error, so subsequent control behavior may be invalid.'
        elif event.component == 'renderer':
            trigger = f'analysis/render/health.stats.error is present: {event.reason!r}.'
            interpretation = 'The renderer reported an error. This can explain missing visual evidence, but not by itself a flight-control failure.'
        else:
            trigger = f'recorded event category {event.component!r} with reason {event.reason!r}.'
            interpretation = 'This event was recorded by the flight runtime; inspect the surrounding replay samples for its control effect.'
        checks = (
            ('vision reason', details.get('vision_reason')),
            ('image age', details.get('image_age_s')),
            ('bbox', details.get('processed_bbox') or details.get('latest_image')),
            ('dx', details.get('target_error_x_px')),
            ('dy', details.get('target_error_y_px')),
            ('TTC raw', details.get('optical_ttc_raw_s', details.get('ttc_raw_s'))),
            ('TTC valid', details.get('ttc_valid')),
            ('attitude age', details.get('attitude_age_s')),
            ('thrust saturated', details.get('thrust_saturated')),
            ('angle limited', details.get('angle_limited')),
            ('slew limited', details.get('slew_limited')),
        )
        evidence = '\n'.join(f'  {name}: {"available" if value is not None else "missing"}' for name, value in checks)
        when = 'unsynced' if event.time_s is None else f'{event.time_s:.3f} s'
        interval = '' if event.time_s is None or event.end_s is None or event.end_s <= event.time_s else f'–{event.end_s:.3f} s'
        return '\n'.join((
            f'{event.severity.upper()} · {event.component} · {event.phase or "phase unavailable"}',
            f'time: {when}{interval}',
            f'reason: {event.reason}',
            '',
            'Trigger condition:',
            trigger,
            '',
            'What it means for control:',
            interpretation,
            '',
            'Recorded evidence (available means the field was present):',
            evidence,
            '',
            'Important: missing evidence does not create an issue by itself. It only limits how confidently the replay can explain the control response.',
            '',
            'Raw recorded details:',
            json.dumps(details, indent=2, sort_keys=True),
        ))

    def toggle_play(self):
        if not self.samples:
            return
        self.playing = not self.playing
        self.play.setText('Pause' if self.playing else 'Play')
        self.last_wall = time.monotonic()
        if self.playing:
            self.playback_elapsed_s = 0.0
        self.timer.start() if self.playing else self.timer.stop()

    def _tick(self):
        if not self.playing or not self.samples:
            return
        now = time.monotonic()
        self.playback_elapsed_s += (now - (self.last_wall or now)) * float(self.speed.currentData())
        self.last_wall = now
        while self.index < len(self.samples) - 1:
            delta = max(0., self.samples[self.index + 1].time_s - self.samples[self.index].time_s)
            if self.playback_elapsed_s < delta:
                break
            self.playback_elapsed_s -= delta
            self.index += 1
        self._render()
        if self.index >= len(self.samples) - 1:
            self.playing = False
            self.timer.stop()
            self.play.setText('Play')

    def _render(self):
        self.camera_scene.clear()
        self.ball_scene.clear()
        if self.index < 0:
            self.progress.setText('No LOS tracking samples')
            self.camera_frame_counter.setText('Frame —/—')
            self.camera_timeline.setEnabled(False)
            for value in self.category_values.values():
                value.setText('—')
                value.setStyleSheet('font:11px monospace; color:#d9e6f4')
            return
        sample = self.samples[self.index]
        if sample.frame is not None:
            self.frame_requested.emit(sample.frame, sample.bbox)
        else:
            self.clear_timeline_frame('No recorded camera frame is linked to this tracking sample.')
        width, height = 640., 480.
        self.camera_scene.addRect(0, 0, width, height, QtGui.QPen(QtGui.QColor('#58b8f5'), 3))
        center_pen = QtGui.QPen(QtGui.QColor('#70d6ad'), 2)
        self.camera_scene.addLine(width / 2 - 14, height / 2, width / 2 + 14, height / 2, center_pen)
        self.camera_scene.addLine(width / 2, height / 2 - 14, width / 2, height / 2 + 14, center_pen)
        if sample.bbox is not None:
            self.camera_scene.addRect(*sample.bbox, QtGui.QPen(QtGui.QColor('#eab85a'), 3))
        self.camera_view.fitInView(self.camera_scene.itemsBoundingRect().adjusted(-12, -12, 12, 12), QtCore.Qt.AspectRatioMode.KeepAspectRatio)
        self._render_ball(sample)
        fmt = lambda value, unit='': '—' if value is None else f'{value:+.3f}{unit}'
        attitude = '—' if sample.measured_attitude_deg is None else '/'.join(f'{v:+.2f}°' for v in sample.measured_attitude_deg)
        self.progress.setText(f'{self.index + 1}/{len(self.samples)} · t={sample.time_s:.3f}s · {sample.phase}')
        with QtCore.QSignalBlocker(self.camera_timeline):
            self.camera_timeline.setValue(self.index)
        self.camera_frame_counter.setText(f'Frame {self.index + 1}/{len(self.samples)}')
        values = {
            'phase': sample.phase,
            'reason': sample.reason,
            'navigator': sample.navigator_state or '—',
            'dx': fmt(sample.dx_px, ' px'),
            'dy': fmt(sample.dy_px, ' px'),
            'bbox age': fmt(sample.bbox_age_s, ' s'),
            'ttc raw': fmt(sample.ttc_raw_s, ' s'),
            'ttc filtered': fmt(sample.ttc_filtered_s, ' s'),
            'ttc valid': str(sample.ttc_valid) if sample.ttc_valid is not None else '—',
            'attitude age': fmt(sample.measured_attitude_age_s, ' s'),
            'measured R/P/Y': attitude,
            'roll': fmt(sample.requested_roll_deg, '°'),
            'pitch': fmt(sample.requested_pitch_deg, '°'),
            'yaw rate': fmt(sample.requested_yaw_rate_deg_s, '°/s'),
            'thrust': fmt(sample.requested_thrust_n, ' N'),
        }
        for field, value in values.items():
            self.category_values[field].setText(value)
        self.category_values['phase'].setStyleSheet(
            'font: bold 11px monospace; color:#70d6ad'
        )
        ttc_color = '#70d6ad' if sample.ttc_valid is True else '#ff8282' if sample.ttc_valid is False else '#d9e6f4'
        for field in ('ttc raw', 'ttc filtered', 'ttc valid'):
            self.category_values[field].setStyleSheet(f'font: bold 11px monospace; color:{ttc_color}')

    def _render_ball(self, sample):
        radius, cx, cy = BALL_RADIUS_PX, BALL_CENTER_X_PX, BALL_CENTER_Y_PX
        center = QtCore.QPointF(cx, cy)
        self.ball_scene.addEllipse(cx - radius, cy - radius, radius * 2, radius * 2,
                                   QtGui.QPen(QtGui.QColor('#58b8f5'), 3), QtGui.QBrush(QtGui.QColor('#203950')))
        roll = 0. if sample.measured_attitude_deg is None else np.radians(sample.measured_attitude_deg[0])
        pitch = 0. if sample.measured_attitude_deg is None else sample.measured_attitude_deg[1]
        pitch = max(-BALL_MAX_PITCH_DEG, min(BALL_MAX_PITCH_DEG, pitch))
        offset = pitch * BALL_PITCH_SCALE_PX_PER_DEG
        # Pitch ladder: five-degree marks make small attitude changes visible.
        ladder_pen = QtGui.QPen(QtGui.QColor('#9bb0c8'), 1)
        for pitch_mark in range(-30, 31, 5):
            mark_offset = pitch_mark * BALL_PITCH_SCALE_PX_PER_DEG
            half_width = 30. if pitch_mark % 10 else 44.
            y = cy + mark_offset
            self.ball_scene.addLine(cx - half_width, y, cx + half_width, y, ladder_pen)
            if pitch_mark % 10 == 0 and pitch_mark:
                label = self.ball_scene.addText(f'{abs(pitch_mark)}°')
                label.setDefaultTextColor(QtGui.QColor('#9bb0c8'))
                label.setPos(cx + half_width + 5, y - 9)
        line = QtCore.QLineF(center + QtCore.QPointF(-130 * np.cos(roll), offset - 130 * np.sin(roll)),
                             center + QtCore.QPointF(130 * np.cos(roll), offset + 130 * np.sin(roll)))
        self.ball_scene.addLine(line, QtGui.QPen(QtGui.QColor('#eab85a'), 4))
        # Roll scale around the top half of the ball.
        for roll_mark in (-60, -45, -30, -15, 0, 15, 30, 45, 60):
            angle = np.radians(roll_mark - 90)
            inner = radius - (13 if roll_mark % 30 == 0 else 8)
            outer = radius - 2
            self.ball_scene.addLine(
                cx + inner * np.cos(angle), cy + inner * np.sin(angle),
                cx + outer * np.cos(angle), cy + outer * np.sin(angle),
                QtGui.QPen(QtGui.QColor('#d9e6f4'), 2 if roll_mark % 30 == 0 else 1),
            )
            if roll_mark in (-60, -30, 30, 60):
                label = self.ball_scene.addText(f'{abs(roll_mark)}°')
                label.setDefaultTextColor(QtGui.QColor('#9bb0c8'))
                label.setPos(cx + (outer + 4) * np.cos(angle) - 8, cy + (outer + 4) * np.sin(angle) - 8)
        pen = QtGui.QPen(QtGui.QColor('#70d6ad'), 2)
        self.ball_scene.addLine(cx - 18, cy, cx + 18, cy, pen)
        self.ball_scene.addLine(cx, cy - 18, cx, cy + 18, pen)
        self.ball_scene.setSceneRect(0, 0, cx * 2, cy * 2)
        self.ball_view.fitInView(self.ball_scene.sceneRect().adjusted(-10, -10, 10, 10), QtCore.Qt.AspectRatioMode.KeepAspectRatio)


class TtcInvestigation(QtWidgets.QWidget):
    """Interactive, read-only replay of the optical TTC estimator."""
    cursor_changed = QtCore.Signal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.episode = None
        self.parameters = TtcParameters()
        self._oracle_max_ttc = 1.
        self._last_export_rows = []
        self._cursor_lines = []
        root = QtWidgets.QVBoxLayout(self)
        controls = QtWidgets.QGroupBox('TTC estimator parameters · changes are session-only')
        form = QtWidgets.QFormLayout(controls)
        self.fields = {}
        descriptions = {
            'window_s': ('Recent bbox history used by the fit.', 'Recent bbox-size history used for the reciprocal-size fit.'),
            'min_span_s': ('History duration required before fitting.', 'Minimum elapsed history required before fitting TTC.'),
            'min_growth_px': ('Required apparent target growth.', 'Minimum increase in apparent bbox size required to call the target closing.'),
            'max_relative_residual': ('Reject fits above this error.', 'Maximum normalized fit error accepted as a qualified estimate.'),
            'max_image_age_s': ('Reject stale bbox measurements.', 'Maximum bbox age accepted by the replay before resetting history.'),
            'min_samples': ('Samples required for a fit.', 'Minimum number of bbox samples required for a fit.'),
        }

        def described_control(widget, description):
            row = QtWidgets.QHBoxLayout()
            row.setContentsMargins(0, 0, 0, 0)
            row.addWidget(widget)
            note = QtWidgets.QLabel(description)
            note.setStyleSheet('color:#9bb0c8; font-size:11px;')
            row.addWidget(note, 1)
            return row

        for key, label, minimum, maximum, step, decimals in (
            ('window_s', 'History window (s)', .05, 10., .05, 2),
            ('min_span_s', 'Minimum span (s)', .01, 10., .05, 2),
            ('min_growth_px', 'Minimum size growth (px)', .01, 100., .1, 2),
            ('max_relative_residual', 'Maximum fit residual', .001, 10., .01, 3),
            ('max_image_age_s', 'Maximum image age (s)', .001, 10., .01, 3),
        ):
            widget = QtWidgets.QDoubleSpinBox()
            widget.setRange(minimum, maximum)
            widget.setSingleStep(step)
            widget.setDecimals(decimals)
            widget.setValue(getattr(self.parameters, key))
            widget.setFixedWidth(88)
            self.fields[key] = widget
            form.addRow(label, described_control(widget, descriptions[key][0]))
        self.fields['min_samples'] = QtWidgets.QSpinBox()
        self.fields['min_samples'].setRange(2, 256)
        self.fields['min_samples'].setValue(self.parameters.min_samples)
        self.fields['min_samples'].setFixedWidth(88)
        form.addRow('Minimum samples', described_control(self.fields['min_samples'], descriptions['min_samples'][0]))
        self.butterworth_enabled = QtWidgets.QCheckBox('Show Butterworth-filtered estimate')
        self.butterworth_enabled.setChecked(True)
        self.butterworth_enabled.toggled.connect(lambda _checked: self.recalculate())
        self.butterworth_cutoff = QtWidgets.QDoubleSpinBox()
        self.butterworth_cutoff.setRange(.01, 20.)
        self.butterworth_cutoff.setSingleStep(.1)
        self.butterworth_cutoff.setDecimals(2)
        self.butterworth_cutoff.setValue(1.)
        self.butterworth_cutoff.setSuffix(' Hz')
        self.butterworth_cutoff.setFixedWidth(88)
        self.low_pass_enabled = QtWidgets.QCheckBox('Show low-pass estimate')
        self.low_pass_enabled.setChecked(False)
        self.low_pass_enabled.toggled.connect(lambda _checked: self.recalculate())
        self.low_pass_alpha = QtWidgets.QDoubleSpinBox()
        self.low_pass_alpha.setRange(.01, 1.)
        self.low_pass_alpha.setSingleStep(.05)
        self.low_pass_alpha.setDecimals(2)
        self.low_pass_alpha.setValue(.25)
        self.low_pass_alpha.setFixedWidth(70)
        filter_row = QtWidgets.QHBoxLayout()
        filter_row.addWidget(self.butterworth_enabled)
        filter_row.addWidget(QtWidgets.QLabel('Hz'))
        filter_row.addWidget(self.butterworth_cutoff)
        filter_row.addWidget(QtWidgets.QLabel('cutoff'))
        filter_row.addSpacing(12)
        filter_row.addWidget(self.low_pass_enabled)
        filter_row.addWidget(QtWidgets.QLabel('α'))
        filter_row.addWidget(self.low_pass_alpha)
        filter_row.addWidget(QtWidgets.QLabel('smoothing'))
        filter_row.addStretch(1)
        self.y_axis_range = QtWidgets.QComboBox()
        for value in TTC_Y_AXIS_RANGES_S:
            self.y_axis_range.addItem(f'{value:g} s', value)
        self.y_axis_range.setFixedWidth(78)
        self.y_axis_range.setCurrentIndex(0)
        self.y_axis_range.currentIndexChanged.connect(lambda _index: self._apply_view())
        self.reset_view_button = QtWidgets.QPushButton('Reset view to tracking start')
        self.reset_view_button.clicked.connect(self._apply_view)
        self.recalculate_button = QtWidgets.QPushButton('Recalculate from episode start')
        self.recalculate_button.clicked.connect(self.recalculate)
        self.export_button = QtWidgets.QPushButton('Export TTC CSV')
        mark_export_button(self.export_button)
        self.export_button.clicked.connect(self._export_csv)
        self.more_button = QtWidgets.QToolButton()
        self.more_button.setText('…')
        self.more_button.setToolTip('Explain TTC parameters and filter overlays')
        self.more_button.clicked.connect(self._show_parameter_help)
        root.addWidget(controls)
        filters = QtWidgets.QGroupBox('Filter overlays')
        filters_layout = QtWidgets.QHBoxLayout(filters)
        filters_layout.setContentsMargins(8, 6, 8, 6)
        filters_layout.addLayout(filter_row)
        root.addWidget(filters)
        actions = QtWidgets.QGroupBox('Graph actions')
        actions_layout = QtWidgets.QHBoxLayout(actions)
        actions_layout.setContentsMargins(8, 6, 8, 6)
        actions_layout.addWidget(self.recalculate_button)
        actions_layout.addWidget(self.export_button)
        actions_layout.addWidget(QtWidgets.QLabel('Y-axis range'))
        actions_layout.addWidget(self.y_axis_range)
        actions_layout.addWidget(self.reset_view_button)
        actions_layout.addWidget(self.more_button)
        actions_layout.addStretch(1)
        root.addWidget(actions)
        self.status = QtWidgets.QLabel('Select an episode to begin TTC analysis.')
        self.status.setStyleSheet('color:#9bb0c8; padding:3px')
        root.addWidget(self.status)
        self.raw_plot = self._plot('Recorded TTC · raw and filtered', 'TTC', 's')
        self.estimate_plot = self._plot('Recalculated TTC estimate · qualified values only', 'TTC', 's')
        self.error_plot = self._plot('TTC estimate vs hit-time ground truth', 'Estimated TTC', 's')
        self.error_plot.setLabel('bottom', 'Ground-truth TTC to hit', units='s')
        self.raw_plot.setXLink(self.estimate_plot)
        self.estimate_plot.setXLink(self.raw_plot)
        root.addWidget(self.raw_plot, 1)
        root.addWidget(self.estimate_plot, 1)
        root.addWidget(self.error_plot, 1)

        for key, description in descriptions.items():
            self.fields[key].setToolTip(description[1])
        self.butterworth_cutoff.setToolTip('Offline second-order Butterworth low-pass cutoff for the estimate overlay.')
        self.low_pass_alpha.setToolTip('Causal exponential low-pass alpha in (0, 1]. Higher values follow the estimate more closely.')

    def _show_parameter_help(self):
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle('TTC parameter guide')
        dialog.resize(620, 420)
        layout = QtWidgets.QVBoxLayout(dialog)
        text = QtWidgets.QLabel(
            '<b>Production estimate</b><br>'
            'TTC is fitted from reciprocal apparent bbox size. Invalid samples remain gaps and reset boundaries are preserved.<br><br>'
            '<b>History window</b> controls how much recent bbox motion is fitted.<br>'
            '<b>Minimum span / samples</b> prevent estimates from starting too early.<br>'
            '<b>Minimum growth</b> requires measurable target expansion.<br>'
            '<b>Maximum residual</b> rejects noisy reciprocal-size fits.<br>'
            '<b>Maximum image age</b> rejects stale detections.<br><br>'
            '<b>Butterworth overlay</b> is an offline, zero-phase, second-order low-pass view of qualified estimates.<br>'
            '<b>Low-pass overlay</b> is a causal exponential filter; alpha 1 follows the estimate and smaller alpha values smooth more. '
            'It does not change the recorded flight control values.<br><br>'
            '<b>Hit-time oracle</b> is retrospective: it uses target-hit time minus sample time. '
            'It is available only for episodes that recorded target_hit and is never used by the controller.'
        )
        text.setWordWrap(True)
        text.setTextFormat(QtCore.Qt.TextFormat.RichText)
        layout.addWidget(text)
        close = QtWidgets.QPushButton('Close')
        close.clicked.connect(dialog.accept)
        layout.addWidget(close, 0, QtCore.Qt.AlignmentFlag.AlignRight)
        dialog.exec()

    def _export_csv(self):
        if not self._last_export_rows:
            self.status.setText('No TTC samples available to export.')
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, 'Export TTC CSV', 'ttc-analysis.csv', 'CSV files (*.csv)')
        if not path:
            return
        columns = tuple(self._last_export_rows[0])
        try:
            with open(path, 'w', newline='', encoding='utf-8') as output:
                writer = csv.DictWriter(output, fieldnames=columns)
                writer.writeheader()
                writer.writerows(self._last_export_rows)
        except OSError as error:
            self.status.setText('TTC CSV export failed: ' + str(error))
            return
        self.status.setText(f'Exported TTC CSV: {path}')

    @staticmethod
    def _plot(title, label, unit):
        plot = pg.PlotWidget(title=title)
        plot.setLabel('left', label, units=unit)
        plot.setLabel('bottom', 'Simulation time', units='s')
        plot.getAxis('left').setWidth(76)
        plot.showGrid(x=True, y=True, alpha=.15)
        plot.addLegend(offset=(10, 5), labelTextSize='10pt')
        return plot

    def set_episode(self, episode):
        self.episode = episode
        self.recalculate()

    def set_cursor(self, time_s):
        for line in self._cursor_lines:
            with QtCore.QSignalBlocker(line):
                line.setValue(time_s)

    def _parameters_from_controls(self):
        return TtcParameters(**{name: widget.value() for name, widget in self.fields.items()})

    def _tracking_bounds(self):
        if self.episode is None or not self.episode.tracking_samples:
            return 0., 1.
        start = self.episode.tracking_time_s
        start = self.episode.tracking_samples[0].time_s if start is None else start
        end = self.episode.tracking_samples[-1].time_s
        terminal = [event.time_s for event in self.episode.events
                    if event.time_s is not None and event.time_s >= start
                    and event.phase in ('LOS_COMMIT', 'FAULT')]
        if terminal:
            end = min(end, min(terminal))
        return float(start), max(float(start), float(end))

    def _apply_view(self):
        if self.episode is None:
            return
        start, end = self._tracking_bounds()
        y_max = float(self.y_axis_range.currentData() or TTC_Y_AXIS_RANGES_S[0])
        for plot in (self.raw_plot, self.estimate_plot):
            plot.setLimits(xMin=start, xMax=end, yMin=0., yMax=y_max)
            plot.setXRange(start, end, padding=0.)
            plot.setYRange(0., y_max, padding=0.)
        oracle_max = max(y_max, self._oracle_max_ttc)
        self.error_plot.setLimits(xMin=0., xMax=oracle_max, yMin=0., yMax=y_max)
        self.error_plot.setXRange(oracle_max, 0., padding=0.)
        self.error_plot.setYRange(0., y_max, padding=0.)
        self.error_plot.setYRange(-y_max, y_max, padding=0.)

    def recalculate(self):
        if self.episode is None:
            return
        try:
            parameters = self._parameters_from_controls().validate()
            result = recalculate(self.episode.tracking_samples, parameters)
        except ValueError as error:
            self.status.setText('Invalid TTC parameters: ' + str(error))
            return
        self.parameters = parameters
        self._cursor_lines.clear()
        self.raw_plot.clear()
        self.estimate_plot.clear()
        self.error_plot.clear()
        tracking_start, tracking_end = self._tracking_bounds()
        if len(self.episode.ttc_raw.times):
            raw_mask = (self.episode.ttc_raw.times >= tracking_start) & (self.episode.ttc_raw.times <= tracking_end)
            self.raw_plot.plot(self.episode.ttc_raw.times[raw_mask], self.episode.ttc_raw.values[raw_mask, 0],
                               pen=pg.mkPen(VISUAL_RAW_COLOR, width=2), name='Recorded raw TTC', connect='finite')
        if len(self.episode.ttc_filtered.times):
            filtered_mask = (self.episode.ttc_filtered.times >= tracking_start) & (self.episode.ttc_filtered.times <= tracking_end)
            self.raw_plot.plot(self.episode.ttc_filtered.times[filtered_mask], self.episode.ttc_filtered.values[filtered_mask, 0],
                               pen=pg.mkPen(VISUAL_FILTERED_COLOR, width=2, style=QtCore.Qt.PenStyle.DashLine),
                               name='Recorded filtered TTC', connect='finite')
        points = result.points
        comparison = compare_to_hit_time(self.episode, points)
        self._oracle_max_ttc = max((value for _, value in comparison.oracle), default=1.)
        raw_by_time = dict(zip(self.episode.ttc_raw.times.tolist(), self.episode.ttc_raw.values[:, 0].tolist()))
        filtered_by_time = dict(zip(self.episode.ttc_filtered.times.tolist(), self.episode.ttc_filtered.values[:, 0].tolist()))
        oracle_by_time = dict(comparison.oracle)
        butter_values = butterworth_filter(points, self.butterworth_cutoff.value())
        low_values = low_pass_filter(points, self.low_pass_alpha.value())
        self._last_export_rows = []
        for point, butter_value, low_value in zip(points, butter_values, low_values):
            self._last_export_rows.append({
                'simulation_time_s': point.time_s,
                'recorded_raw_ttc_s': raw_by_time.get(point.time_s),
                'recorded_filtered_ttc_s': filtered_by_time.get(point.time_s),
                'recalculated_ttc_s': point.value_s if point.valid else None,
                'butterworth_ttc_s': butter_value if self.butterworth_enabled.isChecked() else None,
                'low_pass_ttc_s': low_value if self.low_pass_enabled.isChecked() else None,
                'ground_truth_ttc_s': oracle_by_time.get(point.time_s),
                'ttc_valid': point.valid,
                'ttc_reason': point.reason,
            })
        valid = [(point.time_s, point.value_s) for point in points
                 if point.valid and tracking_start <= point.time_s <= tracking_end]
        if valid:
            self.estimate_plot.plot([item[0] for item in valid], [item[1] for item in valid],
                                    pen=pg.mkPen('#c299f0', width=2), name='Recalculated estimate', connect='finite')
        reason_colors = {'not_closing': '#eab85a', 'unresolved_growth': '#ff8282',
                         'insufficient_history': '#9bb0c8', 'noisy_growth': '#f28f8f',
                         'stale': '#c299f0', 'recovery': '#ffb454', 'clipped_target': '#ff8282'}
        for reason, color in reason_colors.items():
            invalid = [point for point in points if not point.valid and point.reason == reason
                       and tracking_start <= point.time_s <= tracking_end]
            if invalid:
                self.estimate_plot.plot([point.time_s for point in invalid], [0.] * len(invalid),
                                        pen=None, symbol='t', symbolSize=7,
                                        symbolBrush=color, name=reason, connect='finite')
        if self.butterworth_enabled.isChecked() and valid:
            filtered_values = butter_values
            filtered_curve = [(point.time_s, value) for point, value in zip(points, filtered_values)
                              if value is not None and tracking_start <= point.time_s <= tracking_end]
            if filtered_curve:
                self.estimate_plot.plot([item[0] for item in filtered_curve], [item[1] for item in filtered_curve],
                                        pen=pg.mkPen('#70d6ad', width=2, style=QtCore.Qt.PenStyle.DashLine),
                                        name='Butterworth estimate', connect='finite')
        if self.low_pass_enabled.isChecked() and valid:
            filtered_values = low_values
            filtered_curve = [(point.time_s, value) for point, value in zip(points, filtered_values)
                              if value is not None and tracking_start <= point.time_s <= tracking_end]
            if filtered_curve:
                self.estimate_plot.plot([item[0] for item in filtered_curve], [item[1] for item in filtered_curve],
                                        pen=pg.mkPen('#58b8f5', width=2, style=QtCore.Qt.PenStyle.DashDotLine),
                                        name='Low-pass estimate', connect='finite')
        oracle = [(time_s, value) for time_s, value in comparison.oracle
                  if tracking_start <= time_s <= tracking_end]
        if oracle:
            self.estimate_plot.plot([item[0] for item in oracle], [item[1] for item in oracle],
                                    pen=pg.mkPen('#ffffff', width=2, style=QtCore.Qt.PenStyle.DashLine),
                                    name='Hit-time oracle', connect='finite')
            oracle_by_time = dict(oracle)
            estimate_comparison = [(oracle_by_time[point.time_s], point.value_s)
                              for point in points if point.valid and point.time_s in oracle_by_time
                              and tracking_start <= point.time_s <= tracking_end]
            if estimate_comparison:
                self.error_plot.plot([item[0] for item in estimate_comparison], [item[1] for item in estimate_comparison],
                                     pen=None, symbol='o', symbolSize=5, symbolBrush='#c299f0',
                                     name='Estimate', connect='finite')
            if self.butterworth_enabled.isChecked():
                butter_values = butterworth_filter(points, self.butterworth_cutoff.value())
                curve = [(oracle_by_time[point.time_s], value)
                         for point, value in zip(points, butter_values)
                         if value is not None and point.time_s in oracle_by_time
                         and tracking_start <= point.time_s <= tracking_end]
                if curve:
                    self.error_plot.plot([item[0] for item in curve], [item[1] for item in curve],
                                         pen=None, symbol='t', symbolSize=6, symbolBrush='#70d6ad',
                                         name='Butterworth', connect='finite')
            if self.low_pass_enabled.isChecked():
                low_values = low_pass_filter(points, self.low_pass_alpha.value())
                curve = [(oracle_by_time[point.time_s], value)
                         for point, value in zip(points, low_values)
                         if value is not None and point.time_s in oracle_by_time
                         and tracking_start <= point.time_s <= tracking_end]
                if curve:
                    self.error_plot.plot([item[0] for item in curve], [item[1] for item in curve],
                                         pen=None, symbol='s', symbolSize=6, symbolBrush='#58b8f5',
                                         name='Low-pass', connect='finite')
            max_ttc = max(value for _, value in oracle)
            self.error_plot.plot([0., max_ttc], [0., max_ttc],
                                 pen=pg.mkPen('#ffffff', width=2, style=QtCore.Qt.PenStyle.DashLine),
                                 name='Perfect agreement y=x', connect='finite')
        y_max = float(self.y_axis_range.currentData() or TTC_Y_AXIS_RANGES_S[0])
        for plot in (self.raw_plot, self.estimate_plot):
            plot.setLimits(xMin=tracking_start, xMax=tracking_end, yMin=0., yMax=y_max)
            plot.setXRange(tracking_start, tracking_end, padding=0.)
            plot.setYRange(0., y_max, padding=0.)
            line = pg.InfiniteLine(tracking_start,
                                   angle=90, movable=True, pen=pg.mkPen('#ffffff', width=1))
            plot.addItem(line)
            line.sigPositionChanged.connect(lambda line=line: self.cursor_changed.emit(line.value()))
            self._cursor_lines.append(line)
        oracle_max = max(y_max, self._oracle_max_ttc)
        self.error_plot.setLimits(xMin=0., xMax=oracle_max, yMin=0., yMax=y_max)
        self.error_plot.setXRange(oracle_max, 0., padding=0.)
        self.error_plot.setYRange(0., y_max, padding=0.)
        counts = ', '.join(f'{reason}: {count}' for reason, count in sorted(result.counts.items()))
        metrics = ('oracle unavailable: no recorded target_hit event' if comparison.hit_time_s is None else
                   f'oracle hit t={comparison.hit_time_s:.3f}s · MAE {comparison.mean_absolute_error_s:.3f}s · '
                   f'RMSE {comparison.root_mean_square_error_s:.3f}s · bias {comparison.bias_s:+.3f}s · '
                   f'coverage {comparison.coverage:.0%}') if comparison.hit_time_s is not None and comparison.sample_count else (
                       'oracle available, but no qualified estimate overlaps it' if comparison.hit_time_s is not None else
                       'oracle unavailable: no recorded target_hit event')
        self.status.setText(f'Recorded points: {len(self.episode.ttc_raw.times)} raw, {len(self.episode.ttc_filtered.times)} filtered · Recalculated: {counts} · {metrics}')


class Overview(QtWidgets.QMainWindow):
    def __init__(self, assistant_model=None):
        super().__init__()
        self.assistant_model = assistant_model
        self.assistant = None if assistant_model is None else McapAssistant(LlamaCppBackend(assistant_model))
        self.data = self.episode = None
        self.jobs = set()
        self.serial = self.load_token = self.preview_token = self.tracking_frame_token = 0
        self.export_token = -1
        self.assistant_token = -1
        self.assistant_generated_sql = None
        self.assistant_sql_episode_key = None
        self.closing = False
        self.frame_cache = OrderedDict()
        self.last_error = ''
        self.cursor_time_s = 0.
        self.setWindowTitle('FlightLens · Flight Overview')
        self.resize(1760, 1040)
        self.setMinimumSize(1350, 820)
        pg.setConfigOptions(background='#101c2a', foreground='#c8d5e6', antialias=False, useOpenGL=False)
        self.setStyleSheet('''
            QMainWindow, QWidget { background:#142233; color:#d9e6f4; font:13px sans-serif; }
            QPushButton { background:#284c68; padding:8px 15px; border:1px solid #426983; border-radius:5px; }
            QPushButton:hover { background:#356381; } QPushButton:disabled { color:#6f8298; }
            QPushButton[buttonRole="export"] { background:#237a67; border:1px solid #70d6ad; color:#f1fff9; font-weight:600; }
            QPushButton[buttonRole="export"]:hover { background:#2d967e; }
            QPushButton[buttonRole="export"]:disabled { background:#284c68; border-color:#426983; color:#6f8298; }
            QComboBox { padding:7px; background:#1e344a; border:1px solid #3b5269; border-radius:4px; }
            QTabWidget::pane { border:1px solid #30485f; } QTabBar::tab { padding:9px 20px; background:#203950; }
            QTabBar::tab:selected { border-bottom:2px solid #58b8f5; }
            QTableWidget, QPlainTextEdit { background:#101c2a; border:1px solid #30485f; }
            QHeaderView::section { background:#203950; padding:5px; border:0; }
            QTableWidget::item:selected { background:#294d69; }
            QSplitter::handle { background:#263c51; }
        ''')
        root = QtWidgets.QWidget()
        outer = QtWidgets.QVBoxLayout(root)
        outer.setContentsMargins(18, 12, 18, 12)
        top = QtWidgets.QHBoxLayout()
        brand = QtWidgets.QLabel('FLIGHTLENS')
        brand.setStyleSheet('font-size:23px; font-weight:700; color:#70d6ad; letter-spacing:2px')
        top.addWidget(brand)
        self.file_label = QtWidgets.QLabel('Open a local MCAP to begin')
        self.file_label.setTextFormat(QtCore.Qt.TextFormat.PlainText)
        top.addWidget(self.file_label, 1)
        self.episodes = QtWidgets.QComboBox()
        self.episodes.setMinimumWidth(230)
        self.episodes.setAccessibleName('Flight episode')
        self.episodes.currentIndexChanged.connect(self.select_episode)
        top.addWidget(self.episodes)
        self.open_button = QtWidgets.QPushButton('Open MCAP…')
        self.open_button.clicked.connect(self.choose_file)
        top.addWidget(self.open_button)
        self.export_button = QtWidgets.QPushButton('Export MP4…')
        mark_export_button(self.export_button)
        self.export_button.setEnabled(False)
        self.export_button.clicked.connect(self.choose_export)
        top.addWidget(self.export_button)
        self.cancel_export_button = QtWidgets.QPushButton('Cancel export')
        self.cancel_export_button.clicked.connect(self.cancel_export)
        self.cancel_export_button.hide()
        top.addWidget(self.cancel_export_button)
        self.cancel_button = QtWidgets.QPushButton('Cancel loading')
        self.cancel_button.clicked.connect(self.cancel_load)
        self.cancel_button.hide()
        top.addWidget(self.cancel_button)
        outer.addLayout(top)
        self.tabs = QtWidgets.QTabWidget()
        self.content = QtWidgets.QWidget()
        self.tabs.addTab(self.content, 'Flight Overview')
        self.controls = QtWidgets.QWidget()
        self.tabs.addTab(self.controls, 'Attitude & Thrust')
        self.msp_tab = QtWidgets.QWidget()
        self.tabs.addTab(self.msp_tab, 'MSP')
        self.visual_tab = QtWidgets.QWidget()
        self.tabs.addTab(self.visual_tab, 'Visual')
        self.ttc_tab = TtcInvestigation()
        self.ttc_tab.cursor_changed.connect(self.set_cursor)
        self.tabs.addTab(self.ttc_tab, 'TTC')
        self.velocity_tab = QtWidgets.QWidget()
        self.tabs.addTab(self.velocity_tab, 'Velocity')
        self.tracking_tab = TrackingReplay()
        self.tracking_tab.issue_selected.connect(self._select_tracking_issue)
        self.tracking_tab.frame_requested.connect(self._request_tracking_frame)
        self.tabs.addTab(self.tracking_tab, 'Tracking')
        self.assistant_tab = QtWidgets.QWidget()
        self.tabs.addTab(self.assistant_tab, 'Assistant')
        outer.addWidget(self.tabs)
        layout = QtWidgets.QVBoxLayout(self.content)
        cards = QtWidgets.QHBoxLayout()
        self.cards = {}
        for title in ('Outcome', 'Duration', 'Peak altitude', 'First fault', 'Warnings / errors', 'Recording'):
            card = QtWidgets.QLabel(title + '\n—')
            card.setTextFormat(QtCore.Qt.TextFormat.PlainText)
            card.setWordWrap(True)
            card.setMinimumHeight(62)
            card.setStyleSheet('background:#20354a; border-radius:5px; padding:8px; font-weight:600')
            cards.addWidget(card, 2 if title in ('Outcome', 'First fault') else 1)
            self.cards[title] = card
        layout.addLayout(cards)
        self.notices = QtWidgets.QLabel('Read-only analysis · No flight connection · No parameter changes')
        self.notices.setTextFormat(QtCore.Qt.TextFormat.PlainText)
        self.notices.setWordWrap(True)
        self.notices.setStyleSheet('color:#eab85a; padding:3px')
        layout.addWidget(self.notices)
        split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        plots = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        self.trajectory = pg.PlotWidget(title='X–Z trajectory · ▲ start · ■ end · ● cursor')
        self.trajectory.setLabel('bottom', 'World X', units='m')
        self.trajectory.setLabel('left', 'World Z', units='m')
        self.trajectory.setAspectLocked(True)
        self.trajectory.showGrid(x=True, y=True, alpha=.15)
        plots.addWidget(self.trajectory)
        time_panel = QtWidgets.QWidget()
        time_layout = QtWidgets.QVBoxLayout(time_panel)
        time_layout.setContentsMargins(0, 0, 0, 0)
        self.altitude = pg.PlotWidget(title='Altitude · actual, barometer, recorded takeoff goal')
        self.altitude.setLabel('left', 'Altitude', units='m')
        self.margins = pg.PlotWidget(title='BBox edge clearance · zero is boundary contact')
        self.margins.setLabel('left', 'Clearance', units='px')
        for plot in (self.altitude, self.margins):
            plot.getAxis('left').setWidth(64)
            plot.setLabel('bottom', 'Simulation time', units='s')
            plot.showGrid(x=True, y=True, alpha=.15)
            plot.addLegend(offset=(10, 5), labelTextSize='10pt')
            time_layout.addWidget(plot)
        self.margins.setXLink(self.altitude)
        plots.addWidget(time_panel)
        plots.setSizes([580, 850])
        split.addWidget(plots)
        lower = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        event_panel = QtWidgets.QWidget()
        event_layout = QtWidgets.QVBoxLayout(event_panel)
        event_layout.setContentsMargins(0, 0, 0, 0)
        event_layout.addWidget(QtWidgets.QLabel('EVENTS · Select a row to inspect the recorded evidence'))
        self.events = QtWidgets.QTableWidget(0, 5)
        self.events.setHorizontalHeaderLabels(['Time / interval', 'Level', 'Component', 'Phase', 'Recorded event'])
        self.events.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.events.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.events.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.events.verticalHeader().hide()
        self.events.horizontalHeader().setStretchLastSection(True)
        self.events.setColumnWidth(0, 132)
        self.events.setColumnWidth(1, 72)
        self.events.setColumnWidth(2, 86)
        self.events.setColumnWidth(3, 104)
        self.events.itemSelectionChanged.connect(self.select_event)
        event_layout.addWidget(self.events)
        lower.addWidget(event_panel)
        preview_panel = QtWidgets.QWidget()
        preview_layout = QtWidgets.QVBoxLayout(preview_panel)
        preview_layout.setContentsMargins(0, 0, 0, 0)
        self.preview_caption = QtWidgets.QLabel('EVENT CAMERA FRAME · Loaded only when selected')
        self.preview_caption.setTextFormat(QtCore.Qt.TextFormat.PlainText)
        self.preview_caption.setWordWrap(True)
        preview_layout.addWidget(self.preview_caption)
        evidence = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        self.canvas = FrameCanvas()
        self.details = QtWidgets.QPlainTextEdit()
        self.details.setReadOnly(True)
        self.details.setPlaceholderText('Recorded event details appear here. A rejection alone is not proof of root cause.')
        self.details.setStyleSheet('font:11px monospace; background:#101c2a')
        evidence.addWidget(self.canvas)
        evidence.addWidget(self.details)
        evidence.setSizes([330, 260])
        preview_layout.addWidget(evidence)
        lower.addWidget(preview_panel)
        lower.setSizes([750, 650])
        split.addWidget(lower)
        split.setSizes([440, 290])
        layout.addWidget(split, 1)
        self.readout = QtWidgets.QLabel('Cursor: —')
        self.readout.setTextFormat(QtCore.Qt.TextFormat.PlainText)
        layout.addWidget(self.readout)
        self.status = QtWidgets.QLabel('Open MCAP · Inspect events · Follow the shared simulation-time cursor')
        self.status.setTextFormat(QtCore.Qt.TextFormat.PlainText)
        outer.addWidget(self.status)
        self.setCentralWidget(root)
        controls_layout = QtWidgets.QVBoxLayout(self.controls)
        controls_layout.setContentsMargins(12, 12, 12, 12)
        self.attitude_plots = []
        for axis in ('Roll', 'Pitch', 'Yaw'):
            plot = pg.PlotWidget(title=f'{axis} attitude · measured and estimated')
            plot.setLabel('left', 'Angle', units='deg')
            plot.getAxis('left').setWidth(76)
            plot.setLabel('bottom', 'Simulation time', units='s')
            plot.showGrid(x=True, y=True, alpha=.15)
            plot.addLegend(offset=(10, 5), labelTextSize='10pt')
            controls_layout.addWidget(plot)
            self.attitude_plots.append(plot)
        self.thrust_plot = pg.PlotWidget(title='Commanded collective thrust')
        self.thrust_plot.setLabel('left', 'Commanded thrust', units='%')
        for plot in (self.thrust_plot,):
            plot.getAxis('left').setWidth(76)
            plot.setLabel('bottom', 'Simulation time', units='s')
            plot.showGrid(x=True, y=True, alpha=.15)
            plot.addLegend(offset=(10, 5), labelTextSize='10pt')
            controls_layout.addWidget(plot)
        for plot in self.attitude_plots[1:] + [self.thrust_plot]:
            plot.setXLink(self.attitude_plots[0])
        msp_layout = QtWidgets.QVBoxLayout(self.msp_tab)
        msp_layout.setContentsMargins(12, 12, 12, 12)
        self.msp_status = QtWidgets.QLabel('Betaflight MSP attitude · no recording loaded')
        self.msp_status.setStyleSheet('color:#9bb0c8; padding:3px')
        msp_layout.addWidget(self.msp_status)
        self.msp_plots = []
        for axis in ('Roll', 'Pitch', 'Yaw'):
            plot = pg.PlotWidget(title=f'MSP {axis} attitude')
            plot.setLabel('left', 'Angle', units='deg')
            plot.getAxis('left').setWidth(76)
            plot.setLabel('bottom', 'Simulation time', units='s')
            plot.showGrid(x=True, y=True, alpha=.15)
            plot.addLegend(offset=(10, 5), labelTextSize='10pt')
            msp_layout.addWidget(plot)
            self.msp_plots.append(plot)
        for plot in self.msp_plots[1:]:
            plot.setXLink(self.msp_plots[0])
        visual_layout = QtWidgets.QVBoxLayout(self.visual_tab)
        visual_layout.setContentsMargins(12, 12, 12, 12)
        self.visual_status = QtWidgets.QLabel('Visual telemetry · TTC and camera-center errors')
        self.visual_status.setStyleSheet('color:#9bb0c8; padding:3px')
        visual_layout.addWidget(self.visual_status)
        self.visual_plots = []
        for title, label, unit in (('Time to contact · ttc_valid=true', 'TTC', 's'),
                                   ('Camera-center X error', 'X error', 'px'),
                                   ('Camera-center Y error', 'Y error', 'px')):
            plot = pg.PlotWidget(title=title)
            plot.setLabel('left', label, units=unit)
            plot.setLabel('bottom', 'Simulation time', units='s')
            plot.getAxis('left').setWidth(76)
            plot.showGrid(x=True, y=True, alpha=.15)
            plot.addLegend(offset=(10, 5), labelTextSize='10pt')
            visual_layout.addWidget(plot)
            self.visual_plots.append(plot)
        for plot in self.visual_plots[1:]:
            plot.setXLink(self.visual_plots[0])
        velocity_layout = QtWidgets.QVBoxLayout(self.velocity_tab)
        velocity_layout.setContentsMargins(12, 12, 12, 12)
        self.velocity_plots = []
        for axis in ('X', 'Y', 'Z'):
            plot = pg.PlotWidget(title=f'{axis} velocity · ground truth' + (' and barometer-derived' if axis == 'Z' else ''))
            plot.setLabel('left', 'Velocity', units='m/s')
            plot.setLabel('bottom', 'Simulation time', units='s')
            plot.getAxis('left').setWidth(76)
            plot.showGrid(x=True, y=True, alpha=.15)
            plot.addLegend(offset=(10, 5), labelTextSize='10pt')
            velocity_layout.addWidget(plot)
            self.velocity_plots.append(plot)
        slant_plot = pg.PlotWidget(title='X–Z slant velocity · ground truth')
        slant_plot.setLabel('left', 'Velocity', units='m/s')
        slant_plot.setLabel('bottom', 'Simulation time', units='s')
        slant_plot.getAxis('left').setWidth(76)
        slant_plot.showGrid(x=True, y=True, alpha=.15)
        slant_plot.addLegend(offset=(10, 5), labelTextSize='10pt')
        velocity_layout.addWidget(slant_plot)
        self.velocity_plots.append(slant_plot)
        for plot in self.velocity_plots[1:]:
            plot.setXLink(self.velocity_plots[0])
        self._build_assistant()
        self.cursors = []
        self.selected_position = None

    def _build_assistant(self):
        layout = QtWidgets.QVBoxLayout(self.assistant_tab)
        layout.setContentsMargins(12, 12, 12, 12)
        split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        left = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left)
        left_layout.addWidget(QtWidgets.QLabel('LOCAL FLIGHT ASSISTANT'))
        self.assistant_status = QtWidgets.QLabel(
            'Configure --assistant-model with a GGUF file to enable this tab.'
            if self.assistant is None else 'Ready when an MCAP is loaded.')
        self.assistant_status.setWordWrap(True)
        self.assistant_status.setStyleSheet('color:#9bb0c8; padding:3px')
        left_layout.addWidget(self.assistant_status)
        examples = QtWidgets.QComboBox()
        examples.addItems((
            'Summarize this flight.',
            'What happened at the first fault?',
            'Plot altitude and vertical speed around the first fault.',
            'Plot X, Z, and slant velocity during target search.',
            'Compare actual and estimated attitude.',
            *SQL_EXAMPLES,
        ))
        use_example = QtWidgets.QPushButton('Use example prompt')
        use_example.clicked.connect(lambda: self.assistant_prompt.setPlainText(examples.currentText()))
        left_layout.addWidget(examples)
        left_layout.addWidget(use_example)
        self.assistant_erd_button = QtWidgets.QPushButton('Open database ERD')
        self.assistant_erd_button.clicked.connect(self.open_assistant_erd)
        self.assistant_erd_button.setEnabled(False)
        left_layout.addWidget(self.assistant_erd_button)
        self.assistant_transcript = QtWidgets.QPlainTextEdit()
        self.assistant_transcript.setReadOnly(True)
        self.assistant_transcript.setPlaceholderText('Assistant answers will appear here.')
        left_layout.addWidget(self.assistant_transcript, 1)
        left_layout.addWidget(QtWidgets.QLabel('Generated queries and results'))
        self.assistant_query_output = QtWidgets.QPlainTextEdit()
        self.assistant_query_output.setReadOnly(True)
        self.assistant_query_output.setPlaceholderText('Tool queries and bounded data results will appear here.')
        self.assistant_query_output.setMaximumHeight(180)
        self.assistant_query_output.setStyleSheet('font:10px monospace; background:#101c2a')
        left_layout.addWidget(self.assistant_query_output)
        left_layout.addWidget(QtWidgets.QLabel('Custom prompt'))
        self.assistant_prompt = QtWidgets.QPlainTextEdit()
        self.assistant_prompt.setPlaceholderText('Enter a custom question about the selected episode…')
        self.assistant_prompt.setMaximumHeight(78)
        left_layout.addWidget(self.assistant_prompt)
        controls = QtWidgets.QHBoxLayout()
        controls.addWidget(QtWidgets.QLabel('Result view:'))
        self.assistant_graph = QtWidgets.QComboBox()
        self.assistant_graph.addItems(('Table', 'Graph'))
        self.assistant_graph.currentTextChanged.connect(self._refresh_assistant_result_view)
        controls.addWidget(self.assistant_graph)
        self.assistant_submit = QtWidgets.QPushButton('Run query')
        self.assistant_submit.clicked.connect(self.ask_assistant)
        controls.addWidget(self.assistant_submit)
        self.assistant_generate_sql = QtWidgets.QPushButton('Generate SQL')
        self.assistant_generate_sql.clicked.connect(self.generate_sql)
        controls.addWidget(self.assistant_generate_sql)
        self.assistant_run_sql = QtWidgets.QPushButton('Run SQL')
        self.assistant_run_sql.clicked.connect(self.run_sql)
        self.assistant_run_sql.setEnabled(False)
        controls.addWidget(self.assistant_run_sql)
        self.assistant_cancel = QtWidgets.QPushButton('Cancel')
        self.assistant_cancel.clicked.connect(self.cancel_assistant)
        self.assistant_cancel.hide()
        controls.addWidget(self.assistant_cancel)
        left_layout.addLayout(controls)
        split.addWidget(left)
        result_panel = QtWidgets.QWidget()
        result_layout = QtWidgets.QVBoxLayout(result_panel)
        self.assistant_result_status = QtWidgets.QLabel('SQL results will appear here.')
        self.assistant_result_status.setWordWrap(True)
        result_layout.addWidget(self.assistant_result_status)
        self.assistant_table = QtWidgets.QTableWidget()
        self.assistant_table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.assistant_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        result_layout.addWidget(self.assistant_table, 1)
        self.assistant_plot = pg.PlotWidget(title='Assistant chart result')
        self.assistant_plot.setLabel('bottom', 'Simulation time', units='s')
        self.assistant_plot.setLabel('left', 'Value')
        self.assistant_plot.showGrid(x=True, y=True, alpha=.15)
        self.assistant_plot.addLegend(offset=(10, 5), labelTextSize='10pt')
        result_layout.addWidget(self.assistant_plot, 1)
        split.addWidget(result_panel)
        split.setSizes([620, 760])
        layout.addWidget(split, 1)
        enabled = self.assistant is not None
        self.assistant_prompt.setEnabled(True)
        self.assistant_submit.setEnabled(enabled)
        self.assistant_generate_sql.setEnabled(enabled)
        self.assistant_graph.setEnabled(enabled)
        self._refresh_assistant_result_view()

    def open_assistant_erd(self):
        if self.data is not None and self.data.index_path is not None:
            ErdDialog(self).exec()

    def _show_assistant_table(self, result):
        if not isinstance(result, dict) or not isinstance(result.get('columns'), list):
            self.assistant_table.clear()
            self.assistant_table.setRowCount(0)
            self.assistant_table.setColumnCount(0)
            return
        columns, rows = result['columns'], result.get('rows', [])
        self.assistant_table.setColumnCount(len(columns))
        self.assistant_table.setHorizontalHeaderLabels([str(column) for column in columns])
        self.assistant_table.setRowCount(len(rows))
        for row_index, row in enumerate(rows):
            for column_index, value in enumerate(row[:len(columns)]):
                self.assistant_table.setItem(row_index, column_index, QtWidgets.QTableWidgetItem(str(value)))
        self.assistant_table.resizeColumnsToContents()

    def _refresh_assistant_result_view(self):
        graph = self.assistant_graph.currentText() == 'Graph'
        self.assistant_table.setVisible(not graph)
        self.assistant_plot.setVisible(graph)

    def ask_assistant(self):
        if self.assistant is None or self.data is None or self.episode is None:
            self.assistant_status.setText('Load an MCAP and configure --assistant-model first.')
            return
        if any(job.kind == 'assistant' for job in self.jobs):
            return
        prompt = self.assistant_prompt.toPlainText().strip()
        if not prompt:
            self.assistant_status.setText('Enter a question or choose an example prompt.')
            return
        data, episode = self.data, self.episode
        job = self._task('assistant', lambda progress, cancel: self.assistant.answer(
            data, episode, prompt, cancel=cancel, progress=progress))
        self.assistant_token = job.token
        self.assistant_submit.setEnabled(False)
        self.assistant_cancel.show()
        self.assistant_status.setText('Analyzing selected MCAP data…')
        self._append_assistant('You', prompt)
        job.start()

    def generate_sql(self):
        if self.assistant is None or self.data is None or self.episode is None:
            self.assistant_status.setText('Load an MCAP and configure --assistant-model first.')
            return
        if any(job.kind in ('assistant', 'sql_preview', 'sql_run') for job in self.jobs):
            return
        prompt = self.assistant_prompt.toPlainText().strip()
        if not prompt:
            self.assistant_status.setText('Enter a custom prompt first.')
            return
        data, episode = self.data, self.episode
        job = self._task('sql_preview', lambda progress, cancel: self.assistant.generate_sql(
            data, episode, prompt, cancel=cancel))
        self.assistant_token = job.token
        self.assistant_generate_sql.setEnabled(False)
        self.assistant_run_sql.setEnabled(False)
        self.assistant_cancel.show()
        self.assistant_status.setText('Generating SQL without executing it…')
        self._append_assistant('You', prompt)
        job.start()

    def run_sql(self):
        if (self.data is None or self.episode is None or not self.assistant_generated_sql
                or self.assistant_sql_episode_key != self.episode.key):
            self.assistant_status.setText('Generate valid SQL for the selected episode first.')
            return
        if any(job.kind in ('assistant', 'sql_preview', 'sql_run') for job in self.jobs):
            return
        data, episode, sql = self.data, self.episode, self.assistant_generated_sql
        job = self._task('sql_run', lambda progress, cancel: McapAnalysisTools(data, episode).query_sql(sql))
        self.assistant_token = job.token
        self.assistant_run_sql.setEnabled(False)
        self.assistant_generate_sql.setEnabled(False)
        self.assistant_cancel.show()
        self.assistant_status.setText('Running validated SQL…')
        job.start()

    def cancel_assistant(self):
        for job in self.jobs:
            if job.kind == 'assistant':
                job.cancelled.set()
        self.assistant_status.setText('Cancelling assistant…')

    def _append_assistant(self, speaker, text):
        self.assistant_transcript.appendPlainText(f'{speaker}: {text}')

    def _show_assistant_trace(self, trace):
        rendered = []
        for item in trace:
            result = item['result']
            if item.get('name') == 'query_sql' and isinstance(result, dict):
                result = dict(result, rows=result.get('rows', [])[:20], preview=True)
            rendered.append(f"{item['name']}({json.dumps(item['arguments'], sort_keys=True)})\n"
                            f"{json.dumps(result, indent=2, sort_keys=True)}")
        self.assistant_query_output.setPlainText('\n\n'.join(rendered))

    def _show_assistant_chart(self, chart):
        self.assistant_plot.clear()
        if not chart:
            return
        self.assistant_plot.setTitle(chart['title'])
        self.assistant_plot.setLabel('bottom', chart['x_label'], units=chart['x_unit'])
        for index, item in enumerate(chart['series']):
            self.assistant_plot.plot(chart['x'], item['values'], pen=pg.mkPen(COLORS[index % len(COLORS)], width=2),
                                     name=f"{item['name']} ({item['unit']})" if item['unit'] else item['name'],
                                     connect='finite')
        self.assistant_plot.enableAutoRange()

    def _show_assistant_result(self, trace):
        sql_result = next((item['result'] for item in reversed(trace)
                           if item.get('name') == 'query_sql' and isinstance(item.get('result'), dict)), None)
        self._show_assistant_table(sql_result)
        if sql_result is None:
            self.assistant_result_status.setText('No SQL table result.')
            return
        self.assistant_result_status.setText(
            f"SQL: {sql_result.get('sql', '')}\nRows: {sql_result.get('row_count', 0)}")
        self._show_assistant_chart(sql_chart(sql_result))
        self._refresh_assistant_result_view()

    def _show_sql_preview(self, result):
        self.assistant_generated_sql = result['sql']
        self.assistant_sql_episode_key = self.episode.key
        self.assistant_query_output.setPlainText('SQL preview (not executed):\n' + result['sql'])
        self.assistant_result_status.setText('SQL validated successfully. Click Run SQL to execute it.')
        self.assistant_run_sql.setEnabled(True)
        self.assistant_status.setText('SQL ready. It has not been executed.')

    def choose_file(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Open flight recording', '', 'MCAP recordings (*.mcap *.partial);;All files (*)')
        if path:
            self.open_file(path)

    def choose_export(self):
        if self.data is None or self.episode is None or any(job.kind == 'export' for job in self.jobs):
            return
        number = self.episodes.currentIndex() + 1
        dialog = ExportDialog(self.data.path.with_name(f'{self.data.path.stem}-episode-{number}.mp4'), self)
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        destination = Path(dialog.destination.text()).expanduser()
        overwrite = False
        if destination.exists():
            overwrite = QtWidgets.QMessageBox.question(self, 'Replace MP4?', f'Replace {destination}?',
                QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
                QtWidgets.QMessageBox.StandardButton.No) == QtWidgets.QMessageBox.StandardButton.Yes
            if not overwrite:
                return
        self.start_export(destination, include_boxes=dialog.include_boxes.isChecked(), overwrite=overwrite)

    def start_export(self, destination, *, include_boxes=False, overwrite=False):
        if self.data is None or self.episode is None or any(job.kind == 'export' for job in self.jobs):
            return
        data, episode = self.data, self.episode
        job = self._task('export', lambda progress, cancel: export_mp4(data, episode, destination,
            include_boxes=include_boxes, overwrite=overwrite, progress=progress, cancel=cancel))
        self.export_token = job.token
        self.export_button.setEnabled(False)
        self.cancel_export_button.show()
        self.status.setText('Preparing MP4 export…')
        job.start()

    def cancel_export(self):
        for job in self.jobs:
            if job.kind == 'export':
                job.cancelled.set()
        self.status.setText('Cancelling MP4 export…')

    def _task(self, kind, operation):
        self.serial += 1
        job = Task(kind, self.serial, operation)
        self.jobs.add(job)
        job.result.connect(lambda result, job=job: self._result(job, result))
        job.failed.connect(lambda message, job=job: self._failed(job, message))
        job.progress.connect(lambda message, job=job: self._set_job_progress(job, message))
        job.finished.connect(lambda job=job: self._finished(job))
        return job

    def _set_job_progress(self, job, message):
        if job.cancelled.is_set():
            return
        if job.token in (self.load_token, self.export_token):
            self.status.setText(message)
        elif job.token == self.assistant_token:
            self.assistant_status.setText(message)

    def open_file(self, path):
        for job in self.jobs:
            job.cancelled.set()
        self.preview_token = -1
        self.export_token = -1
        self.export_button.setEnabled(False)
        self.cancel_export_button.hide()
        self.last_error = ''
        self.content.setEnabled(False)
        self.episodes.setEnabled(False)
        self.cancel_button.show()
        self.status.setText(f'Loading {path} · scalar channels only; images stay on disk')
        job = self._task('load', lambda progress, cancel: load_flight(path, progress, cancel))
        self.load_token = job.token
        job.start()

    def cancel_load(self):
        for job in self.jobs:
            if job.kind == 'load':
                job.cancelled.set()
        self.load_token = -1
        self.cancel_button.hide()
        self.content.setEnabled(True)
        self.episodes.setEnabled(True)
        self.export_button.setEnabled(self.data is not None and not any(job.kind == 'export' for job in self.jobs))
        self.status.setText('Loading cancelled. Previous flight retained.' if self.data else 'Loading cancelled.')

    def _result(self, job, result):
        if self.closing or job.cancelled.is_set():
            return
        if job.kind == 'load' and job.token == self.load_token:
            self.apply_data(result)
        elif job.kind == 'sql_preview' and job.token == self.assistant_token:
            self._show_sql_preview(result)
        elif job.kind == 'sql_run' and job.token == self.assistant_token:
            trace = ({'name': 'query_sql', 'arguments': {'sql': result['sql']}, 'result': result},)
            self._show_assistant_trace(trace)
            self._show_assistant_table(result)
            self._show_assistant_chart(sql_chart(result))
            self._refresh_assistant_result_view()
            self.assistant_result_status.setText(f"SQL executed. Rows: {result['row_count']}")
            self.assistant_status.setText('SQL complete.')
        elif job.kind == 'export' and job.token == self.export_token:
            self.status.setText(f'Exported MP4: {result}')
        elif job.kind == 'frame' and job.token == self.preview_token:
            reference, frame, bbox = result
            self.frame_cache[reference] = frame
            self.frame_cache.move_to_end(reference)
            while len(self.frame_cache) > FRAME_CACHE_SIZE:
                self.frame_cache.popitem(last=False)
            self.canvas.show_frame(frame, bbox)
            self.tracking_tab.show_frame(frame, bbox)
        elif job.kind == 'tracking_frame' and job.token == self.tracking_frame_token:
            reference, frame, bbox = result
            self.frame_cache[reference] = frame
            self.frame_cache.move_to_end(reference)
            while len(self.frame_cache) > FRAME_CACHE_SIZE:
                self.frame_cache.popitem(last=False)
            self.tracking_tab.show_frame(frame, bbox)
        elif job.kind == 'assistant' and job.token == self.assistant_token:
            self._append_assistant('FlightLens', result.text)
            self._show_assistant_trace(result.trace)
            self._show_assistant_chart(result.chart)
            self._show_assistant_result(result.trace)
            self.assistant_status.setText('Ready.')

    def _failed(self, job, message):
        if self.closing or job.cancelled.is_set():
            return
        if job.kind == 'load' and job.token == self.load_token:
            self.last_error = message
            self.status.setText('Could not open recording: ' + message)
            self.cancel_button.hide()
            self.content.setEnabled(True)
            self.episodes.setEnabled(True)
            self.export_button.setEnabled(self.data is not None and not any(j.kind == 'export' for j in self.jobs))
        elif job.kind == 'export' and job.token == self.export_token:
            self.last_error = message
            self.status.setText('Could not export MP4: ' + message)
        elif job.kind == 'frame' and job.token == self.preview_token:
            self.preview_caption.setText('EVENT CAMERA FRAME · Exact record could not be loaded')
            self.canvas.clear_frame('Exact frame unavailable: ' + message)
            self.tracking_tab.clear_timeline_frame('Exact issue frame unavailable: ' + message)
        elif job.kind == 'tracking_frame' and job.token == self.tracking_frame_token:
            self.tracking_tab.clear_timeline_frame('Exact timeline frame unavailable: ' + message)
        elif job.kind == 'assistant' and job.token == self.assistant_token:
            if message != '':
                self._append_assistant('FlightLens error', message)
            self.assistant_status.setText('Assistant failed: ' + message)
        elif job.kind == 'sql_preview' and job.token == self.assistant_token:
            self.assistant_status.setText('SQL generation failed: ' + message)
        elif job.kind == 'sql_run' and job.token == self.assistant_token:
            self.assistant_status.setText('SQL execution failed: ' + message)

    def _finished(self, job):
        self.jobs.discard(job)
        if job.kind == 'export':
            self.cancel_export_button.hide()
            self.export_button.setEnabled(self.data is not None and not any(j.kind == 'load' for j in self.jobs))
            if job.token == self.export_token and job.cancelled.is_set() and not self.closing:
                self.status.setText('MP4 export cancelled. Existing files preserved.')
        elif job.kind == 'assistant':
            self.assistant_cancel.hide()
            self.assistant_submit.setEnabled(self.assistant is not None and self.data is not None)
            if job.token == self.assistant_token and job.cancelled.is_set() and not self.closing:
                self.assistant_status.setText('Assistant request cancelled.')
        elif job.kind in ('sql_preview', 'sql_run'):
            self.assistant_cancel.hide()
            self.assistant_generate_sql.setEnabled(self.assistant is not None and self.data is not None and not any(
                j.kind in ('assistant', 'sql_preview', 'sql_run') for j in self.jobs))
            self.assistant_run_sql.setEnabled(self.assistant_generated_sql is not None and self.assistant_sql_episode_key == (self.episode.key if self.episode else None))
            if job.token == self.assistant_token and job.cancelled.is_set() and not self.closing:
                self.assistant_status.setText('SQL request cancelled.')
        job.deleteLater()
        if self.closing and not self.jobs:
            self.close()

    def apply_data(self, data):
        self.data = data
        self.frame_cache.clear()
        self.cancel_button.hide()
        self.content.setEnabled(True)
        self.episodes.setEnabled(True)
        self.export_button.setEnabled(not any(job.kind == 'export' for job in self.jobs))
        self.file_label.setText(data.path.name + f'  ·  {data.size / 1024**2:.1f} MiB')
        self.file_label.setToolTip(str(data.path))
        self.setWindowTitle('FlightLens · ' + data.path.name)
        with QtCore.QSignalBlocker(self.episodes):
            self.episodes.clear()
            for episode in data.episodes:
                self.episodes.addItem(episode.key[1], episode.key)
                self.episodes.setItemData(self.episodes.count()-1, f"Physics {episode.key[0]}\nEpisode {episode.key[1]}", QtCore.Qt.ItemDataRole.ToolTipRole)
            self.episodes.setCurrentIndex(data.default_episode)
        self.select_episode(data.default_episode)
        self.status.setText(f'Loaded {len(data.episodes)} episode(s) · Read-only · Timeline images load on demand')
        self.assistant_submit.setEnabled(self.assistant is not None and self.episode is not None)
        self.assistant_generate_sql.setEnabled(self.assistant is not None and self.episode is not None)
        self.assistant_generated_sql = None
        self.assistant_sql_episode_key = None
        self.assistant_run_sql.setEnabled(False)
        self.assistant_erd_button.setEnabled(self.data.index_path is not None)
        self.assistant_status.setText('Ready.') if self.assistant is not None else None

    def _curve(self, plot, samples, column, label, color, dashed=False):
        x, y = samples.plot(column)
        pen = pg.mkPen(color, width=2, style=QtCore.Qt.PenStyle.DashLine if dashed else QtCore.Qt.PenStyle.SolidLine)
        curve = plot.plot(x, y, pen=pen, name=label, connect='finite',
                          symbol='o' if len(samples.times) == 1 else None, symbolSize=5, symbolBrush=color)
        curve.setClipToView(True)
        curve.setDownsampling(auto=True, method='peak')
        return curve

    def select_episode(self, index):
        if self.data is None or not 0 <= index < len(self.data.episodes):
            return
        for job in self.jobs:
            if job.kind == 'frame':
                job.cancelled.set()
        self.preview_token = -1
        self.episode = episode = self.data.episodes[index]
        self.tracking_tab.set_episode(episode)
        self.ttc_tab.set_episode(episode)
        if hasattr(self, 'assistant_plot'):
            self.assistant_plot.clear()
            self.assistant_table.clear()
            self.assistant_table.setRowCount(0)
            self.assistant_table.setColumnCount(0)
            self.assistant_result_status.setText('SQL results will appear here.')
            self.assistant_query_output.clear()
            self.assistant_generated_sql = None
            self.assistant_sql_episode_key = None
            self.assistant_run_sql.setEnabled(False)
        summary = episode.summary
        number = lambda value: 'Unavailable' if value is None else f'{value:.2f} m'
        values = {'Outcome': summary.outcome, 'Duration': f'{summary.duration_s:.2f} s',
                  'Peak altitude': number(summary.max_altitude_m), 'First fault': summary.first_fault or 'None recorded',
                  'Warnings / errors': f'{summary.warnings} / {summary.errors}', 'Recording': summary.integrity}
        for title, value in values.items():
            self.cards[title].setText(title + '\n' + value)
        y_range = '' if summary.y_range_m is None else f' · World Y range {summary.y_range_m[0]:+.2f}…{summary.y_range_m[1]:+.2f} m'
        self.notices.setText(episode.trajectory_origin + y_range + '\n' + ' '.join((*self.data.notices, *episode.notices)))
        for plot in (self.trajectory, self.altitude, self.margins, *self.attitude_plots, self.thrust_plot, *self.msp_plots, *self.visual_plots, *self.velocity_plots):
            plot.clear()
        tx, x = episode.trajectory.plot(0)
        _, z = episode.trajectory.plot(2)
        self.trajectory.plot(x, z, pen=pg.mkPen(COLORS[0], width=2), connect='finite')
        if len(episode.trajectory.times):
            points = episode.trajectory.values
            self.trajectory.plot([points[0, 0]], [points[0, 2]], pen=None, symbol='t1', symbolBrush=COLORS[2], symbolSize=12)
            self.trajectory.plot([points[-1, 0]], [points[-1, 2]], pen=None, symbol='s', symbolBrush=COLORS[1], symbolSize=10)
        self.selected_position = self.trajectory.plot([], [], pen=None, symbol='o', symbolBrush='#ffffff', symbolPen=COLORS[0], symbolSize=11)
        self._curve(self.altitude, episode.trajectory, 2, episode.trajectory_origin, COLORS[0])
        self._curve(self.altitude, episode.barometer, 0, 'Barometer altitude', COLORS[1])
        self._curve(self.altitude, episode.goal, 0, 'Takeoff goal · barometer datum', COLORS[2], True)
        for column, name in enumerate(('Left', 'Top', 'Right', 'Bottom')):
            self._curve(self.margins, episode.margins, column, name, COLORS[column])
        self.margins.addItem(pg.InfiniteLine(0, angle=0, pen=pg.mkPen('#ff8282', style=QtCore.Qt.PenStyle.DashLine)))
        self.margins.setTitle(episode.margins_origin + ' edge clearance · zero is boundary contact')
        for column, (plot, name) in enumerate(zip(self.attitude_plots, ('Roll', 'Pitch', 'Yaw'))):
            self._curve(plot, episode.attitude, column, 'Measured ' + name, ATTITUDE_MEASURED_COLOR)
            self._curve(plot, episode.feedback_attitude, column, 'Estimated ' + name, ATTITUDE_ESTIMATED_COLOR, dashed=True)
        has_msp = bool(len(episode.msp_attitude.times))
        self.msp_status.setText('Betaflight MSP attitude · measured at the firmware response timestamps'
                                if has_msp else 'Betaflight MSP attitude unavailable in this recording')
        for column, (plot, name) in enumerate(zip(self.msp_plots, ('Roll', 'Pitch', 'Yaw'))):
            if has_msp:
                self._curve(plot, episode.msp_attitude, column, 'MSP measured ' + name, MSP_MEASURED_COLOR)
                self._curve(plot, episode.msp_attitude_estimate, column, 'Kalman estimate ' + name,
                            MSP_ESTIMATED_COLOR, dashed=True)
            self._curve(plot, episode.attitude, column, 'Physical reference ' + name, ATTITUDE_MEASURED_COLOR)
            self._curve(plot, episode.feedback_attitude, column, 'Controller feedback ' + name,
                        ATTITUDE_ESTIMATED_COLOR, dashed=True)
        if len(episode.ttc_raw.times) or len(episode.ttc_filtered.times):
            if len(episode.ttc_raw.times):
                self._curve(self.visual_plots[0], episode.ttc_raw, 0, 'Raw TTC', VISUAL_RAW_COLOR)
            if len(episode.ttc_filtered.times):
                self._curve(self.visual_plots[0], episode.ttc_filtered, 0, 'Filtered TTC', VISUAL_FILTERED_COLOR, dashed=True)
        if len(episode.camera_error_x.times):
            self._curve(self.visual_plots[1], episode.camera_error_x, 0, 'X error from camera center', VISUAL_RAW_COLOR)
        if len(episode.camera_error_y.times):
            self._curve(self.visual_plots[2], episode.camera_error_y, 0, 'Y error from camera center', VISUAL_RAW_COLOR)
        ttc_label = 'raw and filtered TTC' if len(episode.ttc_raw.times) and len(episode.ttc_filtered.times) \
            else 'filtered TTC' if len(episode.ttc_filtered.times) else 'TTC unavailable'
        ttc_values = np.concatenate((episode.ttc_raw.values[:, 0], episode.ttc_filtered.values[:, 0])) \
            if len(episode.ttc_raw.times) or len(episode.ttc_filtered.times) else np.array([])
        if len(ttc_values):
            display_max = max(1., float(np.percentile(ttc_values, TTC_DISPLAY_PERCENTILE)))
            self.visual_plots[0].setYRange(0., display_max, padding=.05)
            if float(np.max(ttc_values)) > display_max:
                ttc_label += f' · axis clipped above {display_max:.1f} s'
        self.visual_status.setText(f'Visual telemetry · {ttc_label}; camera-center error from recorded detections')
        self._curve(self.thrust_plot, episode.thrust, 0, 'Commanded thrust', COLORS[3])
        for column, (plot, axis) in enumerate(zip(self.velocity_plots, ('X', 'Y', 'Z'))):
            self._curve(plot, episode.velocity, column, 'Ground truth ' + axis, COLORS[column])
            if axis == 'Z':
                self._curve(plot, episode.barometer_velocity, 0, 'Barometer-derived Z', COLORS[3], dashed=True)
        slant_velocity = series((time_s, (speed,)) for time_s, speed in zip(
            episode.velocity.times, np.hypot(episode.velocity.values[:, 0], episode.velocity.values[:, 2])))
        self._curve(self.velocity_plots[3], slant_velocity, 0, 'Ground truth X–Z speed', COLORS[3])
        if episode.tracking_time_s is not None:
            for plot in (*self.attitude_plots, *self.msp_plots, *self.visual_plots, self.thrust_plot):
                plot.addItem(pg.InfiniteLine(episode.tracking_time_s, angle=90,
                                              pen=pg.mkPen('#ffb454', width=2, style=QtCore.Qt.PenStyle.DashLine),
                                              label='Tracking starts',
                                              labelOpts={'color': '#ffb454', 'position': .9}))
        self.cursors = []
        for plot in (self.altitude, self.margins, *self.attitude_plots, self.thrust_plot, *self.msp_plots, *self.visual_plots, *self.velocity_plots):
            line = pg.InfiniteLine(0, angle=90, movable=True, pen=pg.mkPen('#ffffff', width=1))
            plot.addItem(line)
            line.sigPositionChanged.connect(lambda line=line: self.set_cursor(line.value()))
            self.cursors.append(line)
        self.trajectory.enableAutoRange()
        self.altitude.enableAutoRange()
        self.margins.enableAutoRange(axis='y')
        for plot in self.attitude_plots:
            plot.enableAutoRange()
        for plot in self.msp_plots:
            plot.enableAutoRange()
        for plot in self.visual_plots:
            plot.enableAutoRange()
        if len(ttc_values):
            self.visual_plots[0].setYRange(0., display_max, padding=.05)
        self.thrust_plot.enableAutoRange()
        for plot in self.velocity_plots:
            plot.enableAutoRange()
        self.events.setUpdatesEnabled(False)
        with QtCore.QSignalBlocker(self.events):
            self.events.clearSelection()
            self.events.setCurrentCell(-1, -1)
            self.events.setRowCount(len(episode.events))
            for row, event in enumerate(episode.events):
                if event.time_s is None:
                    when = datetime.fromtimestamp(event.arrival_ns / 1e9, timezone.utc).strftime('%H:%M:%S UTC')
                    when += ' · unsynced'
                else:
                    when = f'{event.time_s:.3f} s'
                    if event.end_s is not None and event.end_s > event.time_s:
                        when += f'–{event.end_s:.3f}'
                for column, text in enumerate((when, event.severity.upper(), event.component, event.phase, event.reason)):
                    item = QtWidgets.QTableWidgetItem(text)
                    item.setForeground(QtGui.QColor(SEVERITY[event.severity]))
                    item.setToolTip(text)
                    self.events.setItem(row, column, item)
        self.events.setUpdatesEnabled(True)
        self.canvas.clear_frame('Select an event to inspect its exact camera frame.')
        self.details.clear()
        self.preview_caption.setText('EVENT CAMERA FRAME · No video playback')
        self.set_cursor(float(episode.trajectory.times[0]) if len(episode.trajectory.times) else 0.)
        if episode.events:
            preferred = next((i for i, e in enumerate(episode.events) if e.severity == 'error'),
                             next((i for i, e in enumerate(episode.events) if e.severity == 'warning'), 0))
            self.events.selectRow(preferred)

    def set_cursor(self, time_s):
        if self.episode is None or not np.isfinite(time_s):
            return
        self.cursor_time_s = float(time_s)
        if hasattr(self, 'ttc_tab'):
            self.ttc_tab.set_cursor(time_s)
        for line in self.cursors:
            with QtCore.QSignalBlocker(line):
                line.setValue(time_s)
        samples = self.episode.trajectory
        index = samples.nearest(time_s)
        if index is None:
            self.selected_position.setData([], [])
        else:
            self.selected_position.setData([samples.values[index, 0]], [samples.values[index, 2]])
        context = self.episode.context_at(time_s)
        age = context.get('image_age_s')
        age_text = 'unavailable' if age is None else f'{age * 1000:.1f} ms'
        def flag(name):
            value = context.get(name)
            return 'unavailable' if value is None else 'yes' if value else 'no'
        self.readout.setText(f'Cursor {time_s:.3f} s · {context.get("phase", "phase unavailable")} · Image age {age_text} · Thrust saturated {flag("thrust_saturated")} · Angle limited {flag("angle_limited")} · Slew limited {flag("slew_limited")}')

    def _select_tracking_issue(self, event):
        if self.episode is None:
            return
        row = next((index for index, candidate in enumerate(self.episode.events) if candidate == event), None)
        if row is not None:
            self.events.selectRow(row)

    def _request_tracking_frame(self, reference, bbox):
        if self.data is None or reference is None:
            return
        if reference in self.frame_cache:
            self.frame_cache.move_to_end(reference)
            self.tracking_tab.show_frame(self.frame_cache[reference], bbox)
            return
        for job in self.jobs:
            if job.kind == 'tracking_frame':
                job.cancelled.set()
        self.tracking_tab.clear_timeline_frame('Loading recorded camera frame…')
        job = self._task('tracking_frame',
                         lambda progress, cancel: (reference, read_frame(self.data, reference, cancel), bbox))
        self.tracking_frame_token = job.token
        job.start()

    def select_event(self):
        row = self.events.currentRow()
        if self.episode is None or not 0 <= row < len(self.episode.events):
            return
        event = self.episode.events[row]
        self.events.scrollToItem(self.events.item(row, 0), QtWidgets.QAbstractItemView.ScrollHint.EnsureVisible)
        self.preview_token = -1
        for job in self.jobs:
            if job.kind == 'frame':
                job.cancelled.set()
        if event.time_s is not None:
            self.set_cursor(event.time_s)
        self.details.setPlainText(json.dumps(dict(severity=event.severity, component=event.component,
                                                  reason=event.reason, simulation_time_s=event.time_s,
                                                  details=plain(event.details)), indent=2))
        if event.frame is None:
            self.preview_caption.setText('EVENT CAMERA FRAME · Exact identity unavailable')
            self.canvas.clear_frame('Exact frame unavailable. This event has no matching recorded image identity.')
            self.tracking_tab.clear_timeline_frame('No recorded camera frame for this issue.')
            return
        ref = event.frame
        when = 'unsynced event' if event.time_s is None else f'event at {event.time_s:.3f} s'
        self.preview_caption.setText(f'EXACT EVENT FRAME · {when} · frame {ref.identity[3]}\nPreview stays tied to this event when the cursor moves')
        if ref in self.frame_cache:
            self.canvas.show_frame(self.frame_cache[ref], event.bbox)
            self.tracking_tab.show_frame(self.frame_cache[ref], event.bbox)
            return
        self.canvas.clear_frame('Loading exact event frame…')
        data = self.data
        job = self._task('frame', lambda progress, cancel: (ref, read_frame(data, ref, cancel), event.bbox))
        self.preview_token = job.token
        job.start()

    def closeEvent(self, event):
        self.closing = True
        for job in self.jobs:
            job.cancelled.set()
        if self.jobs:
            event.ignore()
            self.status.setText('Waiting for background tasks to stop…')
        else:
            event.accept()


def main():
    parser = argparse.ArgumentParser(description='FlightLens · read-only MCAP flight overview')
    parser.add_argument('recording', nargs='?')
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG_PATH,
                        help='YAML settings file')
    parser.add_argument('--assistant-model', help='Path to a local GGUF model for the Assistant tab')
    args = parser.parse_args()
    try:
        config = load_config(args.config)
    except ValueError as error:
        parser.error(str(error))
    application = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    application.setApplicationName('FlightLens')
    window = Overview(args.assistant_model or config['assistant_model'])
    window.show()
    if args.recording:
        QtCore.QTimer.singleShot(0, lambda: window.open_file(args.recording))
    return application.exec()


if __name__ == '__main__':
    raise SystemExit(main())
