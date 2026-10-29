"""One desktop overview backed by immutable, independently testable flight data."""
import argparse
from collections import OrderedDict
from collections.abc import Mapping
from datetime import datetime, timezone
import json
from pathlib import Path
from threading import Event as CancelEvent

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg
import yaml

from .reader import Cancelled, load_flight, read_frame
from .model import series
from .video import export_mp4
from .assistant import LlamaCppBackend, McapAnalysisTools, McapAssistant

COLORS = ['#58b8f5', '#eab85a', '#70d6ad', '#c299f0']
ATTITUDE_MEASURED_COLOR = '#58b8f5'
ATTITUDE_ESTIMATED_COLOR = '#ff8282'
MSP_MEASURED_COLOR = '#70d6ad'
MSP_ESTIMATED_COLOR = '#c299f0'
VISUAL_RAW_COLOR = '#eab85a'
VISUAL_FILTERED_COLOR = '#70d6ad'
TTC_DISPLAY_PERCENTILE = 95.
SEVERITY = {'info': '#9bb0c8', 'warning': '#eab85a', 'error': '#ff8282'}
FRAME_CACHE_SIZE = 3
DEFAULT_CONFIG_PATH = Path('config/flight_lens.yaml')
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

class Overview(QtWidgets.QMainWindow):
    def __init__(self, assistant_model=None):
        super().__init__()
        self.assistant_model = assistant_model
        self.assistant = None if assistant_model is None else McapAssistant(LlamaCppBackend(assistant_model))
        self.data = self.episode = None
        self.jobs = set()
        self.serial = self.load_token = self.preview_token = 0
        self.export_token = -1
        self.assistant_token = -1
        self.assistant_generated_sql = None
        self.assistant_sql_episode_key = None
        self.closing = False
        self.frame_cache = OrderedDict()
        self.last_error = ''
        self.cursor_time_s = 0.
        self.setWindowTitle('FlightLens · Flight Overview')
        self.resize(1480, 980)
        self.setMinimumSize(1100, 760)
        pg.setConfigOptions(background='#101c2a', foreground='#c8d5e6', antialias=False, useOpenGL=False)
        self.setStyleSheet('''
            QMainWindow, QWidget { background:#142233; color:#d9e6f4; font:13px sans-serif; }
            QPushButton { background:#284c68; padding:8px 15px; border:1px solid #426983; border-radius:5px; }
            QPushButton:hover { background:#356381; } QPushButton:disabled { color:#6f8298; }
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
        self.velocity_tab = QtWidgets.QWidget()
        self.tabs.addTab(self.velocity_tab, 'Velocity')
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
        self.status.setText(f'Loaded {len(data.episodes)} episode(s) · Read-only · Images loaded only for selected events')
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
            return
        ref = event.frame
        when = 'unsynced event' if event.time_s is None else f'event at {event.time_s:.3f} s'
        self.preview_caption.setText(f'EXACT EVENT FRAME · {when} · frame {ref.identity[3]}\nPreview stays tied to this event when the cursor moves')
        if ref in self.frame_cache:
            self.canvas.show_frame(self.frame_cache[ref], event.bbox)
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
