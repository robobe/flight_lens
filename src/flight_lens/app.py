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

from .reader import Cancelled, load_flight, read_frame
from .video import export_mp4

COLORS = ['#58b8f5', '#eab85a', '#70d6ad', '#c299f0']
SEVERITY = {'info': '#9bb0c8', 'warning': '#eab85a', 'error': '#ff8282'}
FRAME_CACHE_SIZE = 3


def plain(value):
    if isinstance(value, Mapping):
        return {k: plain(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [plain(v) for v in value]
    return value


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


class Overview(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.data = self.episode = None
        self.jobs = set()
        self.serial = self.load_token = self.preview_token = 0
        self.export_token = -1
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
        self.velocity_tab = QtWidgets.QWidget()
        self.tabs.addTab(self.velocity_tab, 'Velocity')
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
        self.attitude_plot = pg.PlotWidget(title='Attitude · actual and controller feedback')
        self.attitude_plot.setLabel('left', 'Angle', units='deg')
        self.thrust_plot = pg.PlotWidget(title='Commanded collective thrust')
        self.thrust_plot.setLabel('left', 'Commanded thrust', units='%')
        for plot in (self.attitude_plot, self.thrust_plot):
            plot.getAxis('left').setWidth(76)
            plot.setLabel('bottom', 'Simulation time', units='s')
            plot.showGrid(x=True, y=True, alpha=.15)
            plot.addLegend(offset=(10, 5), labelTextSize='10pt')
            controls_layout.addWidget(plot)
        self.thrust_plot.setXLink(self.attitude_plot)
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
        self.velocity_plots[1].setXLink(self.velocity_plots[0])
        self.velocity_plots[2].setXLink(self.velocity_plots[0])
        self.cursors = []
        self.selected_position = None

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
        job.progress.connect(lambda message, job=job: self.status.setText(message)
                             if job.token in (self.load_token, self.export_token) and not job.cancelled.is_set() else None)
        job.finished.connect(lambda job=job: self._finished(job))
        return job

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
        elif job.kind == 'export' and job.token == self.export_token:
            self.status.setText(f'Exported MP4: {result}')
        elif job.kind == 'frame' and job.token == self.preview_token:
            reference, frame, bbox = result
            self.frame_cache[reference] = frame
            self.frame_cache.move_to_end(reference)
            while len(self.frame_cache) > FRAME_CACHE_SIZE:
                self.frame_cache.popitem(last=False)
            self.canvas.show_frame(frame, bbox)

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

    def _finished(self, job):
        self.jobs.discard(job)
        if job.kind == 'export':
            self.cancel_export_button.hide()
            self.export_button.setEnabled(self.data is not None and not any(j.kind == 'load' for j in self.jobs))
            if job.token == self.export_token and job.cancelled.is_set() and not self.closing:
                self.status.setText('MP4 export cancelled. Existing files preserved.')
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
        summary = episode.summary
        number = lambda value: 'Unavailable' if value is None else f'{value:.2f} m'
        values = {'Outcome': summary.outcome, 'Duration': f'{summary.duration_s:.2f} s',
                  'Peak altitude': number(summary.max_altitude_m), 'First fault': summary.first_fault or 'None recorded',
                  'Warnings / errors': f'{summary.warnings} / {summary.errors}', 'Recording': summary.integrity}
        for title, value in values.items():
            self.cards[title].setText(title + '\n' + value)
        y_range = '' if summary.y_range_m is None else f' · World Y range {summary.y_range_m[0]:+.2f}…{summary.y_range_m[1]:+.2f} m'
        self.notices.setText(episode.trajectory_origin + y_range + '\n' + ' '.join((*self.data.notices, *episode.notices)))
        for plot in (self.trajectory, self.altitude, self.margins, self.attitude_plot, self.thrust_plot, *self.velocity_plots):
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
        for column, name in enumerate(('Roll', 'Pitch', 'Yaw')):
            self._curve(self.attitude_plot, episode.attitude, column, name, COLORS[column])
            self._curve(self.attitude_plot, episode.feedback_attitude, column, 'Estimated ' + name, COLORS[column], dashed=True)
        self._curve(self.thrust_plot, episode.thrust, 0, 'Commanded thrust', COLORS[3])
        for column, (plot, axis) in enumerate(zip(self.velocity_plots, ('X', 'Y', 'Z'))):
            self._curve(plot, episode.velocity, column, 'Ground truth ' + axis, COLORS[column])
            if axis == 'Z':
                self._curve(plot, episode.barometer_velocity, 0, 'Barometer-derived Z', COLORS[3], dashed=True)
        if episode.tracking_time_s is not None:
            for plot in (self.attitude_plot, self.thrust_plot):
                plot.addItem(pg.InfiniteLine(episode.tracking_time_s, angle=90,
                                              pen=pg.mkPen('#ffb454', width=2, style=QtCore.Qt.PenStyle.DashLine),
                                              label='Tracking starts',
                                              labelOpts={'color': '#ffb454', 'position': .9}))
        self.cursors = []
        for plot in (self.altitude, self.margins, self.attitude_plot, self.thrust_plot, *self.velocity_plots):
            line = pg.InfiniteLine(0, angle=90, movable=True, pen=pg.mkPen('#ffffff', width=1))
            plot.addItem(line)
            line.sigPositionChanged.connect(lambda line=line: self.set_cursor(line.value()))
            self.cursors.append(line)
        self.trajectory.enableAutoRange()
        self.altitude.enableAutoRange()
        self.margins.enableAutoRange(axis='y')
        self.attitude_plot.enableAutoRange()
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
    args = parser.parse_args()
    application = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    application.setApplicationName('FlightLens')
    window = Overview()
    window.show()
    if args.recording:
        QtCore.QTimer.singleShot(0, lambda: window.open_file(args.recording))
    return application.exec()


if __name__ == '__main__':
    raise SystemExit(main())
