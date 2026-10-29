import time
from threading import Event

import numpy as np
import pytest
from PySide6 import QtWidgets

from flight_lens.app import ErdDialog, ExportDialog, Overview, load_config, sql_chart
from flight_lens.reader import load_flight, Cancelled


@pytest.fixture(scope='session')
def application():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def until(application,predicate,timeout=5):
    deadline=time.monotonic()+timeout
    while not predicate():
        application.processEvents()
        assert time.monotonic()<deadline,'Qt operation did not complete'
        time.sleep(.002)
    application.processEvents()


def test_load_config_reads_assistant_model(tmp_path):
    path = tmp_path / 'flight_lens.yaml'
    path.write_text('assistant-model: /models/flight.gguf\n')

    assert load_config(path) == {'assistant_model': '/models/flight.gguf'}


def test_sql_result_table_graph_and_erd(application):
    result = {'sql': 'SELECT time_s, value FROM selected_telemetry LIMIT 2',
              'columns': ['time_s', 'value'], 'rows': [[0.0, 1.0], [1.0, 2.0]],
              'row_count': 2, 'truncated': False}
    assert sql_chart(result)['series'][0]['values'] == [1.0, 2.0]
    window = Overview()
    window._show_assistant_result(({'name': 'query_sql', 'result': result},))
    assert window.assistant_table.rowCount() == 2
    window.assistant_graph.setCurrentText('Graph')
    application.processEvents()
    assert not window.assistant_table.isVisible()
    dialog = ErdDialog(window)
    view = dialog.findChild(QtWidgets.QGraphicsView)
    assert 'selected_telemetry' in '\n'.join(item.toPlainText() for item in view.scene().items()
                                               if isinstance(item, QtWidgets.QGraphicsTextItem))
    dialog.close()
    window.close()


@pytest.fixture
def window(application):
    view=Overview()
    view.show()
    yield view
    view.close()
    until(application,lambda:not view.jobs)
    application.processEvents()
    view.deleteLater()
    application.processEvents()


def test_overview_async_loading_cursor_event_and_frame(window,application,flight_file):
    window.open_file(flight_file)
    until(application,lambda:window.data is not None and not window.jobs)
    assert window.tabs.count()==6
    assert window.tabs.tabText(2)=='MSP'
    assert window.tabs.tabText(3)=='Visual'
    assert window.tabs.tabText(5)=='Assistant'
    assert window.assistant_query_output.isReadOnly()
    assert window.assistant_submit.text()=='Run query'
    assert window.assistant_prompt.toPlainText() == ''
    assert window.assistant_generate_sql.text() == 'Generate SQL'
    assert not window.assistant_run_sql.isEnabled()
    assert len(window.visual_plots)==3
    assert len(window.visual_plots[0].plotItem.legend.items)==1
    assert window.events.rowCount()==len(window.episode.events)
    assert 'Mission fault' in window.cards['Outcome'].text()
    assert window.canvas.source_image is not None
    # Auto-focus first fault and show its exact recorded exposure.
    row=window.events.currentRow()
    assert window.episode.events[row].severity=='error'
    assert window.cursor_time_s==.4
    assert all(line.value()==.4 for line in window.cursors)
    window.cursors[0].setValue(.1)
    application.processEvents()
    assert window.cursors[1].value()==.1
    x,z=window.selected_position.getData()
    np.testing.assert_allclose(x,[1.])
    np.testing.assert_allclose(z,[2.])
    assert '30.0 ms' in window.readout.text()
    window.events.selectRow(next(i for i,e in enumerate(window.episode.events) if e.component=='vision'))
    application.processEvents()
    assert window.cursor_time_s==.1
    assert 'clipped_target' in window.details.toPlainText()
    # Applying a new dataset replaces curves, legends and event selection.
    window.apply_data(window.data)
    assert len(window.altitude.plotItem.legend.items)==3
    goal_curve=next(curve for curve in window.altitude.listDataItems() if curve.name()=='Takeoff goal · barometer datum')
    assert goal_curve.opts['symbol']=='o'
    assert len(window.margins.plotItem.legend.items)==4
    assert len(window.attitude_plots)==3
    assert all(len(plot.plotItem.legend.items)==2 for plot in window.attitude_plots)
    for plot in window.attitude_plots:
        measured = next(curve for curve in plot.listDataItems() if curve.name().startswith('Measured '))
        estimated = next(curve for curve in plot.listDataItems() if curve.name().startswith('Estimated '))
        assert measured.opts['pen'].color().name() != estimated.opts['pen'].color().name()
    assert len(window.msp_plots)==3
    assert all(len(plot.plotItem.legend.items)==4 for plot in window.msp_plots)
    assert 'measured at the firmware response timestamps' in window.msp_status.text()
    assert len(window.thrust_plot.plotItem.legend.items)==1
    assert len(window.velocity_plots)==4
    assert all(len(plot.plotItem.legend.items)==1 for plot in window.velocity_plots[:2])
    assert len(window.velocity_plots[2].plotItem.legend.items)==2
    slant_curve=next(curve for curve in window.velocity_plots[3].listDataItems() if curve.name()=='Ground truth X–Z speed')
    np.testing.assert_allclose(slant_curve.getData()[1], [0.,5.])


def test_invalid_file_keeps_existing_flight(window,application,flight_file,tmp_path):
    window.apply_data(load_flight(flight_file))
    old=window.data
    bad=tmp_path/'broken.mcap'
    bad.write_bytes(b'not an MCAP')
    window.open_file(bad)
    until(application,lambda:bool(window.last_error) and not window.jobs)
    assert window.data is old
    assert 'Could not open recording' in window.status.text()
    assert window.content.isEnabled()


def test_replaced_load_and_cancel_ignore_obsolete_results(window,application,flight_file,monkeypatch):
    import flight_lens.app as module
    result=load_flight(flight_file)
    entered=Event()
    def slow(path,progress,cancel):
        if path=='slow':
            entered.set()
            while not cancel():
                time.sleep(.002)
            # Even an operation returning after cancellation cannot replace the newer load.
            return result
        return result
    monkeypatch.setattr(module,'load_flight',slow)
    window.open_file('slow')
    until(application,entered.is_set)
    window.open_file('fast')
    until(application,lambda:window.data is not None and not window.jobs)
    assert window.data is result
    entered.clear()
    window.open_file('slow')
    until(application,entered.is_set)
    window.cancel_load()
    until(application,lambda:not window.jobs)
    assert window.data is result
    assert 'cancelled' in window.status.text()


def test_close_cancels_running_worker(window,application,monkeypatch):
    import flight_lens.app as module
    entered=Event()
    def slow(path,progress,cancel):
        entered.set()
        while not cancel():
            time.sleep(.002)
        raise Cancelled()
    monkeypatch.setattr(module,'load_flight',slow)
    window.open_file('slow')
    until(application,entered.is_set)
    window.close()
    until(application,lambda:not window.jobs)
    assert not window.isVisible()


def test_image_error_is_explicit_and_time_zoom_is_shared(window,application,flight_file,monkeypatch):
    import flight_lens.app as module
    def unavailable(*args,**kwargs):
        raise ValueError('image record missing')
    monkeypatch.setattr(module,'read_frame',unavailable)
    window.apply_data(load_flight(flight_file))
    until(application,lambda:not window.jobs)
    assert 'could not be loaded' in window.preview_caption.text()
    assert window.canvas.source_image is None
    window.altitude.setXRange(0.,.2,padding=0)
    application.processEvents()
    assert window.margins.viewRange()[0]==pytest.approx(window.altitude.viewRange()[0])


def test_export_dialog_and_background_export(window, application, flight_file, tmp_path, monkeypatch):
    import flight_lens.app as module
    dialog = ExportDialog(tmp_path/'movie.mp4', window)
    assert not dialog.include_boxes.isChecked()
    dialog.include_boxes.setChecked(True)
    assert dialog.include_boxes.isChecked()
    calls = []
    def export(data, episode, destination, **kwargs):
        calls.append((data, episode, destination, kwargs['include_boxes']))
        kwargs['progress']('Exporting MP4: 50%')
        return destination
    monkeypatch.setattr(module, 'export_mp4', export)
    assert not window.export_button.isEnabled()
    window.apply_data(load_flight(flight_file))
    until(application, lambda: not window.jobs)
    assert window.export_button.isEnabled()
    window.start_export(tmp_path/'movie.mp4', include_boxes=True)
    until(application, lambda: not window.jobs)
    assert calls[0][1] is window.episode and calls[0][3]
    assert 'Exported MP4' in window.status.text()
    assert window.export_button.isEnabled()
    assert window.cancel_export_button.isHidden()


@pytest.mark.parametrize('action', ['cancel', 'open', 'close'])
def test_export_cancel_reopen_and_close(window, application, flight_file, tmp_path, monkeypatch, action):
    import flight_lens.app as module
    entered = Event()
    def export(data, episode, destination, **kwargs):
        entered.set()
        while not kwargs['cancel']():
            time.sleep(.002)
        raise Cancelled()
    monkeypatch.setattr(module, 'export_mp4', export)
    window.apply_data(load_flight(flight_file))
    until(application, lambda: not window.jobs)
    window.start_export(tmp_path/'movie.mp4')
    until(application, entered.is_set)
    assert not window.export_button.isEnabled()
    window.start_export(tmp_path/'second.mp4')
    assert sum(job.kind == 'export' for job in window.jobs) == 1
    if action == 'cancel':
        window.cancel_export()
    elif action == 'open':
        window.open_file(flight_file)
    else:
        window.close()
    until(application, lambda: not window.jobs)
    if action == 'cancel':
        assert 'export cancelled' in window.status.text()
    elif action == 'open':
        assert window.data.path == flight_file
        assert 'Loaded' in window.status.text()
    else:
        assert not window.isVisible()


def test_export_failure_is_visible(window, application, flight_file, tmp_path, monkeypatch):
    import flight_lens.app as module
    def fail(*args, **kwargs):
        raise ValueError('encoder unavailable')
    monkeypatch.setattr(module, 'export_mp4', fail)
    window.apply_data(load_flight(flight_file))
    until(application, lambda: not window.jobs)
    window.start_export(tmp_path/'movie.mp4')
    until(application, lambda: not window.jobs)
    assert 'Could not export MP4: encoder unavailable' in window.status.text()
    assert window.export_button.isEnabled()
