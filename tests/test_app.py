import time
from threading import Event

import numpy as np
import pytest
from PySide6 import QtWidgets

from flight_lens.app import Overview
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
    assert window.tabs.count()==1
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
