"""Persistent, read-only query index for validated FlightLens telemetry."""
import sqlite3
from pathlib import Path

import numpy as np

INDEX_VERSION = 1
SIDECAR_SUFFIX = '.flightlens.sqlite'


SCHEMA = """
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS episodes (
    id INTEGER PRIMARY KEY, physics_id TEXT NOT NULL, episode_id TEXT NOT NULL,
    duration_s REAL, outcome TEXT, UNIQUE(physics_id, episode_id));
CREATE TABLE IF NOT EXISTS telemetry (
    episode_id INTEGER NOT NULL, topic TEXT NOT NULL, path TEXT NOT NULL,
    time_s REAL NOT NULL, value REAL NOT NULL, source TEXT NOT NULL DEFAULT 'raw');
CREATE TABLE IF NOT EXISTS events (
    episode_id INTEGER NOT NULL, time_s REAL, end_time_s REAL,
    severity TEXT NOT NULL, component TEXT NOT NULL, reason TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS telemetry_lookup ON telemetry(episode_id, topic, path, time_s);
CREATE INDEX IF NOT EXISTS events_lookup ON events(episode_id, severity, time_s);
"""


def sidecar_path(recording):
    recording = Path(recording)
    return recording.with_name(recording.name + SIDECAR_SUFFIX)


def _flatten(value, prefix=''):
    if isinstance(value, dict):
        for key, child in value.items():
            yield from _flatten(child, f'{prefix}.{key}' if prefix else key)
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            yield from _flatten(child, f'{prefix}[{index}]')
    elif isinstance(value, (int, float)) and not isinstance(value, bool) and np.isfinite(value):
        yield prefix, float(value)


def _episode_rows(episodes):
    return [(physics, episode_id, episode.summary.duration_s, episode.summary.outcome)
            for episode in episodes for physics, episode_id in [episode.key]]


def build_index(recording, records, episodes):
    recording = Path(recording).resolve()
    target = sidecar_path(recording)
    if target.exists():
        raise ValueError(f'Stale or existing index must be removed before rebuilding: {target}')
    connection = sqlite3.connect(target)
    try:
        connection.executescript(SCHEMA)
        stat = recording.stat()
        connection.executemany('INSERT INTO metadata(key,value) VALUES (?,?)',
                               [('version', str(INDEX_VERSION)), ('path', str(recording)),
                                ('size_bytes', str(stat.st_size)), ('modified_ns', str(stat.st_mtime_ns))])
        lookup = {}
        for physics, episode_id, duration, outcome in _episode_rows(episodes):
            cursor = connection.execute('INSERT INTO episodes(physics_id,episode_id,duration_s,outcome) VALUES (?,?,?,?)',
                                         (physics, episode_id, duration, outcome))
            lookup[(physics, episode_id)] = cursor.lastrowid
        rows = []
        for topic, _, value in records:
            source = value.get('source', value.get('final_source', value))
            episode_id = lookup.get((source.get('physics_id'), source.get('episode_id')))
            if episode_id is None:
                continue
            time_s = source.get('simulation_time_s')
            if time_s is None:
                continue
            for path, numeric in _flatten(value):
                rows.append((episode_id, topic, path, time_s, numeric, 'raw'))
        for episode in episodes:
            episode_id = lookup[episode.key]
            derived_series = {
                'trajectory_x': (episode.trajectory, 0), 'trajectory_y': (episode.trajectory, 1),
                'trajectory_z': (episode.trajectory, 2), 'velocity_x': (episode.velocity, 0),
                'velocity_y': (episode.velocity, 1), 'velocity_z': (episode.velocity, 2),
                'barometer_altitude': (episode.barometer, 0), 'vertical_speed': (episode.barometer_velocity, 0),
                'thrust': (episode.thrust, 0), 'ttc_raw': (episode.ttc_raw, 0),
                'ttc_filtered': (episode.ttc_filtered, 0), 'camera_error_x': (episode.camera_error_x, 0),
                'camera_error_y': (episode.camera_error_y, 0),
                'attitude_roll': (episode.attitude, 0), 'attitude_pitch': (episode.attitude, 1), 'attitude_yaw': (episode.attitude, 2),
                'feedback_attitude_roll': (episode.feedback_attitude, 0), 'feedback_attitude_pitch': (episode.feedback_attitude, 1),
                'feedback_attitude_yaw': (episode.feedback_attitude, 2),
                'msp_attitude_roll': (episode.msp_attitude, 0), 'msp_attitude_pitch': (episode.msp_attitude, 1), 'msp_attitude_yaw': (episode.msp_attitude, 2),
                'msp_kalman_roll': (episode.msp_attitude_estimate, 0), 'msp_kalman_pitch': (episode.msp_attitude_estimate, 1), 'msp_kalman_yaw': (episode.msp_attitude_estimate, 2)}
            for name, (samples, column) in derived_series.items():
                rows.extend((episode_id, 'derived', name, float(time_s), float(value), 'derived')
                            for time_s, value in zip(samples.times, samples.values[:, column]))
        connection.executemany('INSERT INTO telemetry VALUES (?,?,?,?,?,?)', rows)
        for episode in episodes:
            episode_id = lookup[episode.key]
            connection.executemany('INSERT INTO events VALUES (?,?,?,?,?,?)',
                                   [(episode_id, event.time_s, event.end_s, event.severity, event.component, event.reason)
                                    for event in episode.events])
        connection.commit()
    finally:
        connection.close()
    return target


def validate_index(recording):
    recording = Path(recording).resolve()
    target = sidecar_path(recording)
    if not target.exists():
        return None
    stat = recording.stat()
    with sqlite3.connect(target) as connection:
        metadata = dict(connection.execute('SELECT key,value FROM metadata'))
    if metadata.get('version') != str(INDEX_VERSION) or metadata.get('path') != str(recording) \
            or metadata.get('size_bytes') != str(stat.st_size) or metadata.get('modified_ns') != str(stat.st_mtime_ns):
        raise ValueError(f'Index is stale for {recording}; remove {target} and reopen it.')
    return target


def query_fields(index_path, episode_key, fields, start=None, end=None, limit=500, aggregation='raw'):
    if not fields or len(fields) > 8 or aggregation not in ('raw', 'mean', 'min', 'max', 'rms'):
        raise ValueError('invalid bounded telemetry query')
    start = -float('inf') if start is None else float(start)
    end = float('inf') if end is None else float(end)
    if start > end or not np.isfinite(start) and start != -float('inf') or not np.isfinite(end) and end != float('inf'):
        raise ValueError('invalid query time bounds')
    limit = min(500, max(1, int(limit)))
    with sqlite3.connect(index_path) as connection:
        episode = connection.execute('SELECT id FROM episodes WHERE physics_id=? AND episode_id=?', episode_key).fetchone()
        if episode is None:
            raise ValueError('unknown episode')
        result = []
        for field in fields:
            if not isinstance(field, dict) or not isinstance(field.get('topic'), str) or not isinstance(field.get('path'), str):
                raise ValueError('invalid telemetry field')
            if aggregation == 'raw':
                sql = 'SELECT time_s,value FROM telemetry WHERE episode_id=? AND topic=? AND path=? AND time_s BETWEEN ? AND ? ORDER BY time_s LIMIT ?'
                rows = connection.execute(sql, (episode[0], field['topic'], field['path'], start, end, limit)).fetchall()
            else:
                function = {'mean': 'AVG', 'min': 'MIN', 'max': 'MAX', 'rms': 'AVG'}[aggregation]
                sql = f'SELECT {function}(value) FROM telemetry WHERE episode_id=? AND topic=? AND path=? AND time_s BETWEEN ? AND ?'
                value = connection.execute(sql, (episode[0], field['topic'], field['path'], start, end)).fetchone()[0]
                rows = [] if value is None else [(start, float(np.sqrt(value)) if aggregation == 'rms' else float(value))]
            result.append(dict(name=field.get('name') or field['path'], topic=field['topic'], path=field['path'],
                               times=[row[0] for row in rows], values=[row[1] for row in rows]))
    return result


def list_fields(index_path, episode_key, limit=2000):
    with sqlite3.connect(index_path) as connection:
        episode = connection.execute('SELECT id FROM episodes WHERE physics_id=? AND episode_id=?', episode_key).fetchone()
        if episode is None:
            raise ValueError('unknown episode')
        return [dict(topic=topic, path=path) for topic, path in connection.execute(
            'SELECT DISTINCT topic,path FROM telemetry WHERE episode_id=? ORDER BY topic,path LIMIT ?',
            (episode[0], min(2000, max(1, int(limit)))))]


def read_derived(index_path, episode_key, names):
    with sqlite3.connect(index_path) as connection:
        episode = connection.execute('SELECT id FROM episodes WHERE physics_id=? AND episode_id=?', episode_key).fetchone()
        if episode is None:
            raise ValueError('unknown episode')
        return {name: connection.execute(
            'SELECT time_s,value FROM telemetry WHERE episode_id=? AND topic=? AND path=? ORDER BY time_s',
            (episode[0], 'derived', name)).fetchall() for name in names}
