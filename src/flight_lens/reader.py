"""Read indexed MCAP scalar channels first; fetch individual images on demand."""
from collections import defaultdict
from dataclasses import replace
import json
from math import asin, atan2, isfinite, sqrt
from pathlib import Path
from types import MappingProxyType

import numpy as np
from bullet_msgs import schema_bytes
from foxglove_schemas_protobuf.RawImage_pb2 import RawImage
from jsonschema import Draft202012Validator, ValidationError
from mcap.reader import make_reader

from .model import Episode, Event, FlightData, Frame, FrameRef, Summary, series

TRUTH = 'analysis/physics/ground_truth'
SENSORS = 'analysis/physics/sensors'
POSE = 'render/drone_1/pose'
MISSION = 'app/drone_1/mission'
BBOX = 'tracking/front/bbox'
FEEDBACK = 'flight/drone_1/feedback'
RC = 'flight/drone_1/rc'
METADATA = 'camera/front/image/metadata'
STATE = 'simulation/state'
END = 'scenario/ended'
PLATFORM = 'flight/drone_1/config'
RENDER_HEALTH = 'analysis/render/health'
SCHEMAS = {TRUTH: 'analysis_ground_truth.json', SENSORS: 'analysis_sensors.json', POSE: 'pose.json',
           MISSION: 'mission.json', BBOX: 'bbox.json', FEEDBACK: 'flight_feedback.json', RC: 'flight_rc.json',
           METADATA: 'image_metadata.json', STATE: 'state.json', END: 'scenario_end.json',
           PLATFORM: 'flight_config.json', RENDER_HEALTH: 'render_health.json'}
TAKEOFF_PHASES = {'TAKEOFF', 'HOVER', 'WAIT_TARGET', 'YAW_SEARCH', 'YAW_CENTER', 'YAW_SETTLE'}
VISION_WARNINGS = {'clipped_target', 'stale_image', 'stale_or_foreign_image', 'image_deadline',
                   'attitude_unaligned', 'bearing_outside_envelope', 'association_jump', 'los_deviation'}
MAX_JSON_BYTES = 1024 * 1024
MAX_IMAGE_BYTES = 64 * 1024 * 1024


class Cancelled(Exception):
    pass


def check_cancel(cancel):
    if cancel():
        raise Cancelled()


def invalid_constant(value):
    raise ValueError('nonfinite JSON value: ' + value)


def finite_float(value):
    result = float(value)
    if not isfinite(result):
        raise ValueError('nonfinite JSON number')
    return result


def decode(payload):
    value = json.loads(payload, parse_constant=invalid_constant, parse_float=finite_float)
    if not isinstance(value, dict):
        raise ValueError('expected a JSON object')
    return value


def freeze(value):
    if isinstance(value, dict):
        return MappingProxyType({k: freeze(v) for k, v in value.items()})
    if isinstance(value, list):
        return tuple(freeze(v) for v in value)
    return value


def frame_identity(source, sequence, measurement_us=None):
    stamp = None if not isinstance(source, dict) else source.get('measurement_us') if measurement_us is None else measurement_us
    if (not isinstance(source, dict) or type(sequence) is not int or sequence < 0
            or type(stamp) is not int or stamp < 0
            or not all(isinstance(source.get(key), str) and source[key] for key in ('physics_id', 'episode_id', 'stream_id'))):
        return None
    return (source.get('physics_id'), source.get('episode_id'), source.get('stream_id'),
            sequence, stamp)


def mission_image(value):
    d = value.get('diagnostics', {})
    timestamp = d.get('processed_image_timestamp_s')
    if type(timestamp) in (int, float) and timestamp >= 0 and isfinite(timestamp) and d.get('processed_image_sequence') is not None:
        source = dict(physics_id=value['physics_id'], episode_id=value['episode_id'],
                      stream_id=d.get('processed_image_stream_id'))
        bbox = d.get('processed_bbox')
        valid = isinstance(bbox, (list, tuple)) and len(bbox) == 4 and all(type(v) in (int, float) and isfinite(v) for v in bbox)
        bbox = tuple(bbox) if valid and bbox[2] > 0 and bbox[3] > 0 else None
        return frame_identity(source, d['processed_image_sequence'], round(timestamp * 1e6)), bbox
    latest = d.get('latest_image', {})
    if not isinstance(latest, dict):
        return None, None
    return frame_identity(latest.get('source'), latest.get('sequence')), None


def pose_rows(records, topic):
    return [(t, v['ground_truth']['position_m'] if topic == TRUTH else v['position_m'])
            for kind, t, _, v in records if kind == topic and t is not None]


def attitude(quaternion):
    try:
        x, y, z, w = (float(value) for value in quaternion)
        norm = sqrt(x * x + y * y + z * z + w * w)
        if norm <= 0:
            return None
        x, y, z, w = (value / norm for value in (x, y, z, w))
        pitch = max(-1., min(1., 2 * (w * y - z * x)))
        return tuple(np.degrees((atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)),
                                 asin(pitch), atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))))
    except (TypeError, ValueError, OverflowError):
        return None


def attitude_rows(records, topic):
    rows = []
    for kind, t, _, value in records:
        if kind != topic or t is None:
            continue
        quaternion = value.get('ground_truth', {}).get('orientation_xyzw') if topic == TRUTH else value.get('orientation_xyzw')
        angles = attitude(quaternion)
        if angles is not None:
            rows.append((t, angles))
    return rows


def unwrap_attitude(rows):
    if not rows:
        return rows
    times = [t for t, _ in rows]
    values = np.array([angles for _, angles in rows], dtype=float)
    values = np.degrees(np.unwrap(np.radians(values), axis=0))
    return list(zip(times, values))


def thrust_value(channel, backend):
    if type(channel) is not int or not 1000 <= channel <= 2000:
        return None
    if backend != 'betaflight':
        return (channel - 1000) / 10.
    if channel == 1000:
        return 0.
    stick = (channel - 1050) / 950.
    fraction = .055 + .945 * max(0., min(1., stick))
    return 100. * fraction * fraction


def build_episode(key, records, frames, boxes, camera, health, partial, backend='internal'):
    truth = pose_rows(records, TRUTH)
    origin = 'Actual ground truth' if truth else 'Rendered pose (may use noisy sensor sources)'
    trajectory = series(truth or pose_rows(records, POSE), 3)
    attitude_rows_actual = unwrap_attitude(attitude_rows(records, TRUTH) or attitude_rows(records, POSE))
    estimated_rows = unwrap_attitude(attitude_rows(records, FEEDBACK))
    barometer, feedback_altitude, goals, margins, processed_margins, thrust_rows = [], [], [], [], [], []
    events, contexts = [], []
    active = {}
    outcome = 'Unknown'
    tracking_time = None
    sequence_times = {v.get('sequence'): t for topic, t, _, v in records
                      if topic == MISSION and t is not None and type(v.get('sequence')) is int}
    for topic, t, arrival, v in sorted(records, key=lambda row: row[2]):
        if topic == SENSORS:
            baro = v['sensors'].get('barometer')
            if baro is not None and baro['episode_id'] == key[1]:
                barometer.append((baro['simulation_time_s'], (baro['altitude_m'],)))
        elif topic == FEEDBACK and t is not None and v['barometer_altitude_m'] is not None:
            feedback_altitude.append((t, (v['barometer_altitude_m'],)))
        elif topic == BBOX and t is not None and v['bbox'] is not None:
            box = v['bbox']
            identity = frame_identity(v.get('source'), v['sequence'])
            ref = frames.get(identity)
            size = (ref.width, ref.height) if ref is not None and ref.width > 0 and ref.height > 0 else camera
            if size is not None:
                x, y, w, h = (box[k] for k in ('x', 'y', 'width', 'height'))
                margins.append((t, (x, y, size[0] - x - w, size[1] - y - h)))
        terminal = v if topic == END else v.get('terminal_result') if topic == STATE else None
        if terminal is not None:
            outcome = 'Success · ' + terminal['reason']
            if active.get('scenario', (None,))[0] != terminal['event_id']:
                events.append(Event(t, t, arrival, 'info', 'scenario', outcome, 'COMPLETED', freeze(terminal)))
                active['scenario'] = (terminal['event_id'], len(events)-1)
        if topic == STATE and v['state'] == 'error' and not outcome.startswith('Success'):
            outcome = 'Physics fault · ' + v['reason']
        if topic == STATE and v['state'] == 'stopped' and outcome == 'Unknown':
            outcome = 'Stopped · ' + v['reason']
        if topic == MISSION:
            phase, reason = v['phase'], v['reason']
            if tracking_time is None and (phase == 'LOS_APPROACH'
                                          or (phase.startswith('LOS_') and str(v.get('navigator_state', '')).upper() == 'TRACK')):
                tracking_time = t
            d = v['diagnostics']
            image, bbox = mission_image(v)
            ref = frames.get(image)
            bbox = bbox if bbox is not None else boxes.get(image)
            details = dict(d, phase=phase, reason=reason, navigator_state=v.get('navigator_state'))
            context = {k: d.get(k) for k in ('vision_reason', 'image_age_s', 'barometer_age_s', 'attitude_age_s',
                                            'thrust_saturated', 'angle_limited', 'slew_limited')}
            for name in ('image_age_s', 'barometer_age_s', 'attitude_age_s'):
                age = context[name]
                if type(age) not in (int, float) or not isfinite(age):
                    context[name] = None
            for name in ('thrust_saturated', 'angle_limited', 'slew_limited'):
                if type(context[name]) is not bool:
                    context[name] = None
            context['phase'] = phase
            if t is not None:
                contexts.append((t, freeze(context)))
                if phase in TAKEOFF_PHASES and d.get('takeoff_goal_altitude_m') is not None:
                    goals.append((t, (d['takeoff_goal_altitude_m'],)))
                if d.get('edge_margins_px') is not None:
                    processed_margins.append((t, d['edge_margins_px']))
            if phase == 'COMPLETED':
                outcome = 'Success · ' + reason
            elif phase == 'FAULT' and not outcome.startswith('Success'):
                outcome = 'Mission fault · ' + reason
            candidates = [('phase', phase, 'error' if phase == 'FAULT' else 'info', phase + ' · ' + reason)]
            vision = d.get('vision_reason', '')
            vision = vision if isinstance(vision, str) else ''
            warning = vision in VISION_WARNINGS or (vision == 'no_target' and phase.startswith('LOS_'))
            candidates.append(('vision', vision if warning else '', 'warning', vision))
            recovering = str(v.get('navigator_state', '')).upper() == 'RECOVER'
            candidates.append(('navigation', reason if recovering else '', 'warning', 'Recovery · ' + reason))
            for category, signature, severity, label in candidates:
                if not signature:
                    active.pop(category, None)
                    continue
                signature = (signature, reason) if category == 'phase' and phase == 'FAULT' else signature
                previous = active.get(category)
                if previous is not None and previous[0] == signature:
                    index = previous[1]
                    events[index] = replace(events[index], end_s=t)
                else:
                    events.append(Event(t, t, arrival, severity, category, label, phase, freeze(details), ref,
                                        None if bbox is None else tuple(bbox)))
                    active[category] = (signature, len(events) - 1)
        elif topic == STATE:
            severity = 'error' if v['state'] == 'error' else 'info'
            signature = (v['state'], v['reason'])
            if active.get('physics', (None,))[0] != signature:
                events.append(Event(t, t, arrival, severity, 'physics', v['state'] + ' · ' + v['reason'], '', freeze(v)))
                active['physics'] = (signature, len(events) - 1)
        elif topic == RENDER_HEALTH and v['stats'].get('error'):
            reason = v['stats']['error']
            if active.get('renderer', (None,))[0] != reason:
                events.append(Event(None, None, arrival, 'error', 'renderer', reason, '', freeze(v)))
                active['renderer'] = (reason, len(events) - 1)
        elif topic == RC:
            sample_time = sequence_times.get(v.get('sequence'))
            value = thrust_value(v.get('channels', [None] * 3)[2], backend)
            if sample_time is not None and value is not None:
                thrust_rows.append((sample_time, (value,)))
    timestamps = [t for _, t, _, _ in records if t is not None]
    contexts = sorted(dict(contexts).items())
    events = tuple(sorted(events, key=lambda e: e.arrival_ns))
    failures = [e for e in events if e.severity == 'error']
    barometer_samples = series(barometer or feedback_altitude)
    goal_samples = series(goals)
    margin_samples = series(margins or processed_margins, 4)
    attitude_samples = series(attitude_rows_actual, 3)
    feedback_attitude_samples = series(estimated_rows, 3)
    thrust_samples = series(thrust_rows)
    notices = []
    drops = [v.get('publisher_drops', 0) for topic, _, _, v in records if topic in (TRUTH, SENSORS)]
    if drops and max(drops):
        notices.append(f'Analysis publisher reports {max(drops):,} cumulative queue drops ({max(drops)-min(drops):,} increase observed).')
    if not truth:
        notices.append('No actual ground truth: rendered pose is a fallback, not physics truth.')
    if not len(trajectory.times):
        notices.append('X–Z trajectory is unavailable.')
    if not len(goal_samples.times):
        notices.append('Takeoff goal unavailable: this recording does not expose an explicit goal.')
    if not frames:
        notices.append('Exact image identity metadata unavailable; event frames cannot be guessed.')
    if not len(margin_samples.times):
        notices.append('BBox margins unavailable: missing boxes, dimensions or source timestamps.')
    if not len(attitude_samples.times):
        notices.append('Physical attitude unavailable: no ground-truth or rendered-pose orientation samples.')
    if not len(feedback_attitude_samples.times):
        notices.append('Controller attitude unavailable: no flight feedback orientation samples.')
    if not len(thrust_samples.times):
        notices.append('Commanded thrust unavailable: no timestamp-aligned RC samples.')
    integrity = 'Unknown'
    if health:
        integrity = 'Complete (recorder)' if health.get('complete') and not partial else 'Incomplete'
        skipped = health.get('skipped_images', 0)
        if skipped:
            notices.append(f'{skipped:,} SHM images were skipped by the recorder.')
        if outcome == 'Unknown' and (health.get('episode') is None or health['episode'] == dict(physics_id=key[0], episode_id=key[1])) and str(health.get('outcome', '')).startswith('mission_fault:'):
            outcome = 'Mission fault · ' + health['outcome'].split(':', 1)[1]
    elif partial:
        integrity = 'Incomplete'
    if not health:
        notices.append('No recorder health record; recording completeness is unknown.')
    finite_z = trajectory.values[:, 2] if len(trajectory.times) else []
    summary = Summary(outcome, max(timestamps) - min(timestamps) if timestamps else 0.,
                      float(np.max(finite_z)) if len(finite_z) else None,
                      (float(np.min(trajectory.values[:, 1])), float(np.max(trajectory.values[:, 1]))) if len(trajectory.times) else None,
                      failures[0].reason if failures else None,
                      sum(e.severity == 'warning' for e in events), len(failures), integrity)
    return Episode(key, trajectory, origin, attitude_samples, feedback_attitude_samples, thrust_samples, tracking_time,
                   barometer_samples, goal_samples,
                   margin_samples, 'Detected bbox' if margins else 'Processed bbox',
                   events, summary, tuple(notices), tuple(t for t, _ in contexts), tuple(d for _, d in contexts),
                   tuple(sorted(frames.values(), key=lambda ref: (ref.log_time_ns, ref.record_sequence))),
                   freeze({identity: boxes[identity] for identity in frames if identity in boxes}))


def load_flight(path, progress=lambda message: None, cancel=lambda: False):
    path = Path(path).resolve()
    stat = path.stat()
    validators = {topic: Draft202012Validator(json.loads(schema_bytes(name))) for topic, name in SCHEMAS.items()}
    records, frames, boxes, cameras, stream_physics, backends = [], {}, {}, {}, {}, {}
    health, warnings = None, defaultdict(int)
    with path.open('rb') as file:
        reader = make_reader(file, validate_crcs=True)
        summary = reader.get_summary()
        if summary is None or not summary.chunk_indexes:
            raise ValueError('Recording has no usable index. Recover/index a copy with MCAP tooling, then reopen it.')
        camera_image_count = None if summary.statistics is None else sum(
            summary.statistics.channel_message_counts.get(cid, 0)
            for cid, channel in summary.channels.items() if channel.topic == 'camera/front/image')
        for metadata in reader.iter_metadata():
            check_cancel(cancel)
            if metadata.name == 'recording_health':
                try:
                    candidate = {key: json.loads(value) for key, value in metadata.metadata.items()}
                    Draft202012Validator(json.loads(schema_bytes('recording_health.json'))).validate(candidate)
                    health = candidate
                except (ValueError, ValidationError, TypeError):
                    warnings['recording_health metadata'] += 1
        for count, (_, channel, message) in enumerate(reader.iter_messages(topics=list(SCHEMAS)), 1):
            check_cancel(cancel)
            if count % 500 == 0:
                progress(f'Decoded {count:,} scalar messages…')
            try:
                if len(message.data) > MAX_JSON_BYTES:
                    raise ValueError('JSON message exceeds 1 MiB')
                value = decode(message.data)
                validators[channel.topic].validate(value)
                records.append((channel.topic, message.log_time, value))
                if channel.topic == PLATFORM:
                    camera = value['camera']
                    cameras[value['physics_id']] = (camera['width_px'], camera['height_px'])
                    stream_physics[value['stream_id']] = value['physics_id']
                    backends[value['physics_id']] = value.get('backend', 'internal')
                elif channel.topic == METADATA:
                    metadata = value['metadata']
                    identity = frame_identity(metadata.get('source'), metadata['sequence'])
                    if identity is not None:
                        width, height = metadata.get('width', 0), metadata.get('height', 0)
                        width = width if type(width) is int and width > 0 else 0
                        height = height if type(height) is int and height > 0 else 0
                        if width * height * 3 > MAX_IMAGE_BYTES:
                            width = height = 0
                        frames[identity] = FrameRef(value['image_topic'], value['image_log_time_ns'], value['image_record_sequence'],
                                                   identity, width, height, metadata['source']['simulation_time_s'])
                elif channel.topic == BBOX and value['image_topic'] == 'camera/front/image' and value['bbox'] is not None:
                    identity = frame_identity(value.get('source'), value['sequence'])
                    if identity is not None:
                        boxes[identity] = tuple(value['bbox'][key] for key in ('x', 'y', 'width', 'height'))
            except (ValueError, TypeError, KeyError, ValidationError, OverflowError):
                warnings[channel.topic] += 1
    sequence_times = {(v.get('physics_id'), v.get('episode_id'), v.get('sequence')): v.get('simulation_time_s')
                      for topic, _, v in records if topic == MISSION}
    grouped = defaultdict(list)
    for topic, arrival, value in records:
        check_cancel(cancel)
        if topic in (PLATFORM, METADATA):
            continue
        source = value.get('source', value.get('final_source', value))
        episode_id = source.get('episode_id')
        physics_id = source.get('physics_id')
        if topic == POSE and not physics_id:
            physics_id = stream_physics.get(value['stream_id'], 'legacy-' + value['stream_id'])
        if not physics_id or not episode_id:
            continue
        t = source.get('simulation_time_s')
        if topic == RC:
            t = sequence_times.get((physics_id, episode_id, value.get('sequence')))
        if t is not None and (not isinstance(t, (float, int)) or not isfinite(t)):
            warnings[topic] += 1
            t = None
        grouped[(physics_id, episode_id)].append((topic, t, arrival, value))
    if not grouped:
        raise ValueError('No supported flight samples found. Expected Bullet mission, physics or rendered-pose topics.')
    keys = sorted(grouped, key=lambda key: min(r[2] for r in grouped[key]))
    episodes = []
    for key in keys:
        check_cancel(cancel)
        refs = {identity: ref for identity, ref in frames.items() if identity[:2] == key}
        episodes.append(build_episode(key, grouped[key], refs, boxes, cameras.get(key[0]), health,
                                      path.name.endswith('.partial'), backends.get(key[0], 'internal')))
    authoritative = None if health is None else health.get('episode')
    default = len(keys) - 1
    if authoritative and (authoritative['physics_id'], authoritative['episode_id']) in keys:
        default = keys.index((authoritative['physics_id'], authoritative['episode_id']))
    notices = tuple(f'{count:,} unsupported or invalid messages skipped on {topic}.' for topic, count in warnings.items())
    progress(f'Loaded {len(records):,} scalar messages; images remain on disk.')
    return FlightData(path, stat.st_size, stat.st_mtime_ns, tuple(episodes), default, notices, camera_image_count)


def check_file(data):
    stat = data.path.stat()
    if (stat.st_size, stat.st_mtime_ns) != (data.size, data.modified_ns):
        raise ValueError('Recording changed since loading. Reopen the recording.')


def decode_frame(channel, message):
    """Validate and unpack one recorded RGB/BGR image, including row padding."""
    if channel.message_encoding != 'protobuf' or len(message.data) > MAX_IMAGE_BYTES:
        raise ValueError('Unsupported or oversized recorded image.')
    image = RawImage.FromString(message.data)
    if (image.width <= 0 or image.height <= 0 or image.step < image.width * 3
            or image.step * image.height > MAX_IMAGE_BYTES or len(image.data) != image.step * image.height
            or image.encoding not in ('rgb8', 'bgr8')):
        raise ValueError('Unsupported recorded image layout.')
    rows = np.frombuffer(image.data, dtype=np.uint8).reshape(image.height, image.step)
    pixels = rows[:, :image.width * 3].reshape(image.height, image.width, 3)
    if image.encoding == 'bgr8':
        pixels = pixels[:, :, ::-1]
    return Frame(image.width, image.height, pixels.tobytes())


def read_frame(data, reference, cancel=lambda: False):
    check_cancel(cancel)
    check_file(data)
    with data.path.open('rb') as file:
        reader = make_reader(file, validate_crcs=True)
        for _, channel, message in reader.iter_messages(topics=[reference.topic], starting_at=reference.log_time_ns, ending_at=reference.log_time_ns):
            check_cancel(cancel)
            if message.sequence != reference.record_sequence:
                continue
            frame = decode_frame(channel, message)
            check_cancel(cancel)
            return frame
    raise ValueError('Exact image record is missing from this recording.')
