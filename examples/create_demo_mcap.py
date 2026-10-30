"""Create the small deterministic MCAP used for FlightLens development."""

import argparse
import json
from pathlib import Path

from bullet_msgs import schema_bytes
from foxglove_schemas_protobuf.RawImage_pb2 import RawImage
from mcap.writer import CompressionType, Writer

from flight_lens.reader import BBOX, END, FEEDBACK, LOS, METADATA, MISSION, MSP, POSE, RC, TRUTH

BASE_NS = 1_700_000_000_000_000_000
PHYSICS_ID = 'demo_physics'
EPISODE_ID = 'demo_episode'
STREAM_ID = 'demo_stream'
SESSION = '0123456789abcdef0123456789abcdef'
IMAGE_TOPIC = 'camera/front/image'
WIDTH_PX = 8
HEIGHT_PX = 6


def source(time_s, step):
    return {
        'physics_id': PHYSICS_ID,
        'episode_id': EPISODE_ID,
        'stream_id': STREAM_ID,
        'step_index': step,
        'simulation_time_s': time_s,
        'measurement_us': 10_000_000 + round(time_s * 1_000_000),
    }


def mission(time_s, step, phase, reason, diagnostics=None):
    return {
        'schema_version': 1,
        'physics_id': PHYSICS_ID,
        'episode_id': EPISODE_ID,
        'app_id': 'demo_app',
        'sequence': step,
        'measurement_us': source(time_s, step)['measurement_us'],
        'simulation_time_s': time_s,
        'phase': phase,
        'reason': reason,
        'relative_altitude_m': 2.0 if phase.startswith('LOS_') else 1.0,
        'vertical_speed_m_s': 0.0,
        'hover_remaining_s': 0.0,
        'target_remaining_s': max(0.0, 0.6 - time_s),
        'navigator_state': 'TRACK' if phase.startswith('LOS_') else 'HOVER',
        'diagnostics': diagnostics or {},
    }


def ground_truth(time_s, step, position, velocity):
    return {
        'schema_version': 1,
        'source': source(time_s, step),
        'publisher_drops': 0,
        'ground_truth': {
            'drone_id': 'demo_drone',
            'episode_id': EPISODE_ID,
            'step_index': step,
            'simulation_time_s': time_s,
            'position_m': list(position),
            'orientation_xyzw': [0.0, 0.0, 0.0, 1.0],
            'linear_velocity_m_s': list(velocity),
            'angular_velocity_rad_s': [0.0, 0.0, 0.0],
            'rotor_speeds_rad_s': [420.0, 420.0, 420.0, 420.0],
        },
    }


def feedback(time_s, step, altitude):
    return {
        'schema_version': 1,
        'source': source(time_s, step),
        'barometer_altitude_m': altitude,
        'orientation_xyzw': [0.0, 0.0, 0.0, 1.0],
        'armed': True,
        'mode': 'angle',
        'last_applied_sequence': step,
        'owner_id': 'demo_app',
        'state': 'running',
        'reason': 'tracking',
    }


def msp(time_s, step, yaw):
    return {
        'schema_version': 1,
        'source': source(time_s, step),
        'backend': 'betaflight',
        'timestamp_kind': 'host_receipt',
        'firmware': None,
        'error': None,
        'msp_delivery': {
            'requests_hz': 50.0,
            'responses_hz': 50.0,
            'queue_delay_mean_s': 0.001,
            'queue_delay_max_s': 0.002,
            'response_turnaround_mean_s': 0.003,
            'response_turnaround_max_s': 0.005,
        },
        'sensor_feedback': {
            'window_s': 1.0,
            'expected_hz': 120.0,
            'observed_hz': 120.0,
            'feedback_age_s': 0.01,
            'attitude_age_s': 0.01,
            'barometer_age_s': 0.01,
            'image_age_s': 0.02,
            'image_attitude_offset_s': 0.0,
            'vision_reason': 'usable',
            'mission_phase': 'LOS_APPROACH',
        },
        'rc_frames_sent': step,
        'last_rc_sent_at_s': time_s,
        'attitude': {'roll_rad': 0.0, 'pitch_rad': 0.0, 'yaw_rad': yaw, 'received_at': time_s},
        'altitude': None,
        'status': None,
    }


def los(time_s, step, ttc):
    return {
        'schema_version': 1,
        'app_id': 'demo_app',
        'sequence': step,
        'source': source(time_s, step),
        'generated_us': source(time_s, step)['measurement_us'],
        'navigation_sequence': step,
        'navigator_state': 'track',
        'stage': 'ttc',
        'reason': 'tracking',
        'mission_phase': 'LOS_APPROACH',
        'active': True,
        'elevation_deg': 0.0,
        'azimuth_deg': 0.0,
        'bearing_valid': True,
        'elevation_rate_deg_s': 0.0,
        'azimuth_rate_deg_s': 0.0,
        'rate_observable': True,
        'optical_ttc_s': ttc,
        'ttc_valid': True,
        'ttc_reason': 'qualified',
        'ttc_relative_residual': 0.02,
    }


def image_metadata(time_s, step):
    return {
        'schema_version': 1,
        'image_topic': IMAGE_TOPIC,
        'image_log_time_ns': BASE_NS + round(time_s * 1_000_000_000),
        'image_record_sequence': 0,
        'metadata': {
            'session': SESSION,
            'sequence': 1,
            'capture_ns': source(time_s, step)['measurement_us'] * 1000,
            'source': source(time_s, step),
            'width': WIDTH_PX,
            'height': HEIGHT_PX,
        },
    }


def bbox(time_s, step):
    return {
        'schema_version': 2,
        'image_topic': IMAGE_TOPIC,
        'session': SESSION,
        'sequence': 1,
        'capture_ns': source(time_s, step)['measurement_us'] * 1000,
        'source': source(time_s, step),
        'bbox': {'x': 2, 'y': 1, 'width': 4, 'height': 3},
    }


def terminal(time_s, step):
    final_source = source(time_s, step)
    collision = {
        'schema_version': 1,
        'event_id': 'demo_hit',
        'controller_id': 'demo_controller',
        'issued_us': final_source['measurement_us'],
        'source': final_source,
        'target_id': 'red_truck',
        'drone_position_m': [2.0, 0.0, 2.0],
    }
    return {
        'schema_version': 1,
        'event_id': 'demo_hit',
        'collision': collision,
        'final_source': final_source,
        'outcome': 'success',
        'reason': 'target_hit',
        'ended_us': final_source['measurement_us'],
    }


def build_records():
    diagnostics = {
        'vision_reason': 'usable',
        'processed_image_stream_id': STREAM_ID,
        'processed_image_sequence': 1,
        'processed_image_timestamp_s': 10.3,
        'processed_bbox': [2, 1, 4, 3],
        'edge_margins_px': [2, 1, 2, 2],
        'image_age_s': 0.02,
        'target_error_x_px': 0.0,
        'target_error_y_px': -0.5,
        'optical_ttc_raw_s': 1.5,
        'optical_ttc_s': 1.5,
        'ttc_valid': True,
        'ttc_reason': 'qualified',
        'requested_roll_rad': 0.0,
        'reference_pitch_rad': 0.0,
        'thrust_n': 12.0,
    }
    records = [
        (TRUTH, 0.00, ground_truth(0.0, 0, (0.0, 0.0, 1.0), (0.0, 0.0, 0.0))),
        (POSE, 0.01, {
            'schema_version': 3, 'source': source(0.0, 0), 'drone_id': 'demo_drone',
            'episode_id': EPISODE_ID, 'stream_id': STREAM_ID, 'step_index': 0,
            'simulation_time_s': 0.0, 'sequence': 0, 'publisher_drops': 0,
            'position_m': [0.0, 0.0, 1.0], 'orientation_xyzw': [0.0, 0.0, 0.0, 1.0],
            'linear_velocity_m_s': [0.0, 0.0, 0.0],
        }),
        (MISSION, 0.02, mission(0.0, 0, 'TAKEOFF', 'climbing', {'takeoff_goal_altitude_m': 2.0, 'barometer_datum_m': 1.0})),
        (METADATA, 0.30, image_metadata(0.3, 3)),
        (BBOX, 0.31, bbox(0.3, 3)),
        (TRUTH, 0.32, ground_truth(0.3, 3, (1.0, 0.0, 2.0), (3.0, 0.0, 0.0))),
        (FEEDBACK, 0.33, feedback(0.3, 3, 2.0)),
        (MSP, 0.34, msp(0.3, 3, 0.05)),
        (RC, 0.35, {
            'schema_version': 1, 'physics_id': PHYSICS_ID, 'episode_id': EPISODE_ID,
            'app_id': 'demo_app', 'sequence': 3, 'issued_us': 10_300_000,
            'valid_until_us': 10_310_000, 'active': True,
            'channels': [1500, 1500, 1500, 1500, 2000, 2000] + [1000] * 12,
        }),
        (MISSION, 0.36, mission(0.3, 3, 'LOS_APPROACH', 'tracking', diagnostics)),
        (LOS, 0.37, los(0.3, 3, 1.5)),
        (MISSION, 0.46, mission(0.4, 4, 'LOS_DESCENT', 'tracking', {**diagnostics, 'processed_image_timestamp_s': 10.3, 'optical_ttc_raw_s': 1.0, 'optical_ttc_s': 1.0})),
        (LOS, 0.47, los(0.4, 4, 1.0)),
        (MISSION, 0.58, mission(0.6, 6, 'COMPLETED', 'target_hit')),
        (END, 0.60, terminal(0.6, 6)),
    ]
    return records


def write_demo(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('wb') as output:
        writer = Writer(output, compression=CompressionType.NONE)
        writer.start()
        channels = {}
        for topic, arrival_s, packet in build_records():
            if topic not in channels:
                schema_id = writer.register_schema(topic, 'jsonschema', schema_bytes({
                    TRUTH: 'analysis_ground_truth.json', POSE: 'pose.json', MISSION: 'mission.json',
                    BBOX: 'bbox.json', FEEDBACK: 'flight_feedback.json', MSP: 'analysis_msp.json',
                    LOS: 'los_telemetry.json', RC: 'flight_rc.json', METADATA: 'image_metadata.json',
                    END: 'scenario_end.json',
                }[topic]))
                channels[topic] = writer.register_channel(topic, 'json', schema_id)
            stamp = BASE_NS + round(arrival_s * 1_000_000_000)
            writer.add_message(channels[topic], stamp, json.dumps(packet, sort_keys=True, separators=(',', ':')).encode(), stamp)
        image_schema = writer.register_schema('foxglove.RawImage', 'protobuf', b'')
        image_channel = writer.register_channel(IMAGE_TOPIC, 'protobuf', image_schema)
        image = RawImage(width=WIDTH_PX, height=HEIGHT_PX, step=WIDTH_PX * 3, encoding='rgb8', data=bytes([220, 40, 40]) * WIDTH_PX * HEIGHT_PX)
        stamp = BASE_NS + 300_000_000
        writer.add_message(image_channel, stamp, image.SerializeToString(), stamp, 0)
        writer.add_metadata('recording_health', {
            'schema_version': json.dumps(1), 'complete': json.dumps(True), 'reason': json.dumps('closed'),
            'outcome': json.dumps('success'), 'skipped_images': json.dumps(0), 'peak_queue_bytes': json.dumps(0),
            'counts': json.dumps({}), 'launch_id': json.dumps(None),
            'episode': json.dumps({'physics_id': PHYSICS_ID, 'episode_id': EPISODE_ID}),
        })
        writer.finish()
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', nargs='?', type=Path, default=Path(__file__).with_name('demo.mcap'))
    args = parser.parse_args()
    print(write_demo(args.output))


if __name__ == '__main__':
    main()
