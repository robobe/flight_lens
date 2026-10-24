"""Small indexed Bullet MCAP fixtures; GUI tests use Qt's offscreen platform."""
import json
import os
from pathlib import Path

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import pytest
from bullet_msgs import schema_bytes
from foxglove_schemas_protobuf.RawImage_pb2 import RawImage
from mcap.writer import Writer, CompressionType

from flight_lens.reader import SCHEMAS

BASE_NS = 1_700_000_000_000_000_000


def source(t=0., sequence=0, episode='episode', physics='physics'):
    return dict(physics_id=physics, episode_id=episode, stream_id='stream', step_index=sequence,
                simulation_time_s=t, measurement_us=10_000_000 + round(t * 1e6))


def mission(t, phase='LOS_APPROACH', reason='tracking', diagnostics=None, episode='episode', sequence=None):
    return dict(schema_version=1, physics_id='physics', episode_id=episode, app_id='app',
                sequence=round(t * 100) if sequence is None else sequence, measurement_us=source(t)['measurement_us'],
                simulation_time_s=t, phase=phase, reason=reason, relative_altitude_m=2., vertical_speed_m_s=0.,
                hover_remaining_s=0., target_remaining_s=0., navigator_state='TRACK', diagnostics=diagnostics or {})


def truth(t, position=(0., 0., 2.), episode='episode'):
    s = source(t, round(t * 100), episode)
    return dict(schema_version=1, source=s, publisher_drops=0, ground_truth=dict(
        drone_id='drone_1', episode_id=episode, step_index=s['step_index'], simulation_time_s=t,
        position_m=position, orientation_xyzw=[0., 0., 0., 1.], linear_velocity_m_s=[0.]*3,
        angular_velocity_rad_s=[0.]*3, rotor_speeds_rad_s=[0.]*4))


def pose(t, position=(50., 0., 30.), episode='episode'):
    s = source(t, round(t * 100), episode)
    return dict(schema_version=3, source=s, drone_id='drone_1', episode_id=episode, stream_id='stream',
                step_index=s['step_index'], simulation_time_s=t, sequence=s['step_index'], publisher_drops=0,
                position_m=position, orientation_xyzw=[0., 0., 0., 1.], linear_velocity_m_s=[0.]*3)


def health(episode='episode', complete=True, outcome='stop'):
    return dict(schema_version=1, complete=complete, reason='closed', outcome=outcome,
                skipped_images=0, peak_queue_bytes=0, counts={}, launch_id=None,
                episode=dict(physics_id='physics', episode_id=episode))


def frame_metadata(t, sequence, arrival_s, record_sequence=0, session='a'*32, episode='episode'):
    s = source(t, round(t*100), episode)
    return dict(schema_version=1, image_topic='camera/front/image', image_log_time_ns=BASE_NS+round(arrival_s*1e9),
                image_record_sequence=record_sequence, metadata=dict(session=session,sequence=sequence,
                capture_ns=s['measurement_us']*1000, source=s, width=8, height=6))


def write_mcap(path, records, health_data=None, frames=()):
    with Path(path).open('wb') as file:
        writer=Writer(file, compression=CompressionType.NONE)
        writer.start()
        channels={}
        for topic, arrival, packet in records:
            if topic not in channels:
                sid=writer.register_schema(topic, 'jsonschema', schema_bytes(SCHEMAS[topic]))
                channels[topic]=writer.register_channel(topic,'json',sid)
            stamp=BASE_NS+round(arrival*1e9)
            writer.add_message(channels[topic],stamp,json.dumps(packet).encode(),stamp)
        if frames:
            sid=writer.register_schema('foxglove.RawImage','protobuf',b'')
            cid=writer.register_channel('camera/front/image','protobuf',sid)
            for arrival, sequence, pixels, encoding in frames:
                image=RawImage(width=8,height=6,step=24,encoding=encoding,data=pixels)
                stamp=BASE_NS+round(arrival*1e9)
                writer.add_message(cid,stamp,image.SerializeToString(),stamp,sequence)
        if health_data is not None:
            writer.add_metadata('recording_health',{k:json.dumps(v) for k,v in health_data.items()})
        writer.finish()
    return Path(path)


@pytest.fixture
def flight_file(tmp_path):
    from flight_lens.reader import TRUTH,POSE,MISSION,METADATA,BBOX,SENSORS
    s=source(.1,10)
    diagnostics=dict(vision_reason='clipped_target',processed_image_stream_id='stream',
                     processed_image_sequence=42,processed_image_timestamp_s=s['measurement_us']/1e6,
                     processed_bbox=[0,1,2,2],edge_margins_px=[0,1,6,3],image_age_s=.03,thrust_saturated=True)
    baro=dict(episode_id='episode',step_index=10,simulation_time_s=.1,pressure_pa=101000.,altitude_m=2.5)
    sensors=dict(schema_version=1,source=s,publisher_drops=0,sensors=dict(imu=None,barometer=baro,rangefinder=None),attitude_estimate=None)
    records=[(TRUTH,0.,truth(0.,(0.,-1.,1.))), (POSE,.01,pose(0.)),
             (MISSION,.02,mission(0.,'TAKEOFF','climbing',dict(takeoff_goal_altitude_m=3.,barometer_datum_m=1.))),
             (METADATA,.1,frame_metadata(.1,42,.1)),
             (BBOX,.11,dict(schema_version=2,image_topic='camera/front/image',session='a'*32,sequence=42,capture_ns=s['measurement_us']*1000,source=s,bbox=dict(x=0,y=1,width=2,height=2))),
             (TRUTH,.12,truth(.1,(1.,2.,2.))), (SENSORS,.13,sensors),
             (MISSION,.14,mission(.1,diagnostics=diagnostics)),
             (MISSION,.2,mission(.2,diagnostics=diagnostics)),
             (MISSION,.3,mission(.3,diagnostics=dict(vision_reason='usable'))),
             (MISSION,.4,mission(.4,'FAULT','los_clipped_target',diagnostics)),
             (MISSION,.5,mission(.5,'FAULT','los_clipped_target',diagnostics))]
    return write_mcap(tmp_path/'flight.mcap',records,health(),frames=[(.1,0,bytes([255,0,0])*48,'rgb8')])
