from dataclasses import FrozenInstanceError
import json
import numpy as np
import pytest
from mcap.writer import Writer

from flight_lens.model import series
from flight_lens.reader import (load_flight, read_frame, Cancelled, TRUTH, MISSION, METADATA, POSE, BBOX, decode)
from conftest import source, mission, truth, pose, health, frame_metadata, write_mcap


def test_truth_summary_events_and_exact_frame(flight_file):
    data=load_flight(flight_file)
    episode=data.episodes[data.default_episode]
    assert episode.trajectory_origin=='Actual ground truth'
    assert episode.trajectory.values.tolist()==[[0.,-1.,1.],[1.,2.,2.]]
    assert episode.summary.outcome=='Mission fault · los_clipped_target'
    assert episode.summary.max_altitude_m==2.
    assert episode.summary.y_range_m==(-1.,2.)
    assert episode.summary.integrity=='Complete (recorder)'
    assert episode.barometer.values[:,0].tolist()==[2.5]
    assert episode.goal.values[:,0].tolist()==[3.]
    assert episode.margins.values.tolist()==[[0.,1.,6.,3.]]
    warnings=[e for e in episode.events if e.severity=='warning']
    assert len(warnings)==2  # Separate rejection intervals, not 100 Hz packet counts.
    assert warnings[0].time_s==.1 and warnings[0].end_s==.2
    errors=[e for e in episode.events if e.severity=='error']
    assert len(errors)==1 and errors[0].end_s==.5
    assert errors[0].frame is not None
    frame=read_frame(data,errors[0].frame)
    assert frame.rgb==bytes([255,0,0])*48
    with pytest.raises(ValueError):
        episode.trajectory.values[0,0]=99
    with pytest.raises(TypeError):
        errors[0].details['phase']='wrong'
    assert episode.context_at(.1)['image_age_s']==.03
    assert episode.context_at(10)=={}


def test_clipping_is_not_a_flight_fault(tmp_path):
    path=write_mcap(tmp_path/'clip.mcap',[(MISSION,.1,mission(.1,diagnostics={'vision_reason':'clipped_target'}))])
    episode=load_flight(path).episodes[0]
    assert episode.summary.outcome=='Unknown'
    assert episode.summary.errors==0 and episode.summary.warnings==1
    assert episode.summary.integrity=='Unknown'


def test_legacy_fallback_missing_goal_and_frames(tmp_path):
    path=write_mcap(tmp_path/'legacy.mcap',[(POSE,0,pose(0)),(MISSION,.1,mission(.1))])
    episode=load_flight(path).episodes[0]
    assert episode.trajectory_origin.startswith('Rendered pose')
    assert episode.summary.max_altitude_m==30.
    assert len(episode.goal.times)==0
    assert any('goal unavailable' in n for n in episode.notices)
    assert all(e.frame is None for e in episode.events)


def test_episode_identity_and_frame_number_reuse(tmp_path):
    first=frame_metadata(.1,42,.1)
    second=frame_metadata(.2,42,.2,1,session='b'*32)
    d=dict(vision_reason='clipped_target',processed_image_stream_id='stream',processed_image_sequence=42,
           processed_image_timestamp_s=source(.1)['measurement_us']/1e6,processed_bbox=[0,1,2,2])
    records=[(METADATA,.1,first),(METADATA,.2,second),(MISSION,.3,mission(.3,diagnostics=d)),
             (TRUTH,.4,truth(.4)),(MISSION,.5,mission(.1,'FAULT','old_fault',episode='old'))]
    path=write_mcap(tmp_path/'reuse.mcap',records,health(),frames=[(.1,0,bytes([255,0,0])*48,'rgb8'),(.2,1,bytes([0,0,255])*48,'rgb8')])
    data=load_flight(path)
    assert len(data.episodes)==2
    assert data.episodes[data.default_episode].key==('physics','episode')
    warning=next(e for e in data.episodes[data.default_episode].events if e.component=='vision')
    assert read_frame(data,warning.frame).rgb[:3]==bytes([255,0,0])


@pytest.mark.parametrize('box,expected', [((0,1,2,2),0),((1,0,2,2),1),((6,1,2,2),2),((1,4,2,2),3)])
def test_four_bbox_edges(tmp_path,box,expected):
    s=source(.1,10)
    packet=dict(schema_version=2,image_topic='camera/front/image',session='a'*32,sequence=42,capture_ns=s['measurement_us']*1000,source=s,bbox=dict(zip(('x','y','width','height'),box)))
    path=write_mcap(tmp_path/'edge.mcap',[(METADATA,.1,frame_metadata(.1,42,.1)),(BBOX,.2,packet)])
    episode=load_flight(path).episodes[0]
    assert episode.margins.values[0,expected]==0


def test_gap_split_nearest_and_missing_samples():
    values=series([(0,(0,)),(.01,(1,)),(.02,(2,)),(1,(3,)),(1.01,(4,))])
    x,y=values.plot(0)
    assert np.isnan(x).sum()==np.isnan(y).sum()==1
    assert values.nearest(.5) is None
    assert values.nearest(.011)==1
    assert len(series([(0,(float('inf'),))]).times)==0


def test_cancel_errors_changed_file_and_partial(flight_file,tmp_path):
    with pytest.raises(Cancelled):
        load_flight(flight_file,cancel=lambda:True)
    data=load_flight(flight_file)
    ref=next(e.frame for e in data.episodes[0].events if e.frame is not None)
    with flight_file.open('ab') as file:
        file.write(b'changed')
    with pytest.raises(ValueError,match='changed'):
        read_frame(data,ref)
    empty=tmp_path/'empty.mcap'
    with empty.open('wb') as file:
        writer=Writer(file);writer.start();writer.finish()
    with pytest.raises(ValueError,match='index'):
        load_flight(empty)
    partial=write_mcap(tmp_path/'bad.mcap.partial',[(TRUTH,0,truth(0))],health(complete=False))
    assert load_flight(partial).episodes[0].summary.integrity=='Incomplete'
    with pytest.raises(ValueError):
        decode(b'{"x":1e999}')


def test_invalid_optional_diagnostics_do_not_break_loading(tmp_path):
    diagnostics=dict(vision_reason={'unexpected':True},image_age_s='not a number',thrust_saturated='yes',
                     processed_image_sequence=True,processed_image_timestamp_s=10.,processed_bbox='bad',
                     latest_image='bad',edge_margins_px='bad',takeoff_goal_altitude_m='bad')
    path=write_mcap(tmp_path/'optional.mcap',[(MISSION,0,mission(0.,diagnostics=diagnostics)),
                                           (TRUTH,.1,truth(.1))])
    episode=load_flight(path).episodes[0]
    assert episode.context_at(0.)['image_age_s'] is None
    assert episode.context_at(0.)['thrust_saturated'] is None
    assert len(episode.goal.times)==0 and len(episode.margins.times)==0
    assert episode.events[0].frame is None


def test_invalid_samples_are_reported_and_bgr_frame_is_converted(tmp_path):
    metadata=frame_metadata(.1,42,.1)
    records=[(TRUTH,0,truth(0.)), (TRUTH,.05,dict(schema_version=999)), (METADATA,.1,metadata),
             (MISSION,.2,mission(.2,diagnostics=dict(processed_image_stream_id='stream',processed_image_sequence=42,
                processed_image_timestamp_s=source(.1)['measurement_us']/1e6)))]
    path=write_mcap(tmp_path/'bgr.mcap',records,frames=[(.1,0,bytes([0,0,255])*48,'bgr8')])
    data=load_flight(path)
    assert any('1 unsupported or invalid' in n for n in data.notices)
    ref=next(e.frame for e in data.episodes[0].events if e.frame)
    assert read_frame(data,ref).rgb==bytes([255,0,0])*48
    from dataclasses import replace
    with pytest.raises(ValueError,match='missing'):
        read_frame(data,replace(ref,record_sequence=99))


def test_renderer_error_is_unsynced_not_a_guessed_flight_fault(tmp_path):
    from flight_lens.reader import RENDER_HEALTH
    stats={key:0 for key in ('frames','busy_slots','publication_drops','invalid','superseded','sequence_gaps','stale')}
    stats.update(source_age_ms=-1.,controller_id='controller',control_error='',shm_session='',error='render failed',running=True)
    path=write_mcap(tmp_path/'render-error.mcap',[(RENDER_HEALTH,.1,dict(schema_version=1,physics_id='physics',episode_id='episode',generated_us=1,stats=stats))])
    episode=load_flight(path).episodes[0]
    assert episode.events[0].time_s is None
    assert episode.events[0].severity=='error'
    assert episode.summary.outcome=='Unknown'


def test_physics_error_is_a_recorded_fault(tmp_path):
    from flight_lens.reader import STATE
    packet=dict(schema_version=1,physics_id='physics',episode_id='episode',controller_id='godot',state='error',reason='backend_failure',simulation_time_s=.1)
    path=write_mcap(tmp_path/'physics-error.mcap',[(STATE,.1,packet)])
    episode=load_flight(path).episodes[0]
    assert episode.summary.outcome=='Physics fault · backend_failure'
    assert episode.summary.errors==1


def test_success_recovers_from_state_and_deduplicates_terminal_events(tmp_path):
    from flight_lens.reader import STATE,END
    s=source(.1,10)
    collision=dict(schema_version=1,event_id='hit',controller_id='godot',issued_us=s['measurement_us'],source=s,target_id='red_truck',drone_position_m=[50.,0.,5.])
    result=dict(schema_version=1,event_id='hit',collision=collision,final_source=s,outcome='success',reason='target_hit',ended_us=s['measurement_us'])
    packet=dict(schema_version=2,physics_id='physics',episode_id='episode',controller_id='godot',state='stopped',reason='target_hit',simulation_time_s=.1,terminal_result=result)
    path=write_mcap(tmp_path/'success.mcap',[(STATE,.1,packet),(END,.2,result),(END,.3,result)])
    episode=load_flight(path).episodes[0]
    assert episode.summary.outcome=='Success · target_hit'
    assert sum(e.component=='scenario' for e in episode.events)==1


def test_corrupt_chunk_crc_is_rejected(flight_file,tmp_path):
    data=flight_file.read_bytes()
    assert b'"tracking"' in data
    path=tmp_path/'corrupt-crc.mcap'
    path.write_bytes(data.replace(b'"tracking"',b'"trAcking"',1))
    with pytest.raises(Exception,match='CRC|crc'):
        load_flight(path)
