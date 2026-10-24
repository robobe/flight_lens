"""Video timing, frame identity and real FFmpeg lifecycle checks."""
from dataclasses import replace
import json
import shutil
import subprocess
from threading import Timer, Event
import time

import numpy as np
import pytest
from foxglove_schemas_protobuf.RawImage_pb2 import RawImage

from flight_lens.model import Frame
from flight_lens.reader import BBOX, METADATA, TRUTH, Cancelled, decode_frame, load_flight
from flight_lens import video
from conftest import frame_metadata, source, truth, write_mcap


@pytest.fixture
def movie_file(tmp_path):
    records, frames = [], []
    for i, (t, color) in enumerate(((0., (255, 0, 0)), (.01, (0, 255, 0)), (.1, (0, 0, 255)))):
        arrival = 1. + i * 2.  # Wall-clock gaps must not slow the movie down.
        records.append((METADATA, arrival, frame_metadata(t, 42, arrival, i)))
        records.append((TRUTH, arrival, truth(t)))
        frames.append((arrival, i, bytes(color) * 48, 'rgb8'))
    # Same frame number, another episode: never overlay or export its image.
    records.append((METADATA, 8., frame_metadata(0., 42, 8., 3, episode='other')))
    records.append((TRUTH, 8., truth(0., episode='other')))
    frames.append((8., 3, bytes([255, 255, 255]) * 48, 'rgb8'))
    s = source(.01, 1)
    records.append((BBOX, 4., dict(schema_version=2, image_topic=video.CAMERA_TOPIC,
        session='a'*32, sequence=42, capture_ns=s['measurement_us']*1000, source=s,
        bbox=dict(x=0, y=1, width=2, height=2))))
    return write_mcap(tmp_path/'movie.mcap', records, frames=frames)


@pytest.fixture
def encoder():
    if not shutil.which('ffmpeg') or not shutil.which('ffprobe'):
        pytest.skip('FFmpeg and ffprobe required for video integration')
    return video.check_encoder()


def decode_video(path):
    raw = subprocess.check_output(['ffmpeg', '-v', 'error', '-i', str(path), '-f', 'rawvideo',
                                   '-pix_fmt', 'rgb24', 'pipe:1'])
    info = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-show_streams', '-of', 'json', str(path)]))['streams'][0]
    pixels = np.frombuffer(raw, dtype=np.uint8).reshape(-1, info['height'], info['width'], 3)
    return info, pixels


def test_export_simulation_clock_episode_selection_and_exact_boxes(movie_file, tmp_path, encoder):
    data = load_flight(movie_file)
    episode = data.episodes[0]
    assert video.frame_schedule(episode.frames) == (1, 2, 1)
    assert [episode.boxes.get(ref.identity) for ref in episode.frames] == [None, (0, 1, 2, 2), None]
    output = tmp_path/'flight.mp4'
    assert video.export_mp4(data, episode, output) == output
    info, pixels = decode_video(output)
    assert info['codec_name'] == 'h264'
    assert info['pix_fmt'] == 'yuv420p'
    assert info['r_frame_rate'] == '30/1'
    assert int(info['nb_frames']) == 4
    assert float(info['duration']) == pytest.approx(4/30, abs=.001)
    assert np.argmax(pixels.mean(axis=(1, 2)), axis=1).tolist() == [0, 1, 1, 2]
    boxed = tmp_path/'boxed.mp4'
    video.export_mp4(data, episode, boxed, include_boxes=True)
    _, annotated = decode_video(boxed)
    np.testing.assert_array_equal(annotated[[0, 3]], pixels[[0, 3]])
    assert not np.array_equal(annotated[1:3], pixels[1:3])
    assert not list(tmp_path.glob('.flightlens-*'))


def test_resampling_equal_timestamps_and_high_frame_rate(movie_file):
    refs = load_flight(movie_file).episodes[0].frames
    def schedule(times):
        return video.frame_schedule([replace(refs[0], simulation_time_s=t) for t in times])
    assert schedule([4.]) == (1,)
    assert schedule([4., 4., 4.1]) == (0, 3, 1)
    assert schedule([0., .01, .02]) == (1, 0, 0)
    assert schedule([0., .1, .2]) == (3, 3, 1)
    for times in ([0., None], [0., float('nan')], [1., 0.]):
        with pytest.raises(ValueError, match='timestamps'):
            schedule(times)
    with pytest.raises(ValueError, match='metadata'):
        video.frame_schedule([])


def test_image_unpack_padding_boxes_and_odd_size():
    from types import SimpleNamespace
    image = RawImage(width=5, height=3, step=18, encoding='bgr8', data=(bytes([0, 0, 255])*5+b'pad')*3)
    frame = decode_frame(SimpleNamespace(message_encoding='protobuf'), SimpleNamespace(data=image.SerializeToString()))
    assert frame.rgb == bytes([255, 0, 0])*15
    original = np.frombuffer(video.render_frame(frame), dtype=np.uint8).reshape(4, 6, 3)
    assert np.all(original[:3, :5] == [255, 0, 0])
    assert not original[3].any() and not original[:, 5].any()
    # The only modified pixels are the requested border. There is no text overlay.
    big = Frame(40, 30, bytes([0])*40*30*3)
    pixels = np.frombuffer(video.render_frame(big, (4, 5, 20, 16)), dtype=np.uint8).reshape(30, 40, 3)
    assert np.all(pixels[5:7, 4:24] == video.BOX_COLOR)
    assert not pixels[7:19, 6:22].any()
    assert not pixels[:5].any()
    with pytest.raises(ValueError, match='bounding box'):
        video.render_frame(big, (39, 0, 5, 5))
    image.step = 1
    with pytest.raises(ValueError, match='layout'):
        decode_frame(SimpleNamespace(message_encoding='protobuf'), SimpleNamespace(data=image.SerializeToString()))


def test_odd_size_h264_integration(tmp_path, encoder, monkeypatch, flight_file):
    data = load_flight(flight_file)
    monkeypatch.setattr(video, 'decode_frame', lambda *args: Frame(5, 3, bytes([255, 0, 0])*15))
    output = tmp_path/'odd.mp4'
    video.export_mp4(data, data.episodes[0], output)
    info, _ = decode_video(output)
    assert (info['width'], info['height']) == (6, 4)


def test_failure_and_cancellation_preserve_existing_output(movie_file, tmp_path, encoder, monkeypatch):
    data = load_flight(movie_file)
    episode = data.episodes[0]
    output = tmp_path/'existing.mp4'
    output.write_bytes(b'keep existing video')
    with pytest.raises(ValueError, match='already exists'):
        video.export_mp4(data, episode, output)
    def cancel(message):
        raise Cancelled()
    monkeypatch.setattr(video, '_write_frame', lambda *args: cancel('stop'))
    with pytest.raises(Cancelled):
        video.export_mp4(data, episode, output, overwrite=True)
    assert output.read_bytes() == b'keep existing video'
    assert not list(tmp_path.glob('.flightlens-*'))


def test_changed_input_missing_frames_and_resolution_change(movie_file, tmp_path, encoder, monkeypatch):
    data = load_flight(movie_file)
    episode = data.episodes[0]
    output = tmp_path/'bad.mp4'
    refs = (*episode.frames[:-1], replace(episode.frames[-1], record_sequence=999))
    with pytest.raises(ValueError, match='missing'):
        video.export_mp4(data, replace(episode, frames=refs), output)
    decode = video.decode_frame
    calls = []
    def resize(*args):
        calls.append(1)
        return decode(*args) if len(calls) == 1 else Frame(2, 2, bytes(12))
    monkeypatch.setattr(video, 'decode_frame', resize)
    with pytest.raises(ValueError, match='resolution'):
        video.export_mp4(data, episode, output)
    movie_file.touch()
    with pytest.raises(ValueError, match='changed'):
        video.export_mp4(data, episode, output)
    assert not output.exists() and not list(tmp_path.glob('.flightlens-*'))


def test_encoder_errors_and_cancellation_of_blocked_pipe(movie_file, tmp_path, monkeypatch):
    data = load_flight(movie_file)
    output = tmp_path/'failure.mp4'
    executable = tmp_path/'encoder'
    executable.write_text('#!/usr/bin/env python3\nimport sys\nsys.stderr.write("encoder test failure\\n")\nsys.exit(7)\n')
    executable.chmod(0o700)
    monkeypatch.setattr(video, 'check_encoder', lambda: str(executable))
    with pytest.raises(ValueError, match='FFmpeg encoding failed'):
        video.export_mp4(data, data.episodes[0], output)
    assert not output.exists()
    executable.write_text('#!/usr/bin/env python3\nimport time\ntime.sleep(60)\n')
    monkeypatch.setattr(video, 'decode_frame', lambda *args: Frame(200, 200, bytes(200*200*3)))
    cancelled = Event()
    timer = Timer(.3, cancelled.set)
    started = time.monotonic()
    timer.start()
    try:
        with pytest.raises(Cancelled):
            video.export_mp4(data, data.episodes[0], output, cancel=cancelled.is_set)
    finally:
        timer.cancel()
    assert time.monotonic() - started < 3
    assert not output.exists() and not list(tmp_path.glob('.flightlens-*'))


def test_missing_encoder_and_unusable_output(movie_file, tmp_path, monkeypatch):
    monkeypatch.setattr(video.shutil, 'which', lambda name: None)
    with pytest.raises(ValueError, match='FFmpeg on PATH'):
        video.check_encoder()
    data = load_flight(movie_file)
    with pytest.raises(ValueError, match='MCAP recording'):
        video.export_mp4(data, data.episodes[0], movie_file)
    with pytest.raises(ValueError, match='ending in .mp4'):
        video.export_mp4(data, data.episodes[0], tmp_path/'wrong.avi')


def test_incomplete_metadata_does_not_silently_omit_images(tmp_path):
    records = [(TRUTH, 0., truth(0.)), (METADATA, 0., frame_metadata(0., 1, 0.))]
    path = write_mcap(tmp_path/'missing-metadata.mcap', records,
        frames=[(0., 0, bytes(144), 'rgb8'), (1., 1, bytes(144), 'rgb8')])
    data = load_flight(path)
    with pytest.raises(ValueError, match='timing metadata'):
        video.export_mp4(data, data.episodes[0], tmp_path/'missing.mp4')


def test_confirmed_overwrite_and_publication_race(movie_file, tmp_path, encoder, monkeypatch):
    data = load_flight(movie_file)
    output = tmp_path/'existing.mp4'
    output.write_bytes(b'old')
    video.export_mp4(data, data.episodes[0], output, overwrite=True)
    assert decode_video(output)[0]['nb_frames'] == '4'
    output.unlink()
    publish = video.os.link
    def race(source, destination):
        destination.write_bytes(b'created while encoding')
        return publish(source, destination)
    monkeypatch.setattr(video.os, 'link', race)
    with pytest.raises(FileExistsError):
        video.export_mp4(data, data.episodes[0], output)
    assert output.read_bytes() == b'created while encoding'
    assert not list(tmp_path.glob('.flightlens-*'))
