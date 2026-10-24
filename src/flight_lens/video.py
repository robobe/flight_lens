"""Bounded, cancellable simulation-time MP4 export through installed FFmpeg."""
from math import ceil, floor, isfinite
import os
from pathlib import Path
import selectors
import shutil
import subprocess
from tempfile import NamedTemporaryFile, TemporaryFile
import time

import numpy as np
from mcap.reader import make_reader

from .reader import check_cancel, check_file, decode_frame

FPS = 30
CAMERA_TOPIC = 'camera/front/image'
TICK_EPSILON = 1e-6
PIPE_POLL_S = .1
ENCODER_FINISH_TIMEOUT_S = 60
ENCODER_PROBE_TIMEOUT_S = 10
BOX_COLOR = (255, 180, 84)
BOX_LINE_PX = 2


def check_encoder():
    executable = shutil.which('ffmpeg')
    if executable is None:
        raise ValueError('MP4 export requires FFmpeg on PATH with the libx264 encoder.')
    try:
        result = subprocess.run([executable, '-hide_banner', '-encoders'], capture_output=True,
                                text=True, timeout=ENCODER_PROBE_TIMEOUT_S, check=True)
    except (OSError, subprocess.SubprocessError) as error:
        raise ValueError(f'Could not check FFmpeg: {error}') from error
    if not any(len(row.split()) > 1 and row.split()[1] == 'libx264' for row in result.stdout.splitlines()):
        raise ValueError('This FFmpeg installation does not provide the libx264 H.264 encoder.')
    return executable


def frame_schedule(references):
    """At each 30 Hz tick use the latest exposure, never a future exposure."""
    if not references:
        raise ValueError('No front-camera images with episode metadata. Reopen a recording with image metadata.')
    times = [ref.simulation_time_s for ref in references]
    if any(type(t) not in (float, int) or not isfinite(t) or t < 0 for t in times):
        raise ValueError('Image simulation timestamps are missing or invalid. No other clock is substituted.')
    if any(b < a for a, b in zip(times, times[1:])):
        raise ValueError('Image simulation timestamps decrease within this episode.')
    ticks = [(t - times[0]) * FPS for t in times]
    boundaries = [ceil(t - TICK_EPSILON) for t in ticks]
    total = floor(ticks[-1] + TICK_EPSILON) + 1
    return tuple(b - a for a, b in zip(boundaries, (*boundaries[1:], total)))


def render_frame(frame, bbox=None):
    """Annotate a copy and pad the bottom/right edges for H.264 chroma layout."""
    pixels = np.frombuffer(frame.rgb, dtype=np.uint8).reshape(frame.height, frame.width, 3)
    if bbox is not None:
        x, y, width, height = bbox
        if (not all(isfinite(v) for v in bbox) or x < 0 or y < 0 or width <= 0 or height <= 0
                or x + width > frame.width or y + height > frame.height):
            raise ValueError('Recorded bounding box exceeds the image dimensions.')
        pixels = pixels.copy()
        left, top = floor(x), floor(y)
        right, bottom = ceil(x + width), ceil(y + height)
        pixels[top:min(top + BOX_LINE_PX, bottom), left:right] = BOX_COLOR
        pixels[max(top, bottom - BOX_LINE_PX):bottom, left:right] = BOX_COLOR
        pixels[top:bottom, left:min(left + BOX_LINE_PX, right)] = BOX_COLOR
        pixels[top:bottom, max(left, right - BOX_LINE_PX):right] = BOX_COLOR
    if frame.height % 2 or frame.width % 2:
        pixels = np.pad(pixels, ((0, frame.height % 2), (0, frame.width % 2), (0, 0)))
    return pixels.tobytes()


def _write_frame(process, selector, pixels, cancel):
    remaining = memoryview(pixels)
    while remaining:
        check_cancel(cancel)
        if process.poll() is not None:
            raise BrokenPipeError('FFmpeg stopped accepting frames.')
        if selector.select(PIPE_POLL_S):
            try:
                size = os.write(process.stdin.fileno(), remaining)
            except BlockingIOError:
                continue
            remaining = remaining[size:]


def export_mp4(data, episode, destination, *, include_boxes=False, overwrite=False,
               progress=lambda message: None, cancel=lambda: False):
    check_cancel(cancel)
    check_file(data)
    destination = Path(destination).expanduser().resolve()
    if destination == data.path.resolve():
        raise ValueError('The video destination must not replace the MCAP recording.')
    if destination.suffix.lower() != '.mp4':
        raise ValueError('Choose an output filename ending in .mp4.')
    if destination.exists() and not overwrite:
        raise ValueError('Output already exists. Confirm replacement before exporting.')
    references = tuple(ref for ref in episode.frames if ref.topic == CAMERA_TOPIC)
    counts = frame_schedule(references)
    known = {(ref.log_time_ns, ref.record_sequence) for item in data.episodes
             for ref in item.frames if ref.topic == CAMERA_TOPIC}
    if data.camera_image_count is not None and len(known) != data.camera_image_count:
        raise ValueError('Some camera images lack unique episode/timing metadata. Reopen a fully stamped recording.')
    executable = check_encoder()
    expected = {(ref.log_time_ns, ref.record_sequence): (ref, count) for ref, count in zip(references, counts)}
    if len(expected) != len(references):
        raise ValueError('Ambiguous image record identity in this episode.')
    process = None
    temporary = None
    written = 0
    total = sum(counts)
    dimensions = None
    with TemporaryFile() as errors, selectors.DefaultSelector() as selector:
        try:
            with NamedTemporaryFile(prefix='.flightlens-', suffix='.mp4', dir=destination.parent, delete=False) as file:
                temporary = Path(file.name)
            with data.path.open('rb') as file:
                reader = make_reader(file, validate_crcs=True)
                for _, channel, message in reader.iter_messages(topics=[CAMERA_TOPIC],
                        starting_at=references[0].log_time_ns, ending_at=references[-1].log_time_ns):
                    check_cancel(cancel)
                    match = expected.pop((message.log_time, message.sequence), None)
                    if match is None:
                        if (message.log_time, message.sequence) not in known:
                            raise ValueError('An image record lacks episode/timing metadata.')
                        continue
                    reference, count = match
                    frame = decode_frame(channel, message)
                    if dimensions is None:
                        dimensions = (frame.width, frame.height)
                        width, height = frame.width + frame.width % 2, frame.height + frame.height % 2
                        process = subprocess.Popen([executable, '-hide_banner', '-loglevel', 'error', '-nostdin',
                            '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s:v', f'{width}x{height}',
                            '-r', str(FPS), '-i', 'pipe:0', '-an', '-c:v', 'libx264', '-preset', 'fast',
                            '-crf', '18', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(temporary)],
                            stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=errors, bufsize=0)
                        os.set_blocking(process.stdin.fileno(), False)
                        selector.register(process.stdin, selectors.EVENT_WRITE)
                    if dimensions != (frame.width, frame.height):
                        raise ValueError('Image resolution changes within this episode; export requires a fixed resolution.')
                    pixels = render_frame(frame, episode.boxes.get(reference.identity) if include_boxes else None)
                    for _ in range(count):
                        _write_frame(process, selector, pixels, cancel)
                        written += 1
                        if written % FPS == 0:
                            progress(f'Exporting MP4: {written:,}/{total:,} frames ({written / total:.0%})')
            if expected:
                raise ValueError('An exact image record is missing from this recording.')
            if process is None:
                raise ValueError('No image records found for this episode.')
            selector.unregister(process.stdin)
            process.stdin.close()
            deadline = time.monotonic() + ENCODER_FINISH_TIMEOUT_S
            while process.poll() is None:
                check_cancel(cancel)
                if time.monotonic() >= deadline:
                    raise ValueError('FFmpeg did not finish encoding within 60 seconds.')
                time.sleep(PIPE_POLL_S)
            if process.returncode:
                errors.seek(0)
                raise ValueError('FFmpeg encoding failed: ' + errors.read()[-4096:].decode(errors='replace'))
            check_cancel(cancel)
            check_file(data)
            if overwrite:
                os.replace(temporary, destination)
            else:
                # Publish without a race that could replace an unconfirmed existing file.
                os.link(temporary, destination)
                temporary.unlink()
            return destination
        except BrokenPipeError as error:
            errors.seek(0)
            raise ValueError('FFmpeg encoding failed: ' + errors.read()[-4096:].decode(errors='replace')) from error
        finally:
            if process is not None:
                if process.poll() is None:
                    process.kill()
                process.wait()
                if not process.stdin.closed:
                    process.stdin.close()
            if temporary is not None:
                temporary.unlink(missing_ok=True)
