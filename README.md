# FlightLens

Read-only desktop flight analysis for local Bullet MCAP recordings. The **Flight
Overview** tab combines X–Z trajectory, altitude, bbox edge clearance, an event
summary and the exact recorded camera frame for a selected event. The **Attitude
& Thrust** tab plots physical and controller attitude, commanded collective thrust,
and the start of LOS tracking.

## Run

This independent project belongs to the shared Bullet uv workspace. From this
folder, install all workspace packages without removing other projects' tools:

```bash
uv sync --locked --all-packages --all-extras --all-groups
uv run --locked --all-packages --all-extras --all-groups flight-lens
```

Open a file directly:

```bash
uv run --locked --all-packages --all-extras --all-groups flight-lens \
  ../bullet_render/recordings/<flight>.mcap
```

Workspace task **FlightLens: Open** starts the same desktop window. Click **Open
MCAP…** to select a local recording. No browser, server, live flight connection,
configuration edits or changes to the original recording are involved. PySide6
Essentials provides the Qt desktop widgets, and PyQtGraph provides the plots.
Linux/Ubuntu with a working desktop display is the verified environment.

## Flight Overview

| View | Recorded information |
| --- | --- |
| X–Z trajectory | World X and world Z in metres, equal-distance aspect ratio, start triangle, end square and selected-time circle |
| Altitude over simulation time | Actual world altitude, barometer altitude and explicit controller takeoff goal during takeoff/hover/search |
| Bbox edge clearance | Left/top/right/bottom clearances in pixels; zero is boundary contact, negative values are beyond the boundary |
| Summary | Outcome, recorded duration, peak altitude, first recorded fault, warning/error interval counts and recording integrity |
| Events | Phase changes, visual rejection and navigation recovery intervals, physics state changes and renderer errors |
| Event frame | One original image with its bbox, when an exact recorded identity and image are available |

## Attitude & Thrust

The attitude graph shows roll, pitch and yaw in degrees from the recorded physical
pose and controller feedback when available. The thrust graph shows normalized
collective thrust reconstructed from the recorded RC command; it is not measured
delivered motor thrust. A vertical marker identifies the first recorded LOS tracking
sample (`TRACK` / `LOS_APPROACH`). Missing topics remain visible as unavailable data
instead of preventing the recording from opening.

## Velocity

The Velocity tab shows ground-truth X, Y and Z velocity in metres per second.
The Z panel also shows vertical velocity derived from the recorded barometer
altitude, using the sample timestamps. Its line is distinct from ground truth;
it is unavailable when fewer than two altitude samples are recorded.
The X–Z slant-velocity panel shows ground-truth travel speed in the displayed
flight plane, calculated as `sqrt(vx² + vz²)`.

Drag the white cursor in either time plot; both plots share time and horizontal
zoom. The position circle follows that simulation time on the trajectory, and
the footer shows nearby recorded data ages/limits. Clicking an event selects its
time, details and camera frame. The camera preview stays tied to the **selected
event**, even when the time cursor subsequently moves. There is no in-app video playback.
Y displacement is reported separately because the X–Z view hides lateral motion.

Repeated status packets become intervals rather than hundreds of event rows.
Counts are recorded operational event intervals/transitions, not packet counts.
Clipping is a warning/rejection, not proof of a failed flight or its cause.
Flight outcome and recorder completeness are independent. A renderer error
without a trustworthy simulation timestamp is listed as **unsynced**; selecting
it does not invent a cursor time. Cursor readouts use a nearest sample within
100 ms; data outside that window is unavailable. Curves break at gaps larger
than five times the series' median positive sampling interval.

## Source handling and compatibility

- `analysis/physics/ground_truth` is preferred for each episode. If absent,
  `render/drone_1/pose` is explicitly labelled **Rendered pose**, which may
  contain noisy sensor-derived altitude or attitude. Peak altitude refers to
  the displayed source, not inferred ground truth.
- Barometer readings come from `analysis/physics/sensors`, with selected feedback
  altitude as an older-file fallback. These are the sensor's exposed outputs,
  including configured filtering; raw values are not reconstructed.
- The optional mission diagnostic fields `takeoff_goal_altitude_m` and
  `barometer_datum_m` expose the controller's sensor-based takeoff reference.
  The goal is not reconstructed for older recordings and is not presented as
  a descent reference during LOS tracking.
- Edge clearances use stamped detector boxes and recorded dimensions, or
  processed-box margin diagnostics when detector data is absent. The plot title
  identifies which source is used.
- Event images require `<image topic>/metadata`, keyed by original physics,
  episode, stream, frame sequence and capture timestamp. The image itself is
  located by recorded image log time and recorder sequence. Frame counter reuse
  after renderer restart therefore cannot select the wrong exposure. Arrival
  proximity alone never substitutes for an exact match.
- The authoritative episode in `recording_health` is selected by default. With
  older files the most recently first-seen episode is selected; the episode
  dropdown allows inspection of other episodes in the same file.
- Missing topics, invalid/unsupported packets, absent goals and recording skips
  are visible notices. Indexed partial files show **Incomplete**. Unindexed or
  corrupt files produce an error; recorded CRCs are checked. Recover/index a copy using MCAP tooling first.

Scalar channels load on a cancellable worker. A file change supersedes older
jobs, while cancellation/error retains a previously loaded flight. Images stay
on disk until selected; the GUI caches at most three decoded frames. Files that
change after loading must be reopened before loading an image.

## Export MP4

Open an MCAP, select its episode, and click **Export MP4…**. Choose a destination
and optionally check **Include bounding boxes** (off by default). Only exact-frame
detector boxes are drawn; missing detections remain unannotated. No text or flight
telemetry is added. The default filename is `<recording>-episode-<number>.mp4`.

The selected episode's front-camera images are encoded as H.264, `yuv420p`, at
30 FPS with fast-start enabled. Playback follows recorded **simulation time**,
starting at the first image: recording-time pauses do not extend the movie. Each
video tick uses the latest available exposure, repeating or dropping frames as
necessary. The movie ends at the last simulation-time tick covered by the images,
with one frame of display duration. Odd image dimensions are padded with black
along the bottom/right edges, without scaling or changing image proportions.

FFmpeg with the `libx264` encoder must be installed on PATH. The app checks it
before conversion and reports missing support; it does not install system tools.
Missing/decreasing simulation timestamps, missing image records, invalid layouts,
changing image resolution and changed input files cause an explicit error.

Export runs in the background with progress and **Cancel export**. Opening another
recording or closing FlightLens also cancels it. Images are streamed rather than
loaded together into memory. Output is encoded to a temporary sibling file and
published only on success. Replacing an existing output requires confirmation;
failures and cancellation preserve existing files. The original MCAP is never
modified. Export currently supports one selected episode and front camera at a
time; audio and other overlays are not included.

## SQLite llama.cpp example

The standalone `examples/sqlite_llama_query.py` script creates a deterministic
SQLite table with 1,000 `(time, value)` samples, asks the local GGUF model to
translate natural language into SQL, validates the result as a read-only
`SELECT`, and executes at most 50 rows.

Run it from this directory with the configured model:

```bash
uv run --locked --extra assistant python examples/sqlite_llama_query.py
```

The REPL prints example prompts when it starts. Use `--assistant-model` to
override the YAML model path or `--database` to choose the demo database file.
Generated SQL is displayed before the bounded result table. Writes, multiple
statements, unknown tables/columns, and oversized limits are rejected.

## Development and checks

```bash
uv run --locked --all-packages --all-extras --all-groups pytest -q
```

The default `config/flight_lens.yaml` points the optional local assistant at the
configured GGUF model. Override the settings file or model path when needed:

```bash
uv run --locked --extra assistant --package flight-lens flight-lens \
  --config config/flight_lens.yaml

uv run --locked --extra assistant --package flight-lens flight-lens \
  --assistant-model /path/to/model.gguf
```

For NVIDIA CUDA offloading, install the CUDA-enabled llama.cpp backend before
launching FlightLens:

```bash
CMAKE_ARGS="-DGGML_CUDA=on" uv pip install --python .venv/bin/python \
  --reinstall --no-cache-dir llama-cpp-python
```

FlightLens requests all model layers on the GPU (`n_gpu_layers=-1`). A CPU-only
build remains usable but will not provide CUDA acceleration.

Qt tests use its offscreen platform. Real-display checks additionally exercise
large recordings, rendering, event selection and exact image preview. See
[the saved v1 plan](docs/plan-v1.md) for verification outcomes.

The implementation separates `model.py` (immutable data), `reader.py` (MCAP and
analysis), and `app.py` (workers and widgets). Additional topic-specific tabs can
consume the same loaded dataset without adding recording/networking to widgets.
Live data, additional tabs, comparison, in-app playback and automated tuning/diagnosis
are deferred.
