# MP4 export

Status: implemented and verified.

Add Export MP4 to FlightLens for the complete selected episode's front camera.
Ask whether to include exact-frame bounding boxes (unchecked by default). No text
or telemetry overlays. Use simulation time at 30 FPS, repeating/dropping frames
as necessary. Encode H.264/yuv420p with fast-start through installed FFmpeg.

Keep metadata in memory and stream image records sequentially. Share RGB/BGR
validation with event previews. Preserve dimensions, padding odd sizes to even;
reject missing/decreasing timing, corrupt frames and changing resolution.

Run on a cancellable worker with progress. Save to a temporary sibling MP4 and
publish only after successful encoding. Cancellation, file changes and encoder
errors must preserve the original recording and any existing output. Opening a
different recording or closing the window cancels conversion.

Verify timing/resampling, episode and exact-box selection, image layouts,
dependency errors, cancellation and output preservation. Use real FFmpeg/ffprobe
integration tests and export the recorded failing flight for visual checks.

No new Python dependencies, audio, video playback, hardware validation or
additional camera selection are included.

## Verification

- `uv run --locked --no-sync pytest -q`: 36 tests passed, including real
  FFmpeg/ffprobe exports, timestamp resampling, exact boxes, episode isolation,
  cancellation of a blocked encoder, failed writes and replacement races.
- Converted `bullet_landscape_20261024T182354.798893Z.mcap` with bounding boxes:
  1,056 recorded images, 1,055 video ticks, 640×480 H.264/yuv420p at 30 FPS,
  duration 35.166667 seconds. Metadata coverage checked against the MCAP index.
- Visually inspected decoded first, middle and clipped-target frames and the
  offscreen Qt export dialog. No live desktop interaction was tested.
- Original recording and existing output files are preserved on failure.
