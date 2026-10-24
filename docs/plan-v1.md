# FlightLens v1

Status: Implemented, verification complete. Implementation commit: `197874a3dae29ce7ecac74e372dbc9801fe66706`.

Create an independent desktop project in the shared uv workspace. PySide6
Essentials and PyQtGraph provide one Flight Overview tab. Keep MCAP reading,
immutable scalar data, summary/event analysis and widgets separate.

Show X–Z trajectory (metres), actual/barometer altitude with the recorded takeoff
goal, and four bbox edge clearances versus simulation time. Link time-plot zoom
and cursor, trajectory position and event selection. Summary includes outcome,
duration, maximum altitude, first fault, counts, recording integrity and Y range.
Collapse repeated status into phase and rejection/recovery intervals. Clipping
is a rejection/warning, not proof of flight failure. Event preview loads one
exact frame and bbox by original source identity, never nearest arrival time.

Prefer actual analysis ground truth; label old rendered-pose fallback explicitly.
Missing data stays unavailable. Add optional recorded takeoff goal/barometer datum
to mission diagnostics. Respect recorded episode identity and permit episode
selection for multi-episode files. Split plot curves at >5 median sample intervals.

Read scalar topics on a cancellable worker, lazily read images, ignore obsolete
results, and keep GUI operations on the main thread. Read-only indexed MCAP,
including legacy files; report corrupt/unindexed files without repairing them.
Test pure analysis, all bbox edges, source matching/counter reuse, foreign epochs,
missing data, GUI synchronization/cancellation/errors and real-display layout.
Exercise an existing ~800 MB recording without loading video into RAM.

Defer live data, video playback, extra tabs, automatic diagnosis/tuning, parameter
editing and run comparison remain deferred.


Verification outcomes:
- 21 FlightLens reader/analysis/Qt checks passed: all four edges, warning versus
  flight fault, event intervals, source and epoch matching, frame counter reuse,
  BGR conversion, lazy images, immutable data, missing/invalid data, partial/index
  errors and chunk CRC validation, gap splitting, cursor/zoom linking, cancellation, stale results,
  safe worker shutdown, explicit image errors, physical faults and terminal-result recovery/deduplication.
- All 84 bullet_app checks passed, including explicit goal/datum telemetry.
- uv resolved and locked Qt dependencies without removing existing workspace
  packages. Lock consistency and whitespace checks passed.
- Real X11 desktop verification loaded the existing 759.3 MiB legacy MCAP in
  ~3.1 s and a 1245.5 MiB current recording in ~9.4 s. GUI timer callbacks
  continued throughout loading; image pixels remained on disk until selection.
  A final CRC-validated read of the current recording completed in ~10.0 s.
  Peak GUI process RSS was about 300 MiB for the current recording (not a
  universal memory/performance guarantee).
- Visually inspected the three plots, selected-event frame and bbox, warning
  rows, trajectory cursor and legacy missing-data notices. Snapshot:
  /tmp/flight-lens-overview.png. Timing records:
  /tmp/flight-lens-comparison-metrics.json and /tmp/flight-lens-desktop-metrics.json.

Project initialized as an independent Git repository and registered in the
shared workspace. Only Linux/X11 was checked.
Existing recordings correctly lack the newly added explicit goal; future mission
status publications include the controller goal and sensor datum without changing
control state. Additional tabs, live data and tuning remain deferred.

Supporting commits:
- App goal/datum telemetry: `08cbf83561c127a459039373ec1d2bee05809f06`.
- Shared workspace and dependency lock: `963445ff2a001ffdecdf9ea6fc97cb045c67ca25`.
