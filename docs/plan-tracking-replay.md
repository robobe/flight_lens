# Tracking-control replay and tuning tab

## Goal

Provide a read-only replay of all `LOS_*` mission/control ticks with a camera
schematic, bbox/error/TTC values, attitude, command values, and step or
simulation-rate playback controls.

## Implementation steps

1. Retain mission tracking samples and nearest bbox/LOS/attitude/RC values in
   the immutable FlightLens episode model.
2. Add the Tracking tab with a 2D artificial horizon, camera rectangle,
   telemetry panel, step controls, and simulation-rate playback.
3. Add reader/model and Qt regression coverage.

## Interface changes

Add an internal `TrackingSample` model field to `Episode`; no MCAP or recorder
schema changes are required.

## Verification

Run the FlightLens test suite and manually replay a recording containing LOS
approach, acquisition, recovery, and commit phases.

## Status

Implemented. Includes sub-1× playback accumulation, 0.1× playback, camera
timeline scrubbing, issue navigation, recorded frame evidence, and
control-oriented issue explanations. Verification passed with 52 tests.

## Implementation commit

`0dd2b6f2fd9be60bc796535df0ce89455074bf84`
