# Deterministic demo MCAP fixture

## Goal

Provide a tiny, reproducible MCAP recording for FlightLens development and manual UI testing.

## Implementation steps

- Add an example generator with a stable one-episode flight, tracking/TTC data, target-hit completion, and a linked tiny camera frame.
- Check in the generated `examples/demo.mcap` fixture despite the repository-wide MCAP ignore rule.
- Document regeneration and launch commands.
- Add reader regression coverage for generated and committed fixtures.

## Interface changes

- `examples/create_demo_mcap.py [output]` writes a deterministic demo recording, defaulting to `examples/demo.mcap`.

## Verification

- Generate into a temporary path and load with `load_flight`.
- Load the committed fixture and assert tracking, bbox/dx/dy, TTC, and image data.
- Run `uv run --locked --package flight-lens pytest -q` and `git diff --check`.

## Scope and defaults

- The fixture is development/manual-test data, not a production recording.
- The generator is the source of truth; the binary fixture is force-added for instant opening.

## Status

Implemented and verified.

## Implementation commit

`8dc605d`
