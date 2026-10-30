# Tracking CSV export

## Goal

Export the selected episode's tracking timeline, including bounding-box geometry and camera-center errors, and give all export actions a consistent visual treatment.

## Implementation steps

- Add a Tracking-tab CSV export action with one row per tracking sample and blank values for missing measurements.
- Include simulation time, phase/reason, bbox coordinates and size, bbox age, and `dx`/`dy` errors in the CSV.
- Add a shared export-button style used by MP4, TTC, and tracking exports.
- Add unit/UI regression coverage and update the plan menu after implementation.

## Interface changes

- Tracking tab exposes `Export tracking CSV`.
- CSV columns are `time_s`, `phase`, `reason`, `bbox_x`, `bbox_y`, `bbox_width`, `bbox_height`, `bbox_age_s`, `dx_px`, and `dy_px`.

## Verification

- Run `uv run --locked --package flight-lens pytest -q`.
- Run `git diff --check`.

## Scope and defaults

- Export includes every loaded tracking sample for the selected episode.
- Existing replay behavior and MP4/TTC exports remain unchanged.

## Status

Implemented and verified.

## Implementation commit

`1deacd0`
