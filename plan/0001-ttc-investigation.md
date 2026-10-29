# TTC investigation tab

## Goal

Add a read-only TTC investigation tab that recalculates production-faithful optical TTC from tracking bboxes and compares it with recorded TTC.

## Implementation

- Add a pure TTC replay estimator with validated in-memory parameters.
- Add stacked raw/estimate plots, reason markers, summary counts, and a recalculate button.
- Use the selected episode and shared simulation-time cursor.

## Verification

Unit-test estimator qualification, rejection, reset, and validation behavior, then run the full FlightLens test suite.

## Status

Pending

## Implementation commit

Pending
