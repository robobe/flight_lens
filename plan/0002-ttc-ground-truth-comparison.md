# TTC ground-truth comparison

## Goal

Add a retrospective target-hit-time TTC oracle to the TTC investigation tab and compare it with the replayed estimate.

## Implementation

- Derive oracle TTC only from a recorded `target_hit` completion event.
- Plot the oracle beside the recalculated and Butterworth estimate curves.
- Report MAE, RMSE, bias, and valid coverage for overlapping qualified samples.
- Keep the oracle investigation-only and never feed it into flight control.

## Verification

Test successful and non-hit episodes, metric alignment, missing oracle behavior, and run the full FlightLens suite.

## Status

Implemented

## Implementation commit

`b367df5`
