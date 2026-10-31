# Causal TTC smoothing

## Goal

Add a real-time-safe TTC smoother that handles irregular sample intervals and
resets across invalid estimates, then use it to review the recorded flight.

## Implementation

- Add a time-aware exponential low-pass function to `ttc_analysis` with a
  configurable time constant and no future-sample dependence.
- Preserve the existing fixed-alpha helper for compatibility with current UI
  controls and callers.
- Add unit coverage for validation, irregular timing, and invalid-gap reset.
- Generate a TTC output graph from the recorded native TTC samples using the new
  causal filter.

## Verification

- Run the focused TTC analysis tests and the complete FlightLens test suite.
- Inspect the generated graph and run diff checks.

## Status

Pending.

## Implementation commit

Pending.
