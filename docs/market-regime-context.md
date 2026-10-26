# Optional Market-Regime Context

`TradingAgentsGraph.propagate()`, `_run_graph()`, and
`Propagator.create_initial_state()` accept `market_regime_context=None` as an
optional final argument. Omission leaves existing prompts, policy, and state
behavior unchanged. Supplying an empty or malformed object is not omission: it
creates an explicit unavailable marker, never a run failure or a directional fact.

## Snapshot Contract

The producer supplies JSON-compatible values with these fields:

```json
{
  "schema_version": 1,
  "run_date": "2026-08-19",
  "data_date": "2026-08-18",
  "status": "COMPLETE",
  "universe": {
    "name": "LSTM inference universe",
    "expected_count": 190,
    "valid_count": 190
  },
  "values": {
    "mean_return_5d": -0.03,
    "mean_return_20d": -0.07,
    "above_sma50_fraction": 0.20,
    "above_sma200_fraction": 0.40
  },
  "source": {
    "path": "/producer/inference.csv",
    "sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
  },
  "missing_fields": [],
  "exclusions": {}
}
```

The example digest is illustrative; the producer supplies the actual file hash.
Returns are fractions, not percentage points. They must be finite and at least
-1. Breadth fractions must be within [0, 1]. Booleans and numeric strings are not
numbers. Counts must be integers with `0 <= valid_count <= expected_count`.

`COMPLETE` requires positive coverage of the entire expected universe, all four
metrics, and no missing fields. `PARTIAL` permits valid-subset coverage and missing
metrics: omit the value or use null and name that metric in `missing_fields`.
Partial snapshots require positive coverage and at least one valid metric.
`UNAVAILABLE` exposes no directional values, even if the supplied object contains
numbers. Exclusions remain in the audit snapshot, not instructions to the model.

## Validation Ownership

The root producer validates the required previous completed NSE exchange date
and hashes the actual source bytes. TA does not read source files, caches, or
network data to reconstruct the snapshot. It validates schema version, strict
YYYY-MM-DD dates, run-date equality, `data_date < run_date`, source path and digest
shape, counts, finite numeric ranges, missing fields, and JSON shape.

When validated LSTM context is also supplied, the regime `data_date` must equal
the LSTM signal `data_timestamp` date. Mismatch makes only the regime unavailable;
it does not invalidate the otherwise usable LSTM run. TA has no standalone NSE
calendar dependency. Without LSTM context, exact previous-session verification
remains a producer responsibility; an earlier date alone does not prove freshness.

## State and Reports

- `market_regime_context`: a defensive copy of the exact supplied JSON snapshot.
- `market_regime_validation`: effective `status` and validation `errors`, separate
  from the original producer status.
- `market_regime_context_note`: neutral dated facts with valid/expected coverage,
  or an explicit unavailable/missing-facts note.
- `market_regime_report`: rendered facts, validation outcome, and exact supplied
  JSON for audit. Invalid audit values are not projected into agent prompts.

These keys are absent on omitted-context runs. They persist in graph state and
`full_states_log_*.json`. The CLI writes `market_regime.md` and includes that
section in `complete_report.md`; external report writers can use
`market_regime_report` directly. Non-JSON Python objects have no exact JSON
representation and are instead archived under `invalid_input_repr`.

`MARKET_REGIME_PARTIAL` and `MARKET_REGIME_UNAVAILABLE` are descriptive quality
tags. They never mark an otherwise usable analysis FAILED.

## Role and Decision Semantics

All four independent analysts get neutral facts through `get_analyst_context()`.
They receive no model rank, score, or strategy caveat. All eight downstream roles
get the same facts through `get_data_quality_instruction()`. Only runs with both
the new context and validated LSTM evidence get the additional strategy caveat:
pullback within established momentum, a high score does not establish safety,
and no calibrated bull/bear probabilities are supplied.

Downstream prompts explicitly reconcile observed entry support with the strongest
contradictory evidence and explain the target-before-stop path within the supplied
horizon (seven sessions for the current LSTM contract). A confirmed reversal,
catalyst, or relative strength is not a universal prerequisite. Missing facts and
forecast uncertainty remain separate. When dates match and both values exist,
downstream context includes `ticker_minus_universe_return_20d` as a derived
percentage-point difference, not a requirement or probability.

Research, trader, and portfolio schemas optionally expose
`regime_fit_assessment` (`SUPPORTIVE`, `ADVERSE`, `MIXED`, `UNKNOWN`) and
`regime_fit_reason`. Their rendered decision reports include those fields when
supplied; legacy responses can omit them. Unavailable facts imply UNKNOWN, not
ADVERSE. Regime assessment is not a deterministic BUY/HOLD/SELL mapping.

Bounded grounding checks compare explicitly named regime numeric claims with
supplied values. They distinguish ticker facts and conditional scenarios. They
do not certify arbitrary prose or infer missing values. A factual repair gets at
most one attempt; for new-context runs, a repair that changes the explicit action
is rejected and the original draft retained with visible unresolved issues.
There is no Shadow5 gate or other admission rule in this consumer.

## Offline Verification

Use the existing project environment and exclude live integration tests:

```sh
python -m pytest tests/test_market_regime_context.py -q
python -m pytest -q -m 'not integration'
```

The consumer tests use local fake model responses and a real compiled LangGraph
to verify routing, all three actions, debug/non-debug propagation, unavailable
and partial contexts, omitted behavior, schemas, grounding and artifact persistence.
The repository also contains opt-in live API tests marked `integration`; an
unfiltered run can call them when credentials are present.
