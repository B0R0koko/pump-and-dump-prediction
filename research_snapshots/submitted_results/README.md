# Submitted manuscript results

This is the original submitted paper, not a newly executed result. `reported_tables.json`
preserves its numerical tables and displayed precision. The paper's submitted tables
and figures use these values. The additional fixed-seed notebook tables and bootstrap
intervals use the current frozen predictions, never substituted historical estimates.

Matching fitted models and per-event predictions for the original 80-event test run
were not found. The October prediction archive uses a different 52-event split and
cannot supply those predictions. Aggregate tables alone cannot reconstruct paired
bootstrap results. The cause of the numerical differences is not established; model
pipelines already set random seeds, so missing seeds are not asserted as the cause.

The original PDF/source are retained for provenance, including superseded wording.
The current paper keeps the revised limitations and temporal-testing description.
See `provenance.json` for source identifiers, hashes, and the additional-run artifacts.
