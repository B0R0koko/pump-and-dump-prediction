---
type: module
title: "core"
created: 2026-04-19
updated: 2026-04-19
tags:
  - module
  - core
status: seed
related:
  - "[[market_data]]"
  - "[[preprocessing]]"
  - "[[features]]"
sources: []
path: "core/"
language: python
purpose: "Shared domain types, column constants, paths, and time utilities used by every other module."
maintainer: borokoko
last_updated: 2026-04-19
linked_issues: []
depends_on: []
used_by:
  - "[[market_data]]"
  - "[[preprocessing]]"
  - "[[features]]"
  - "[[backtest-pipelines]]"
  - "[[backtest-portfolio]]"
---

# core

## Purpose
Foundational layer. Defines the vocabulary the rest of the codebase speaks: domain types (`PumpEvent`, `CurrencyPair`, `FeatureType`, `NamedTimeDelta`, `Exchange`), canonical column names, dataset paths, and time utilities.

## Public surface
- Domain types listed above.
- `core/paths.py` — dataset path constants. Datasets expected under `/var/lib/pumps/data`.

## Key files
- `core/paths.py` — dataset path constants
- `core/exchange.py` — `Exchange` enum
- `core/pump_event.py` — `PumpEvent` dataclass
- `core/currency_pair.py` — `CurrencyPair` dataclass
- `core/feature_type.py` — `FeatureType` enum
- `core/columns.py` — canonical column name constants shared by all modules
- `core/time_utils.py` — time window and offset utilities
- `core/utils.py` — miscellaneous shared helpers

## Used by
- [[market_data]], [[preprocessing]], [[features]], [[backtest-pipelines]], [[backtest-portfolio]]

## Design notes
- Hardcoded path `/var/lib/pumps/data` is the contract; `CLAUDE.md` says do not introduce alternatives.
