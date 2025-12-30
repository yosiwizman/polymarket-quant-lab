# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Overview

Polymarket Quant Lab (`pmq`) is a quantitative trading platform for Polymarket prediction markets. Phase 1 focuses on market data ingestion, paper trading, and signal detection with safety guardrails. **No live trading is implemented** — all trades are simulated.

## Commands

### Setup
```powershell
poetry install
```

### Run CLI
```powershell
poetry run pmq --help
poetry run pmq sync              # Fetch market data from Gamma API
poetry run pmq scan              # Scan for arbitrage signals
poetry run pmq paper run         # Run paper trading loop
poetry run pmq report            # Generate PnL report
```

### Testing
```powershell
poetry run pytest                              # Run all tests
poetry run pytest tests/test_signals.py        # Run single test file
poetry run pytest -k "test_arb"                # Run tests matching pattern
poetry run pytest --cov=pmq --cov-report=html  # With coverage
```

### Linting & Type Checking
```powershell
poetry run ruff check src tests      # Lint
poetry run ruff format src tests     # Format
poetry run mypy src                  # Type check
```

## Architecture

### Data Flow
1. `GammaClient` fetches market data from Polymarket's Gamma API (`https://gamma-api.polymarket.com`)
2. `DAO` persists markets/signals/trades to SQLite (`data/pmq.db`)
3. Strategy scanners (`ArbitrageScanner`, `StatArbScanner`) detect trading signals
4. `PaperLedger` executes simulated trades with `SafetyGuard` enforcing limits

### Key Components

- **`src/pmq/cli.py`** — Typer CLI entry point. Commands: `sync`, `scan`, `paper run/positions/trades`, `report`
- **`src/pmq/gamma_client.py`** — HTTP client for Gamma API with file-based caching (`data/cache/gamma/`)
- **`src/pmq/config.py`** — Pydantic Settings with `PMQ_` env prefix. Nested configs: `SafetyConfig`, `ArbitrageConfig`, `StatArbConfig`
- **`src/pmq/models.py`** — Pydantic models for API responses (`GammaMarket`, `GammaEvent`) and internal types (`ArbitrageSignal`, `PaperTrade`, `PaperPosition`)
- **`src/pmq/storage/`** — SQLite layer: `db.py` (connection), `dao.py` (data access), `schema.sql` (DDL)
- **`src/pmq/strategies/`** — Trading logic: `arb.py` (arbitrage detection), `statarb.py` (stat-arb pairs), `paper.py` (paper ledger + safety)

### Configuration

Environment variables with `PMQ_` prefix configure the app:
- `PMQ_SAFETY_KILL_SWITCH` — Emergency halt (default: false)
- `PMQ_SAFETY_MAX_POSITIONS`, `PMQ_SAFETY_MAX_NOTIONAL_PER_MARKET`, `PMQ_SAFETY_MAX_TRADES_PER_HOUR` — Safety limits
- `PMQ_ARB_THRESHOLD`, `PMQ_ARB_MIN_LIQUIDITY` — Arbitrage scanner config
- `PMQ_STATARB_ENTRY_THRESHOLD`, `PMQ_STATARB_EXIT_THRESHOLD` — Stat-arb thresholds

Stat-arb market pairs are configured in `config/pairs.yml`.

### Code Patterns

- **Pydantic v2** for all models with `model_config = ConfigDict(...)`. Use `model_validate()` not deprecated `parse_obj()`
- **Type hints** are strict (`mypy --strict`). Tests are exempt from `disallow_untyped_defs`
- **Ruff** for linting/formatting. B008 is ignored (Typer default arguments)
- **Global singletons** via `get_settings()`, `get_database()` with `reset_*()` functions for testing
- **Context managers** on clients: `GammaClient`, `Database` support `with` statements

### Database

SQLite schema in `src/pmq/storage/schema.sql`. Tables: `markets`, `arb_signals`, `statarb_signals`, `paper_trades`, `paper_positions`, `audit_log`.

Auto-initialized on first access via `get_database()`.
