# Polymarket Quant Lab

A quantitative trading platform for Polymarket prediction markets. Phase 1 focuses on market data ingestion, paper trading, and signal detection with safety guardrails.

> ⚠️ **IMPORTANT**: This is Phase 1 - no live trading is implemented. All trades are simulated paper trades. No API keys or wallets are required.

## Features

- **Market Data Ingestion**: Fetch market metadata and prices from Polymarket's Gamma API
- **Signal Detection**:
  - Deterministic arbitrage: Detect when YES + NO prices < threshold
  - Statistical arbitrage: Monitor spread divergence between correlated market pairs
- **Paper Trading**: Simulate trades with virtual positions and track theoretical PnL
- **Safety Guardrails**:
  - Position limits
  - Notional caps per market
  - Rate limiting
  - Kill switch
- **CLI Interface**: Easy-to-use commands for all operations
- **SQLite Storage**: Local database for markets, signals, and trades

## Installation (Windows)

### Prerequisites

- Python 3.12+
- [Poetry](https://python-poetry.org/docs/#installation) package manager

### Setup

```powershell
# Clone the repository
git clone https://github.com/yosiwizman/polymarket-quant-lab.git
cd polymarket-quant-lab

# Install dependencies
poetry install

# Verify installation
poetry run pmq --version
```

## CLI Usage

### Sync Market Data

Fetch and cache market data from the Polymarket Gamma API:

```powershell
# Fetch top 200 markets by volume
poetry run pmq sync

# Fetch more markets
poetry run pmq sync --limit 500

# Clear cache before syncing
poetry run pmq sync --clear-cache
```

### Scan for Signals

Analyze markets for trading opportunities:

```powershell
# Scan for arbitrage signals
poetry run pmq scan

# Show top 10 signals with custom threshold
poetry run pmq scan --top 10 --arb-threshold 0.98

# Fetch fresh data from API instead of using cache
poetry run pmq scan --from-api

# Set minimum liquidity requirement
poetry run pmq scan --min-liquidity 500
```

### Paper Trading

Run the paper trading strategy loop:

```powershell
# Run paper trading for 5 minutes
poetry run pmq paper run --minutes 5

# Customize trading parameters
poetry run pmq paper run --minutes 10 --quantity 20 --interval 60

# Dry run (detect signals but don't execute trades)
poetry run pmq paper run --dry-run

# View current positions
poetry run pmq paper positions

# View recent trades
poetry run pmq paper trades --limit 50
```

### Generate Reports

View trading performance summary:

```powershell
poetry run pmq report
```

### Operator Console (Phase 1.5)

Start the local web dashboard:

```powershell
# Start local-only dashboard (no auth required)
poetry run pmq serve

# Custom host/port
poetry run pmq serve --host 127.0.0.1 --port 9000
```

Then open http://localhost:8080 in your browser. The dashboard shows:
- Trading statistics (trades, positions, PnL)
- Runtime state (last scan, errors, etc.)
- Recent signals and trades
- Open positions

### Operator Loop

Run a long-running operator loop with automatic error recovery:

```powershell
# Run continuously (Ctrl+C to stop)
poetry run pmq run

# Custom interval (seconds between cycles)
poetry run pmq run --interval 120

# Limit number of cycles
poetry run pmq run --cycles 10

# Customize market limit and trade quantity
poetry run pmq run --limit 500 --quantity 25
```

Features:
- Exponential backoff on errors (5s → 10s → ... → 300s)
- Automatic recovery after transient failures
- Runtime state logged to database (viewable in dashboard)
- Graceful shutdown on Ctrl+C

### Export Data

Export trading data to CSV files:

```powershell
# Export all data (signals, trades, positions)
poetry run pmq export all

# Export specific data types
poetry run pmq export signals
poetry run pmq export trades
poetry run pmq export positions

# Custom output directory
poetry run pmq export all --out ./exports

# Filter signals by type
poetry run pmq export signals --type ARBITRAGE
poetry run pmq export signals --type STAT_ARB

# Limit number of records
poetry run pmq export signals --limit 100
```

### Backtesting (Phase 2)

Run deterministic backtests on historical data.

#### Collecting Snapshots

First, collect market snapshots for backtesting:

```powershell
# Single snapshot (manual)
poetry run pmq sync --snapshot

# Automated collection with scheduler (Phase 2.5)
poetry run pmq snapshots run --interval 60 --limit 200 --duration-minutes 60

# Run indefinitely (Ctrl+C to stop)
poetry run pmq snapshots run --interval 60 --duration-minutes 0

# With order book microstructure data (Phase 4.9, default: ON)
poetry run pmq snapshots run --interval 60 --with-orderbook

# Disable order book fetching for faster collection
poetry run pmq snapshots run --interval 60 --no-orderbook
```

The scheduler:
- Collects snapshots at regular intervals
- Uses exponential backoff on API errors
- Respects `PMQ_SNAPSHOT_KILL=true` environment variable
- Does NOT execute any trades - data capture only

#### Running Backtests

```powershell
# Run arbitrage backtest
poetry run pmq backtest run --strategy arb --from 2024-01-01 --to 2024-01-07

# With custom parameters
poetry run pmq backtest run --strategy arb --from 2024-01-01 --to 2024-01-07 --balance 5000 --quantity 20

# Run stat-arb backtest with pairs config
poetry run pmq backtest run --strategy statarb --pairs config/pairs.yml --from 2024-01-01 --to 2024-01-07
```

#### Viewing Results

```powershell
# List recent backtest runs
poetry run pmq backtest list

# View detailed report for a run
poetry run pmq backtest report --run-id <run-id>

# Show individual trades
poetry run pmq backtest report --run-id <run-id> --trades

# Export results
poetry run pmq backtest export --run-id <run-id> --format csv --out exports/
poetry run pmq backtest export --run-id <run-id> --format json --out exports/
```

#### Backtest Metrics

Backtests compute the following metrics:
- **Total PnL**: Net profit/loss from the backtest period
- **Max Drawdown**: Maximum peak-to-trough decline as a percentage
- **Win Rate**: Percentage of profitable arb opportunities
- **Sharpe Ratio**: Risk-adjusted return (simplified, no risk-free rate)
- **Trades/Day**: Average number of trades executed per day
- **Capital Utilization**: Average percentage of capital deployed

> ⚠️ **Warning**: Backtest results are not guarantees of future performance. Historical data may not reflect future market conditions.

#### Snapshot Quality & Coverage (Phase 2.5/3.2)

Check snapshot data quality before running backtests:

```powershell
# Check overall snapshot stats
poetry run pmq snapshots summary

# Quality check with explicit time window
poetry run pmq snapshots quality --from 2024-01-01 --to 2024-01-07 --interval 60

# Rolling window: last N minutes (evaluates recent data quality)
poetry run pmq snapshots quality --last-minutes 60 --interval 60

# Last K distinct snapshot times (RECOMMENDED for new data)
# Avoids penalizing for historical gaps
poetry run pmq snapshots quality --last-times 30 --interval 60

# View coverage by market
poetry run pmq snapshots coverage --from 2024-01-01 --to 2024-01-07
```

**Recommended Workflow for New Data Collection:**

1. Start snapshot collection: `pmq snapshots run --interval 60 --duration-minutes 60`
2. Check recent quality: `pmq snapshots quality --last-times 30 --interval 60`
3. Wait until `ready_for_scorecard: Yes` before running backtests
4. Run backtest when maturity score ≥ 70

Quality metrics:
- **Coverage %**: Percentage of expected intervals with data
- **Distinct Times**: Actual unique snapshot timestamps captured
- **Missing Intervals**: Gaps larger than 50% of expected interval
- **Duplicates**: Same market+timestamp entries
- **Maturity Score**: 0-100 readiness indicator (need 70+ for scorecard)
- **Ready for Scorecard**: Boolean - whether data is mature enough for evaluation
- **Status Badge**: healthy (≥95% coverage + ready), degraded (≥80%), unhealthy (<80%)

#### Backtest Manifests

Every backtest run automatically saves a manifest for reproducibility:

- **Config Hash**: SHA256 of all config parameters
- **Git SHA**: Commit hash at run time
- **Snapshot Range**: First/last snapshot times used
- **Market Filter**: List of market IDs (for stat-arb)

Manifests ensure backtests can be reproduced exactly by recording all inputs.

### Strategy Approval & Risk Governance (Phase 3)

Before a strategy can run paper trading, it must be formally approved based on backtest results.

#### Approval Workflow

```powershell
# 1. Run a backtest
poetry run pmq backtest run --strategy arb --from 2024-01-01 --to 2024-01-31

# 2. Evaluate the backtest for approval (shows scorecard)
poetry run pmq approve evaluate --run-id <run-id>

# 3. If scorecard passes, grant approval
poetry run pmq approve grant --run-id <run-id> --name arb --version v1

# 4. Now paper trading requires approval (or --override-unsafe)
poetry run pmq paper run --strategy arb --minutes 10
```

#### Scorecard Criteria

The scorecard evaluates backtests on a 0-100 scale:

| Metric | Weight | Minimum Threshold |
|--------|--------|-------------------|
| PnL | 25 pts | ≥ 0 (non-negative) |
| Max Drawdown | 20 pts | ≤ 25% |
| Win Rate | 15 pts | ≥ 40% |
| Sharpe Ratio | 20 pts | ≥ 0.5 |
| Trades/Day | 10 pts | ≥ 5 trades total |
| Data Quality | 10 pts | ≥ 70% coverage |

A strategy must:
1. Pass all minimum thresholds (no FAILs)
2. Achieve a total score ≥ 60/100

#### Risk Limits

Approved strategies receive recommended risk limits based on performance:

- **Max Notional/Market**: Per-market position limit
- **Max Total Notional**: Total exposure limit
- **Max Positions**: Maximum concurrent positions
- **Max Trades/Hour**: Rate limiting
- **Stop Loss**: Drawdown-based kill trigger

#### Approval Management

```powershell
# List all approvals
poetry run pmq approve list

# Filter by status
poetry run pmq approve list --status APPROVED

# Revoke an approval
poetry run pmq approve revoke --approval-id 1 --reason "Performance degraded"

# View risk events
poetry run pmq approve risk-events
poetry run pmq approve risk-events --severity CRITICAL
```

#### Override Mode

For testing, you can bypass approval checks:

```powershell
# NOT recommended for production
poetry run pmq paper run --strategy arb --override-unsafe
```

This logs a WARN-level risk event but allows execution.

### Evaluation Pipeline (Phase 4)

The evaluation pipeline automates end-to-end strategy validation with a single command:

1. **Quality Check**: Assert data is READY (maturity ≥ 70)
2. **Backtest**: Run deterministic backtest on quality window
3. **Approval**: Evaluate scorecard for go/no-go decision
4. **Paper Run** (optional): Short smoke test
5. **Report**: Generate deterministic report

#### Running an Evaluation

```powershell
# Basic evaluation (requires 30 recent snapshots)
poetry run pmq eval run --strategy arb --version v1 --last-times 30

# With paper trading smoke test
poetry run pmq eval run --strategy arb --version v1 --last-times 30 --paper-minutes 10

# Custom parameters
poetry run pmq eval run --strategy arb --version v1 --last-times 60 --interval 60 --quantity 20 --balance 5000

# Observer strategy (validation mode)
poetry run pmq eval run --strategy observer --version v1 --last-times 30
```

#### Viewing Evaluations

```powershell
# List recent evaluations
poetry run pmq eval list

# Filter by status
poetry run pmq eval list --status PASSED
poetry run pmq eval list --strategy arb

# View detailed report
poetry run pmq eval report --id <eval-id>

# Show artifact details
poetry run pmq eval report --id <eval-id> --artifacts
```

#### Exporting Reports

```powershell
# Export as Markdown
poetry run pmq eval export --id <eval-id> --format md

# Export as JSON
poetry run pmq eval export --id <eval-id> --format json --out reports/

# Export as CSV
poetry run pmq eval export --id <eval-id> --format csv
```

#### Recommended Operator Workflow

```powershell
# 1. Start snapshot collection (Phase 2.5)
poetry run pmq snapshots run --interval 60 --duration-minutes 0  # Run indefinitely

# 2. Wait ~30 minutes for data maturity, then check quality
poetry run pmq snapshots quality --last-times 30 --interval 60

# 3. When READY, run full evaluation
poetry run pmq eval run --strategy arb --version v1 --last-times 30

# 4. If PASSED, proceed to paper trading
poetry run pmq paper run --strategy arb --minutes 60

# 5. Monitor via dashboard
poetry run pmq serve
```

#### Evaluation Artifacts

Each evaluation saves artifacts for reproducibility:

- **QUALITY_JSON**: Quality check results (coverage, maturity, window)
- **BACKTEST_JSON**: Backtest metrics and run ID
- **SCORECARD_TXT**: Human-readable scorecard
- **PAPER_LOG**: Paper trading results (if enabled)
- **REPORT_MD**: Full markdown report
- **REPORT_JSON**: Machine-readable report

#### API Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /api/evals` | List evaluation runs |
| `GET /api/evals/{id}` | Get evaluation details |
| `GET /api/evals/{id}/artifacts/{kind}` | Get specific artifact |
| `GET /api/evals/summary` | Status counts and latest by strategy |

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PMQ_SAFETY_KILL_SWITCH` | `false` | Emergency stop all operations |
| `PMQ_SAFETY_MAX_POSITIONS` | `100` | Maximum open positions |
| `PMQ_SAFETY_MAX_NOTIONAL_PER_MARKET` | `1000` | Max notional per market (USD) |
| `PMQ_SAFETY_MAX_TRADES_PER_HOUR` | `50` | Rate limit on trades |
| `PMQ_ARB_THRESHOLD` | `0.99` | Arbitrage signal threshold |
| `PMQ_ARB_MIN_LIQUIDITY` | `100` | Minimum liquidity for signals |
| `PMQ_STATARB_ENTRY_THRESHOLD` | `0.10` | Stat-arb entry threshold |
| `PMQ_STATARB_EXIT_THRESHOLD` | `0.02` | Stat-arb exit threshold |

### Stat-Arb Pairs Configuration (Phase 4.1)

Statistical arbitrage requires a pairs configuration file that defines which markets to monitor.

#### Generating Pairs

```powershell
# Generate pairs from captured snapshot data (DB-only, no live API calls)
poetry run pmq statarb pairs suggest --last-times 30 --interval 60 --out config/pairs.yml

# Generate with more pairs
poetry run pmq statarb pairs suggest --last-times 30 --top 50 --out config/pairs.yml
```

#### Validating Pairs

```powershell
# Validate an existing pairs config
poetry run pmq statarb pairs validate --pairs config/pairs.yml
```

#### Pairs Config Schema

```yaml
pairs:
  - market_a_id: "123456"          # Required: Market ID (numeric or hex)
    market_b_id: "789012"          # Required: Second market ID
    name: "My Pair"                # Optional: Human-readable name
    correlation: 1.0               # Optional: 1.0 = same direction, -1.0 = inverse
    enabled: true                  # Optional: false to disable pair
    min_liquidity: 100.0           # Optional: Minimum liquidity threshold
    max_spread: 0.05               # Optional: Maximum spread to consider
```

#### Running StatArb Evaluation

```powershell
# StatArb evaluation requires --pairs flag
poetry run pmq eval run --strategy statarb --version v1 --last-times 30 --pairs config/pairs.yml

# With paper trading smoke test
poetry run pmq eval run --strategy statarb --version v1 --pairs config/pairs.yml --paper-minutes 10
```

#### Debugging "0 Trades" Issues

If statarb produces 0 trades, use the explain command:

```powershell
# Analyze why signals weren't generated
poetry run pmq statarb explain --from 2024-12-30 --to 2024-12-30 --pairs config/pairs.yml
```

The explain command shows:
- Number of snapshots in the time window
- Per-pair analysis: coverage, spread statistics, signal counts
- Skip reasons (missing data, no overlapping times, etc.)

**Common causes of 0 trades:**

1. **Entry threshold too high**: Default is 0.10 (10%). Lower it with `PMQ_STATARB_ENTRY_THRESHOLD=0.05`
2. **Missing snapshot data**: Markets in pairs config not captured in snapshots
3. **No overlapping times**: Markets have snapshots at different times
4. **Pairs not enabled**: Check `enabled: true` in pairs config
5. **Markets inactive/closed**: Skipped automatically

#### End-to-End StatArb Workflow

```powershell
# 1. Start snapshot collection
poetry run pmq snapshots run --interval 60 --duration-minutes 60

# 2. Check data quality
poetry run pmq snapshots quality --last-times 30 --interval 60

# 3. Generate pairs from captured data
poetry run pmq statarb pairs suggest --last-times 30 --out config/pairs.yml

# 4. Validate and review pairs
poetry run pmq statarb pairs validate --pairs config/pairs.yml

# 5. Debug potential signals
poetry run pmq statarb explain --from 2024-12-30 --to 2024-12-30 --pairs config/pairs.yml

# 6. Run evaluation
poetry run pmq eval run --strategy statarb --version v1 --pairs config/pairs.yml --last-times 30
```

### StatArb Pair Discovery (Phase 4.2)

For more sophisticated pair discovery using correlation analysis:

```powershell
# Discover pairs using correlation of YES prices (deterministic, no live API calls)
poetry run pmq statarb discover --from 2024-12-01 --to 2024-12-30 --top 20

# Save discovered pairs to config file
poetry run pmq statarb discover --from 2024-12-01 --to 2024-12-30 --out config/pairs.yml

# Customize discovery parameters
poetry run pmq statarb discover --from 2024-12-01 --to 2024-12-30 \
  --top 50 \
  --min-overlap 10 \
  --min-corr 0.5 \
  --out config/pairs.yml
```

#### Validate Pairs Against Snapshot Data

```powershell
# Check that pairs have sufficient overlap in a date range
poetry run pmq statarb validate --from 2024-12-01 --to 2024-12-30 --pairs config/pairs.yml

# With custom minimum overlap requirement
poetry run pmq statarb validate --from 2024-12-01 --to 2024-12-30 --pairs config/pairs.yml --min-overlap 20
```

#### Discovery vs Suggest

| Command | Method | Best For |
|---------|--------|----------|
| `statarb pairs suggest` | Slug-based grouping | Quick pairs from related events |
| `statarb discover` | Correlation analysis | Finding statistically correlated pairs |

The `discover` command computes Pearson correlation between YES prices across overlapping snapshot times. Results are deterministic: same inputs always produce the same output.

> ⚠️ **PAPER ONLY**: All statarb trading is paper-only. No wallets, no live orders, no private keys.

### StatArb Walk-Forward + Tuning (Phase 4.3)

Phase 4.3 introduces research-grade tools for statarb:

1. **Walk-Forward Evaluation**: Split data into TRAIN/TEST to prevent overfitting
2. **Z-Score Mean Reversion**: OLS-fitted hedge ratios with z-score entry/exit signals
3. **Parameter Tuning**: Grid search to find optimal parameters

#### Z-Score Signal Model

The z-score strategy computes:
- **Beta (hedge ratio)**: OLS regression of price A on price B, fitted on TRAIN only
- **Spread**: priceA - beta × priceB
- **Z-score**: (spread - mean) / std, using TRAIN mean/std

Signal logic:
- **Enter Long** (long A, short B): when z ≤ -entry_z
- **Enter Short** (short A, long B): when z ≥ entry_z
- **Exit**: when |z| ≤ exit_z OR max_hold_bars reached

#### Walk-Forward Evaluation

```powershell
# Single walk-forward run with custom parameters
poetry run pmq statarb walkforward \
  --from 2024-12-01 --to 2024-12-30 \
  --pairs config/pairs.yml \
  --train-times 120 --test-times 60 \
  --entry-z 2.0 --exit-z 0.5
```

Output shows:
- **TRAIN Summary**: Fitted pairs, beta values, spread stats
- **TEST Summary**: Scorecard metrics (PnL, Sharpe, win rate) computed on out-of-sample data only

#### Parameter Tuning

```powershell
# Run grid search to find optimal parameters
poetry run pmq statarb tune \
  --from 2024-12-01 --to 2024-12-30 \
  --pairs config/pairs.yml \
  --train-times 120 --test-times 60 \
  --grid config/statarb_grid.yml \
  --out results/statarb_tuning.csv \
  --export-best config/statarb_best.yml

# Or use default grid
poetry run pmq statarb tune --from 2024-12-01 --to 2024-12-30 --pairs config/pairs.yml
```

Grid search:
- Tests all parameter combinations defined in `config/statarb_grid.yml`
- Ranks by Sharpe ratio on TEST data (prevents overfitting)
- Outputs leaderboard to CSV and best config to YAML

#### Grid Config Format

```yaml
# config/statarb_grid.yml
grid:
  lookback: [20, 30, 50]
  entry_z: [1.5, 2.0, 2.5]
  exit_z: [0.3, 0.5, 0.7]
  max_hold_bars: [30, 60, 120]
  cooldown_bars: [5]
  fee_bps: [0.0]
  slippage_bps: [0.0]
```

#### Z-Score Config Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PMQ_STATARB_LOOKBACK` | 30 | Rolling window for z-score |
| `PMQ_STATARB_ENTRY_Z` | 2.0 | Entry when \|z\| >= entry_z |
| `PMQ_STATARB_EXIT_Z` | 0.5 | Exit when \|z\| <= exit_z |
| `PMQ_STATARB_MAX_HOLD_BARS` | 60 | Max bars before forced exit |
| `PMQ_STATARB_COOLDOWN_BARS` | 5 | Bars to wait after exit |
| `PMQ_STATARB_FEE_BPS` | 0.0 | Fee in basis points |
| `PMQ_STATARB_SLIPPAGE_BPS` | 0.0 | Slippage in basis points |

#### Walk-Forward Research Workflow

```powershell
# 1. Collect snapshot data
poetry run pmq snapshots run --interval 60 --duration-minutes 180

# 2. Discover correlated pairs
poetry run pmq statarb discover --from 2024-12-01 --to 2024-12-30 --out config/pairs.yml

# 3. Tune parameters (grid search)
poetry run pmq statarb tune --from 2024-12-01 --to 2024-12-30 \
  --pairs config/pairs.yml \
  --export-best config/statarb_best.yml

# 4. Review best parameters and run walk-forward with them
poetry run pmq statarb walkforward --from 2024-12-01 --to 2024-12-30 \
  --pairs config/pairs.yml \
  --entry-z 2.0 --exit-z 0.5 --max-hold 60

# 5. If scorecard metrics are acceptable, run eval pipeline (walk-forward)
poetry run pmq eval run --strategy statarb --version zscore-v1 \
  --pairs config/pairs.yml --walkforward
```

> 📊 **Research Best Practice**: The scorecard still uses governance gates (PnL, Sharpe, win rate thresholds). Tune parameters to PASS the scorecard, don't bypass it.

### Eval Pipeline Walk-Forward (Phase 4.4)

Phase 4.4 integrates walk-forward evaluation into `pmq eval run` for statarb, ensuring the scorecard sees TEST-only metrics (no data leakage).

#### Walk-Forward Auto-Detection

Walk-forward is automatically enabled for statarb when:
- `--walkforward` flag is set explicitly
- `--statarb-params` is provided
- Strategy version contains "zscore" or "walkforward"
- `config/statarb_best.yml` exists (from tuning)

#### New CLI Flags

```powershell
# Explicit walk-forward evaluation
poetry run pmq eval run --strategy statarb --version zscore-v1 \
  --pairs config/pairs.yml --walkforward --train-times 100 --test-times 50

# Use tuned parameters from grid search
poetry run pmq eval run --strategy statarb --version v1 \
  --pairs config/pairs.yml --statarb-params config/statarb_best.yml

# Disable walk-forward (use legacy backtest)
poetry run pmq eval run --strategy statarb --version v1 \
  --pairs config/pairs.yml --no-walkforward
```

#### Walk-Forward Eval Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--walkforward/--no-walkforward` | auto-detect | Enable/disable walk-forward |
| `--statarb-params` | config/statarb_best.yml | Path to z-score params YAML |
| `--train-times` | 100 | Snapshots for TRAIN segment |
| `--test-times` | 50 | Snapshots for TEST segment |

#### Walk-Forward Eval Output

When walk-forward is enabled, the output shows:
- **Mode**: Walk-Forward (TEST only)
- **TRAIN/TEST Windows**: Chronological time ranges
- **Fitted Pairs**: X/Y pairs successfully fitted
- **TEST Metrics**: PnL, Sharpe, Win Rate, Trades (used for scorecard)

```
Evaluation Results
┏━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Step                ┃ Result                        ┃
┡━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ Mode                │ Walk-Forward (TEST only)      │
│ TRAIN Window        │ 2025-01-01T00:00 → T01:40     │
│ TEST Window         │ 2025-01-01T01:41 → T02:30     │
│ Fitted Pairs        │ 15/20                         │
│ TEST PnL            │ $0.06                         │
│ TEST Sharpe         │ 0.110                         │
│ TEST Win Rate       │ 11.1%                         │
│ TEST Trades         │ 9                             │
│ Backtest Score      │ 34.0/100                      │
│ Approval            │ FAILED                        │
└─────────────────────┴───────────────────────────────┘

Note: Scorecard evaluated on TEST only (walk-forward, no data leakage)
```

#### Complete Walk-Forward Workflow

```powershell
# 1. Collect sufficient snapshot data (train + test)
poetry run pmq snapshots run --interval 60 --duration-minutes 300

# 2. Check data quality
poetry run pmq snapshots quality --last-times 200

# 3. Discover correlated pairs
poetry run pmq statarb discover --from 2024-12-01 --to 2024-12-30 \
  --out config/pairs.yml

# 4. (Optional) Tune parameters
poetry run pmq statarb tune --from 2024-12-01 --to 2024-12-30 \
  --pairs config/pairs.yml --export-best config/statarb_best.yml

# 5. Run evaluation with walk-forward (auto-detects best.yml)
poetry run pmq eval run --strategy statarb --version zscore-v1 \
  --pairs config/pairs.yml --last-times 200

# 6. Export detailed report
poetry run pmq eval export --id <eval-id> --format md
```

> ⚠️ **Why Walk-Forward?** Traditional backtests fit and evaluate on the same data, risking overfitting. Walk-forward splits data chronologically: TRAIN first, TEST second. Parameters are fitted only on TRAIN, and the scorecard sees only TEST metrics—this is the industry standard for research-grade evaluation.

### Gap-Aware Contiguous Windows (Phase 4.5)

Phase 4.5 introduces gap-aware "contiguous" data windows that prevent old session gaps from penalizing recent healthy data.

#### The Problem

When the database contains old snapshot sessions with gaps between them (e.g., from multiple collection sessions), the default `--last-times` window selection spans those gaps. This causes:
- Data quality coverage to drop artificially
- Maturity score penalized even when recent data is healthy
- Eval scorecard sees degraded quality metrics

#### The Solution

Contiguous mode (`--contiguous`) stops at gaps when selecting snapshot times, returning only the most recent contiguous block of data.

```powershell
# Default behavior: contiguous=True for last-times mode
poetry run pmq snapshots quality --last-times 200 --interval 60

# Explicit contiguous mode
poetry run pmq snapshots quality --last-times 200 --interval 60 --contiguous

# Disable contiguous (analyze all times, including across gaps)
poetry run pmq snapshots quality --last-times 200 --interval 60 --no-contiguous
```

#### How It Works

1. Fetches candidate snapshot times (descending from newest)
2. Walks backwards from the newest time
3. Stops when gap between consecutive times exceeds `interval × 2.5` (gap factor)
4. Returns only the contiguous block

#### Contiguous Mode Output

When a gap is detected:

```
Quality Metrics
┏━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Metric                ┃ Value                                          ┃
┡━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ Window Mode           │ last_times                                     │
│ Window                │ Last 200 snapshot times (contiguous)           │
│ Actual Range          │ 2025-01-01T12:00:00 to 2025-01-01T14:00:00     │
│ Contiguous Mode       │ Yes                                            │
│ Gap Cutoff            │ 2025-01-01T10:00:00                            │
│ Times Used            │ 93 of 200 available                            │
│ Coverage              │ 93.0%                                          │
│ Maturity Score        │ 85/100                                         │
└───────────────────────┴────────────────────────────────────────────────┘
```

#### Eval Pipeline Alignment

The evaluation pipeline automatically uses contiguous mode for walk-forward evaluation:
- Requests `train_times + test_times` in contiguous mode
- If fewer times available than requested, scales train/test proportionally
- Quality metrics and walk-forward use the SAME contiguous window
- Reports show contiguous info (requested vs actual times, gap cutoff)

```powershell
# Walk-forward with contiguous windows (default)
poetry run pmq eval run --strategy statarb --version zscore-v1 \
  --pairs config/pairs.yml --walkforward --train-times 100 --test-times 50
```

#### Recommended Workflow

```powershell
# 1. Check data quality with contiguous mode (see actual usable window)
poetry run pmq snapshots quality --last-times 200 --interval 60 --contiguous

# 2. If gap detected, either:
#    a) Collect more recent data to fill the contiguous window
#    b) Reduce train/test requirements to fit available data

# 3. Run eval - it will automatically use contiguous windows
poetry run pmq eval run --strategy statarb --version zscore-v1 \
  --pairs config/pairs.yml --last-times 200

# 4. Review report - shows contiguous info and any scaling applied
poetry run pmq eval export --id <eval-id> --format md
```

> 💡 **Why Contiguous?** Old snapshot sessions with gaps shouldn't penalize recent healthy data. Contiguous mode ensures quality metrics reflect the actual usable data window, preventing false "NOT READY" results.

### Realistic Costs + Market Constraints (Phase 4.6)

Phase 4.6 adds realistic transaction cost modeling and market constraint filtering to prevent overly optimistic evaluation results.

#### Cost Model

All evaluation commands now use realistic cost defaults:
- **fee_bps**: 2.0 bps (approximates Polymarket maker/taker fees)
- **slippage_bps**: 5.0 bps (accounts for market impact)

```powershell
# Tuning with default costs (fee_bps=2.0, slippage_bps=5.0)
poetry run pmq statarb tune --from 2024-12-01 --to 2024-12-30 --pairs config/pairs.yml

# Override costs for sensitivity analysis
poetry run pmq statarb tune --from 2024-12-01 --to 2024-12-30 --pairs config/pairs.yml \
  --fee-bps 5.0 --slippage-bps 10.0

# Walk-forward with custom costs
poetry run pmq statarb walkforward --from 2024-12-01 --to 2024-12-30 \
  --pairs config/pairs.yml --fee-bps 3.0 --slippage-bps 8.0
```

#### Cost Impact

Costs are applied consistently:
1. **Entry**: Pay `(fee_bps + slippage_bps) × notional` on position entry
2. **Exit**: Pay `(fee_bps + slippage_bps) × notional` on position exit
3. **Net PnL**: Raw PnL minus total costs

Reports now show cost assumptions:

```
### Cost Assumptions
- Fee: 2.0 bps
- Slippage: 5.0 bps
```

#### Market Constraints

Pairs can define optional liquidity and spread constraints in `pairs.yml`:

```yaml
pairs:
  - market_a_id: "0x..."
    market_b_id: "0x..."
    name: "ETH_YES/BTC_YES"
    enabled: true
    min_liquidity: 1000.0    # Filter if avg liquidity < 1000
    max_spread: 0.02         # Filter if avg spread > 2%
```

During walk-forward evaluation:
- Pairs violating constraints are filtered BEFORE fitting
- Filter reasons are logged and included in results
- Reports show `eligible_pairs / total_pairs` counts

#### CLI Flags

| Command | Flag | Default | Description |
|---------|------|---------|-------------|
| `statarb tune` | `--fee-bps` | 2.0 | Fee in basis points |
| `statarb tune` | `--slippage-bps` | 5.0 | Slippage in basis points |
| `statarb walkforward` | `--fee-bps` | 2.0 | Fee in basis points |
| `statarb walkforward` | `--slippage-bps` | 5.0 | Slippage in basis points |

#### Example: Cost Sensitivity

```powershell
# Compare no-cost vs realistic costs
poetry run pmq statarb walkforward --from 2024-12-01 --to 2024-12-30 \
  --pairs config/pairs.yml --fee-bps 0 --slippage-bps 0  # Optimistic

poetry run pmq statarb walkforward --from 2024-12-01 --to 2024-12-30 \
  --pairs config/pairs.yml --fee-bps 2 --slippage-bps 5  # Realistic (default)

poetry run pmq statarb walkforward --from 2024-12-01 --to 2024-12-30 \
  --pairs config/pairs.yml --fee-bps 5 --slippage-bps 10 # Conservative
```

> ⚠️ **Why Costs Matter:** Zero-cost backtests are unrealistically optimistic. Real trading incurs fees, spreads, and slippage. Phase 4.6 defaults ensure evaluation results are closer to live performance.

### Quality Window Alignment with Walk-Forward (Phase 4.7)

Phase 4.7 fixes "false-negative" Data Quality failures by aligning quality scoring with the actual contiguous evaluation window (TRAIN+TEST) used by walk-forward evaluation.

#### The Problem

Previously, the quality check computed coverage against the *requested* window size, not the *effective* window actually used by walk-forward:

1. User requests `--train-times 100 --test-times 50` (150 total)
2. Only 93 contiguous times are available due to gaps
3. Walk-forward scales down to use 62 train + 31 test = 93 times
4. Quality was evaluated against 150 expected → 62% coverage → FAIL
5. But the actual evaluation used all 93 available times → should be 100%

This caused false Data Quality failures when data was actually sufficient.

#### The Solution

After walk-forward determines its effective window (train_from to test_to), the pipeline re-evaluates quality on that exact window:

```
# Before Phase 4.7
Quality: 62/150 expected = 41.3% → FAIL (false negative)

# After Phase 4.7  
Effective Quality: 93/93 expected = 100% → PASS
```

#### Effective Window Quality in Reports

Evaluation reports now include an "Effective Window Quality" section when walk-forward is used:

```markdown
### Effective Window Quality (Aligned with Walk-Forward)

- **Effective Window:** 2025-01-01T00:00:00 → 2025-01-01T01:33:00
- **Expected Points:** 94
- **Observed Points:** 93
- **Quality Pct:** 98.9%

*Quality re-evaluated on the exact window used by walk-forward (TRAIN+TEST)*
```

#### How It Works

1. Initial quality check runs with requested window size
2. Walk-forward scales train/test if fewer times available
3. Walk-forward returns effective window boundaries (`train_from` to `test_to`)
4. Quality is re-checked on the effective window using `check_explicit_window()`
5. Scorecard uses the effective quality percentage for approval
6. Reports surface both requested and effective quality metrics

#### New Fields in EvaluationResult

| Field | Type | Description |
|-------|------|-------------|
| `effective_window_from` | str | Start of effective window (train start) |
| `effective_window_to` | str | End of effective window (test end) |
| `effective_expected_points` | int | Expected points for effective window |
| `effective_observed_points` | int | Actual distinct times in effective window |
| `effective_quality_pct` | float | Coverage % for effective window |
| `quality_window_aligned` | bool | True when quality was re-checked on effective window |

#### CLI: Explicit Window Quality

The `snapshots quality` command now uses proper expected/observed computation for explicit windows:

```powershell
# Check quality for a specific time range
poetry run pmq snapshots quality --from 2024-12-01T00:00:00 --to 2024-12-01T01:00:00 --interval 60

# Output shows expected vs observed points
# Expected: floor((end-start)/interval)+1 = 61 points
# Observed: actual distinct snapshot times = 58 points  
# Coverage: 58/61 = 95.1%
```

> 💡 **Why Alignment Matters:** Governance should evaluate quality on the same window that trades are evaluated on. Misaligned windows cause false negatives (rejecting good strategies) or false positives (approving strategies with data issues).

### Eval Pipeline Realism: Costs + Constraints (Phase 4.8)

Phase 4.8 makes `pmq eval run` for StatArb fully "realistic" by propagating cost and constraint parameters into the evaluation pipeline, with transparent reporting.

#### CLI Flags for Realism

New flags on `pmq eval run` allow overriding costs and constraints at evaluation time:

```powershell
# Evaluate with custom costs (override YAML and defaults)
poetry run pmq eval run --strategy statarb --version zscore-v1 \
  --pairs config/pairs.yml --walkforward \
  --fee-bps 3.0 --slippage-bps 8.0

# Evaluate with global constraint overrides
poetry run pmq eval run --strategy statarb --version zscore-v1 \
  --pairs config/pairs.yml --walkforward \
  --min-liquidity 500 --max-spread 0.03

# Full realism: costs + constraints
poetry run pmq eval run --strategy statarb --version zscore-v1 \
  --pairs config/pairs.yml --walkforward \
  --fee-bps 5.0 --slippage-bps 10.0 \
  --min-liquidity 1000 --max-spread 0.02
```

#### Precedence Rules

Costs and constraints follow a clear precedence hierarchy:

1. **CLI flags** override everything (global overrides)
2. **Params YAML** values are used if CLI flag not set (`--statarb-params config/statarb_best.yml`)
3. **Project defaults** used if neither CLI nor YAML set:
   - `fee_bps`: 2.0 (Polymarket maker/taker fees)
   - `slippage_bps`: 5.0 (market impact)
   - Constraints: unset unless provided per-pair in pairs.yml

```powershell
# Precedence example: CLI (3.0) > YAML (5.0) > Default (2.0)
poetry run pmq eval run --strategy statarb --version zscore-v1 \
  --pairs config/pairs.yml --statarb-params config/best.yml \
  --fee-bps 3.0  # Uses 3.0 from CLI, ignores YAML's 5.0
```

#### Constraint Filtering

Global constraint overrides (`--min-liquidity`, `--max-spread`) filter pairs before walk-forward fitting:

1. Load all pairs from `pairs.yml`
2. Apply global constraint overrides (replace per-pair constraints)
3. Filter pairs that don't meet liquidity/spread requirements
4. Run walk-forward on eligible pairs only
5. Report filtering statistics

#### Report Sections

Evaluation reports now include:

**Cost Assumptions:**
```markdown
### Cost Assumptions
- **Fee:** 2.0 bps
- **Slippage:** 5.0 bps
- **Total Round-Trip Cost:** 7.0 bps
```

**Constraint Filtering:**
```markdown
### Constraint Filtering
- **Pairs Before Filtering:** 10
- **Pairs After Filtering:** 7
- **Filtered (Low Liquidity):** 2
- **Filtered (High Spread):** 1
- **Global Min Liquidity:** 500.0
- **Global Max Spread:** 0.03
```

#### New CLI Flags

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--fee-bps` | float | None | Fee override (bps) |
| `--slippage-bps` | float | None | Slippage override (bps) |
| `--min-liquidity` | float | None | Global min liquidity filter |
| `--max-spread` | float | None | Global max spread filter |

#### New EvaluationResult Fields

| Field | Type | Description |
|-------|------|-------------|
| `fee_bps` | float | Effective fee used (bps) |
| `slippage_bps` | float | Effective slippage used (bps) |
| `constraints_applied` | bool | True if any constraints filtered pairs |
| `constraint_min_liquidity` | float | Global min liquidity override (if set) |
| `constraint_max_spread` | float | Global max spread override (if set) |
| `pairs_before_constraints` | int | Total pairs before filtering |
| `pairs_after_constraints` | int | Pairs after constraint filtering |
| `pairs_filtered_low_liquidity` | int | Pairs filtered for low liquidity |
| `pairs_filtered_high_spread` | int | Pairs filtered for high spread |

#### Example Workflow

```powershell
# 1. Tune parameters with realistic costs
poetry run pmq statarb tune --from 2024-12-01 --to 2024-12-30 \
  --pairs config/pairs.yml --fee-bps 2.0 --slippage-bps 5.0 \
  --export-best config/statarb_best.yml

# 2. Evaluate with same costs (auto-loaded from best.yml)
poetry run pmq eval run --strategy statarb --version zscore-v1 \
  --pairs config/pairs.yml --statarb-params config/statarb_best.yml

# 3. Sensitivity analysis: higher costs
poetry run pmq eval run --strategy statarb --version zscore-v1 \
  --pairs config/pairs.yml --fee-bps 5.0 --slippage-bps 10.0

# 4. Add constraint filtering
poetry run pmq eval run --strategy statarb --version zscore-v1 \
  --pairs config/pairs.yml --min-liquidity 1000 --max-spread 0.02
```

> 💡 **Why Realism Matters:** Evaluation should use the same cost assumptions that will apply in live trading. Phase 4.8 ensures costs and constraints are explicit, configurable, and transparently reported.

### Microstructure Truth: Real Order Book Data (Phase 4.9)

Phase 4.9 enriches snapshots and evaluation with REAL bid/ask spread + top-of-book liquidity from Polymarket's public CLOB API, so constraints are based on actual order books rather than heuristics.

#### What's Captured

When `--with-orderbook` is enabled (default), snapshot collection fetches order book data:

| Field | Description |
|-------|-------------|
| `best_bid` | Highest bid price from order book |
| `best_ask` | Lowest ask price from order book |
| `mid_price` | (best_bid + best_ask) / 2 |
| `spread_bps` | ((best_ask - best_bid) / mid_price) × 10,000 |
| `top_depth_usd` | min(bid_notional, ask_notional) at top of book |

#### How It Works

1. **Snapshot Collection**: Each market snapshot fetches order book from CLOB public endpoint
2. **Microstructure Computation**: Computes spread_bps and top_depth_usd from raw order book
3. **Constraint Evaluation**: Uses real spread_bps/top_depth_usd when available, falls back to legacy heuristics otherwise
4. **Evaluation Reporting**: Shows microstructure coverage and median values

#### CLI Usage

```powershell
# Collect snapshots WITH order book data (default)
poetry run pmq snapshots run --interval 60 --with-orderbook

# Disable order book fetching (faster, less data)
poetry run pmq snapshots run --interval 60 --no-orderbook
```

#### Microstructure in Evaluation Reports

When microstructure data is available, evaluation reports include:

```markdown
### Microstructure
- **Snapshots with Order Book:** 80/100 (80.0%)
- **Median Spread:** 150.0 bps
- **Median Top-of-Book Depth:** $250.0
- **Pairs Using Real Spread:** 5
- **Pairs Using Real Depth:** 5
- **Pairs Missing Microstructure:** 2
```

#### Constraint Behavior

When evaluating pair constraints:

| Scenario | Spread Source | Liquidity Source |
|----------|---------------|------------------|
| Microstructure available | `spread_bps` from order book | `top_depth_usd` from order book |
| Microstructure missing | Legacy bid/ask spread heuristic | Legacy liquidity from Gamma API |

Constraint evaluation tracks which source was used and reports it.

#### Schema Changes

The `market_snapshots` table now includes (nullable, backward compatible):

```sql
best_bid REAL,
best_ask REAL,
mid_price REAL,
spread_bps REAL,
top_depth_usd REAL
```

Old databases are automatically migrated when opened.

#### API Endpoints Used

| Endpoint | Purpose |
|----------|----------|
| `https://clob.polymarket.com/book?token_id=...` | Fetch order book for a token |

> ⚠️ **Paper Only**: Phase 4.9 does NOT add any wallet auth, private keys, order placement, or WebSocket user channels. All data is from public read-only endpoints.

> 💡 **Why Microstructure?** Heuristic-based spread and liquidity estimates can be inaccurate. Real order book data ensures constraints filter pairs based on actual market conditions, leading to more realistic evaluation results.

### WebSocket Order Book Streaming (Phase 5.0)

Phase 5.0 adds real-time WebSocket streaming for order book data as an alternative to REST polling. This provides:
- Lower latency order book updates
- Reduced API request volume
- Automatic reconnection with exponential backoff
- Graceful fallback to REST when WebSocket data is stale

#### Order Book Source Selection

```powershell
# REST polling (default, Phase 4.9 behavior)
poetry run pmq snapshots run --interval 60 --orderbook-source rest

# WebSocket streaming with REST fallback
poetry run pmq snapshots run --interval 60 --orderbook-source wss

# Adjust staleness threshold (default: 30s)
poetry run pmq snapshots run --orderbook-source wss --wss-staleness 15
```

#### How It Works

1. **WSS Mode (`--orderbook-source wss`)**:
   - Opens single WebSocket connection to `wss://ws-subscriptions-clob.polymarket.com/ws/market`
   - Subscribes to order book updates for all tracked markets
   - Maintains in-memory cache of latest OrderBookData per token
   - On each snapshot cycle, reads from cache instead of REST polling
   - Falls back to REST if cache entry is stale (older than `--wss-staleness`)

2. **REST Mode (`--orderbook-source rest`, default)**:
   - Fetches order book via REST for each market on every cycle
   - Same behavior as Phase 4.9

#### WebSocket Statistics

When using WSS mode, the CLI reports coverage statistics:

```
Completed 10 cycles in 10 minutes
WSS coverage: 85.0% (170 hits, 30 REST fallbacks)
```

#### Reconnection Behavior

The WebSocket client implements production-grade resilience:
- **Exponential Backoff**: 1s → 2s → 4s → ... → 60s max
- **Jitter**: ±30% randomization to prevent thundering herd
- **Automatic Resubscription**: Re-subscribes to all assets after reconnect
- **Graceful Shutdown**: Clean disconnect on Ctrl+C

#### New CLI Flags

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--orderbook-source` | str | `rest` | Order book source: `rest` or `wss` |
| `--wss-staleness` | float | 30.0 | WSS cache staleness threshold in seconds |

#### MarketWssClient API

For programmatic use:

```python
from pmq.markets import MarketWssClient

client = MarketWssClient(staleness_seconds=30.0)
await client.connect()
await client.wait_connected(timeout=10.0)
await client.subscribe(["0xtoken1", "0xtoken2"])

# Read from cache (thread-safe)
ob = client.get_orderbook("0xtoken1")
if ob and ob.has_valid_book:
    print(f"Best bid: {ob.best_bid}, spread: {ob.spread_bps} bps")

# Check staleness
if client.is_stale("0xtoken1"):
    print("Cache is stale, use REST fallback")

# Get stats
stats = client.get_stats()
print(f"Messages: {stats.messages_received}, Reconnects: {stats.reconnect_count}")

await client.close()
```

> ⚠️ **Paper Only**: Phase 5.0 uses the public Market WebSocket channel only (no auth required). No wallet auth, private keys, or order placement is added. The User WebSocket channel (which requires auth) is not used.

> 💡 **When to Use WSS**: WebSocket mode is ideal for continuous collection sessions where lower latency and reduced API load are beneficial. REST mode is better for short collection runs or when network conditions are unreliable.

### Continuous Snapshot Capture Daemon (Phase 5.1)

Phase 5.1 provides a production-grade daemon for continuous market data capture with:
- Resilient WSS connection with automatic REST fallback
- Coverage tracking per tick (WSS hits, REST fallbacks, stale, missing)
- Daily export artifacts (CSV, JSON, markdown) on UTC rollover
- Clean shutdown on SIGINT/SIGTERM
- Injectable dependencies for testability

#### Basic Usage

```powershell
# Start daemon with defaults (60s interval, WSS mode, infinite runtime)
poetry run pmq ops daemon

# Run for 24 hours with custom interval
poetry run pmq ops daemon --interval 60 --max-hours 24

# Use REST-only mode (no WebSocket)
poetry run pmq ops daemon --orderbook-source rest

# Custom export directory
poetry run pmq ops daemon --export-dir ./data/exports
```

#### CLI Flags

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--interval` | int | 60 | Snapshot interval in seconds |
| `--limit` | int | 200 | Markets per cycle |
| `--orderbook-source` | str | `wss` | `rest` or `wss` |
| `--wss-staleness` | float | 30.0 | WSS staleness threshold (seconds) |
| `--max-hours` | float | None | Max runtime (infinite if not set) |
| `--export-dir` | Path | `exports` | Directory for daily exports |
| `--with-orderbook/--no-orderbook` | bool | True | Fetch order book data |

#### Daily Export Artifacts

The daemon automatically exports artifacts at UTC day rollover (and on shutdown):

| File | Description |
|------|-------------|
| `ticks_YYYY-MM-DD.csv` | Per-tick history with timestamps, counts, coverage |
| `coverage_YYYY-MM-DD.json` | Daily coverage statistics (WSS%, fallbacks, errors) |
| `daemon_summary_YYYY-MM-DD.md` | Human-readable daily summary |

#### Coverage JSON Format

```json
{
  "date": "2024-01-15",
  "total_ticks": 1440,
  "total_snapshots": 288000,
  "wss_hits": 250000,
  "rest_fallbacks": 30000,
  "stale_count": 5000,
  "missing_count": 3000,
  "errors": 2,
  "wss_coverage_pct": 89.3
}
```

#### Graceful Shutdown

The daemon handles SIGINT (Ctrl+C) and SIGTERM gracefully:
1. Stops accepting new ticks
2. Exports current day's artifacts
3. Closes WSS connection
4. Closes REST fetcher
5. Logs "Daemon stopped"

#### Production Deployment

**Linux (systemd):**

```ini
# /etc/systemd/system/pmq-daemon.service
[Unit]
Description=Polymarket Quant Lab Daemon
After=network.target

[Service]
Type=simple
User=pmq
WorkingDirectory=/opt/polymarket-quant-lab
ExecStart=/usr/bin/poetry run pmq ops daemon --interval 60 --export-dir /var/lib/pmq/exports
Restart=always
RestartSec=30

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl enable pmq-daemon
sudo systemctl start pmq-daemon
sudo journalctl -u pmq-daemon -f
```

**Windows (Task Scheduler):**

1. Create a new Basic Task in Task Scheduler
2. Trigger: At startup (or scheduled time)
3. Action: Start a program
4. Program: `poetry`
5. Arguments: `run pmq ops daemon --interval 60 --max-hours 24`
6. Start in: `C:\path\to\polymarket-quant-lab`

> 💡 **Daemon vs Snapshots Run**: Use `pmq ops daemon` for production continuous capture. Use `pmq snapshots run` for shorter ad-hoc collection sessions. The daemon provides better coverage tracking and daily exports.

### Snapshot Exports & Retention (Phase 5.2)

Phase 5.2 extends the ops daemon with production-ready features for long-running deployments:
- Daily snapshot exports to compressed gzip CSV files
- Optional retention cleanup to manage database size
- Operator status command for monitoring

#### Snapshot Exports

At each UTC day rollover (and on shutdown), the daemon exports all snapshots for the completed day:

```powershell
# Enable snapshot exports (default: ON)
poetry run pmq ops daemon --snapshot-export

# Disable snapshot exports
poetry run pmq ops daemon --no-snapshot-export
```

Exported files:
- `exports/snapshots_YYYY-MM-DD.csv.gz` - Compressed CSV with all market snapshots

The export uses atomic writes (temp file → rename) to ensure file integrity, with a 60-second timeout to prevent daemon hangs.

#### Retention Cleanup

Optionally delete old snapshots from the database after successful export:

```powershell
# Keep 30 days of snapshots in DB
poetry run pmq ops daemon --retention-days 30

# Keep 7 days (more aggressive cleanup)
poetry run pmq ops daemon --retention-days 7

# Disable retention (default - keep all snapshots)
poetry run pmq ops daemon
```

Safety features:
- Only deletes snapshots OLDER than the retention cutoff
- Never deletes the current or just-exported day's data
- Only runs after successful export

#### Operator Status Command

Check the current state of your data capture:

```powershell
# Human-readable status
poetry run pmq ops status

# JSON output (for scripting/monitoring)
poetry run pmq ops status --json
```

Output includes:
- Total snapshot count and latest snapshot time
- Daemon last tick timestamp and total ticks
- Latest coverage statistics from daily export

Example output:
```
Polymarket Ops Status

Snapshots:
  Total: 1,234,567
  Latest: 2024-01-15T23:59:00+00:00

Daemon:
  Last tick: 2024-01-15T23:59:00+00:00
  Total ticks: 1440

Latest Coverage (2024-01-14):
  Ticks: 1,440
  Snapshots: 288,000
  WSS hits: 250,000
  REST fallbacks: 30,000
  WSS coverage: 89.3%
```

#### Phase 5.2 CLI Flags

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--snapshot-export/--no-snapshot-export` | bool | True | Export snapshots on day rollover |
| `--snapshot-export-format` | str | `csv_gz` | Export format (only csv_gz supported) |
| `--retention-days` | int | None | Delete snapshots older than N days |

#### 24/7 Production Run Example

```powershell
# Run indefinitely with 30-day retention, exporting daily
poetry run pmq ops daemon --interval 60 --retention-days 30 --export-dir ./data/exports
```

This configuration:
1. Captures snapshots every 60 seconds
2. Exports compressed snapshots at each UTC midnight
3. Deletes snapshots older than 30 days after export
4. Exports coverage stats and tick history daily
5. Stops gracefully on Ctrl+C with final export

### WSS Reliability + Coverage Uplift (Phase 5.3)

Phase 5.3 improves WebSocket reliability and coverage from ~5% to 60%+ by implementing:
- Application-level keepalive ("PING" every 10s, per Polymarket's WSS quickstart)
- Adaptive staleness threshold based on daemon interval
- Enhanced tick logging with detailed freshness breakdown
- Cache age statistics for observability

#### Keepalive Protocol

Polymarket's WSS server expects application-level keepalive messages (not library-level WebSocket pings). Phase 5.3 sends:
- Literal `"PING"` text frame every 10 seconds
- Handles `"PONG"` responses (and gracefully ignores `"PING"` echoes)

This prevents the server from closing idle connections, dramatically improving WSS coverage.

#### Adaptive Staleness

By default, the staleness threshold adapts to the daemon interval:

```
effective_staleness = max(3.0 × interval_seconds, 60.0)
```

Examples:
- 60s interval → 180s staleness (3 missed updates = stale)
- 30s interval → 90s staleness
- 10s interval → 60s staleness (floor)

This prevents false positives where data is marked stale before it has a chance to update.

```powershell
# Use adaptive staleness (default in Phase 5.3)
poetry run pmq ops daemon --interval 60

# Override with explicit staleness threshold
poetry run pmq ops daemon --interval 60 --wss-staleness 120
```

#### Enhanced Tick Logging

Each tick now logs detailed freshness breakdown:

```
Tick completed: markets=200, snapshots=200, wss_fresh=180, wss_stale=10, wss_missing=10,
               rest_fallback=20, cache_age_median=5.2s, cache_age_max=45.1s
```

Coverage metrics:
- **wss_fresh**: Tokens with fresh WSS data (under staleness threshold)
- **wss_stale**: Tokens with stale WSS data (over threshold but present)
- **wss_missing**: Tokens not in WSS cache at all
- **rest_fallback**: Tokens fetched via REST (stale + missing)
- **cache_age_median/max**: Cache age statistics in seconds

#### Phase 5.3 CLI Changes

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--wss-staleness` | float | None | WSS staleness threshold (None = adaptive) |

When `--wss-staleness` is not provided, adaptive staleness is used automatically.

#### WSS Coverage Target

With Phase 5.3, a 15-minute daemon run should achieve:
- **wss_coverage_pct ≥ 60%** (vs ~5% before keepalive)
- Lower REST fallback rate
- More stable long-running sessions

#### Monitoring WSS Health

```powershell
# Run daemon and monitor logs for coverage
poetry run pmq ops daemon --interval 60 --max-hours 0.25

# Check final coverage in daily export JSON
# Or monitor per-tick logs for wss_fresh vs rest_fallback ratio
```

## Project Structure

```
polymarket-quant-lab/
├── src/pmq/
│   ├── __init__.py
│   ├── cli.py              # Typer CLI commands
│   ├── config.py           # Pydantic settings
│   ├── gamma_client.py     # Gamma API client
│   ├── logging.py          # Structured logging
│   ├── models.py           # Pydantic data models
│   ├── storage/
│   │   ├── db.py           # SQLite database
│   │   ├── dao.py          # Data access layer
│   │   └── schema.sql      # Database schema
│   ├── strategies/
│   │   ├── arb.py          # Arbitrage scanner
│   │   ├── statarb.py      # Stat-arb scanner
│   │   └── paper.py        # Paper trading ledger
│   ├── backtest/           # Backtesting framework (Phase 2)
│   │   ├── engine.py       # Core backtest engine
│   │   ├── runner.py       # Strategy orchestration
│   │   └── metrics.py      # Performance metrics
│   ├── quality/            # Snapshot quality (Phase 2.5)
│   │   ├── checks.py       # Gap/duplicate detection
│   │   └── report.py       # Quality reporting
│   ├── governance/         # Strategy approval (Phase 3)
│   │   ├── scorecard.py    # Backtest evaluation
│   │   └── risk_gate.py    # Approval enforcement
│   ├── evaluation/         # Evaluation pipeline (Phase 4)
│   │   ├── pipeline.py     # End-to-end orchestration
│   │   └── reporter.py     # Report generation
│   └── web/
│       ├── app.py          # FastAPI application
│       ├── routes.py       # API endpoints
│       ├── static/         # Static files (favicon)
│       └── templates/      # HTML templates
├── tests/
│   ├── test_gamma_client.py
│   ├── test_signals.py
│   ├── test_paper_ledger.py
│   ├── test_backtest.py    # Backtest tests
│   ├── test_quality.py     # Quality/manifest tests
│   ├── test_governance.py  # Approval/risk gate tests
│   ├── test_evaluation.py  # Evaluation pipeline tests
│   ├── test_statarb.py     # StatArb pairs config tests
│   └── test_web_and_export.py
├── config/
│   └── pairs.yml           # Stat-arb pairs config
├── .github/workflows/
│   └── ci.yml              # GitHub Actions CI
├── pyproject.toml
└── README.md
```

## Safety Notes

- **NO LIVE TRADING**: Phase 1 is paper trading only. No real money is at risk.
- **No API Keys Required**: All data comes from public endpoints.
- **Kill Switch**: Set `PMQ_SAFETY_KILL_SWITCH=true` to halt all operations.
- **Audit Trail**: All operations are logged for review.

## Development

### Run Tests

```powershell
poetry run pytest

# With coverage
poetry run pytest --cov=pmq --cov-report=html
```

### Linting & Type Checking

```powershell
# Lint
poetry run ruff check src tests

# Format
poetry run ruff format src tests

# Type check
poetry run mypy src
```

## API Reference

### Polymarket APIs Used

- **Gamma API** (`https://gamma-api.polymarket.com`): Market metadata and pricing
  - `/markets` - List markets
  - `/markets/{id}` - Get specific market
  - `/events` - List events

### Data Flow

1. `pmq sync` → Fetches from Gamma API → Stores in SQLite
2. `pmq scan` → Reads from SQLite → Detects signals → Saves signals
3. `pmq paper run` → Fetches fresh data → Scans → Executes paper trades → Updates positions

## Roadmap

### Phase 1 ✅
- [x] Market data ingestion from Gamma API
- [x] Arbitrage signal detection
- [x] Statistical arbitrage framework
- [x] Paper trading with SQLite storage
- [x] CLI interface
- [x] Safety guardrails
- [x] CI/CD pipeline

### Phase 1.5 ✅
- [x] Local-only web dashboard (FastAPI + Uvicorn)
- [x] Operator loop with exponential backoff
- [x] CSV data export
- [x] Runtime state tracking
- [x] Cache TTL correctness fix

### Phase 2 ✅
- [x] Deterministic backtesting engine
- [x] Historical market snapshots
- [x] Backtest CLI commands (run, report, export)
- [x] Performance metrics (PnL, drawdown, Sharpe, win rate)
- [x] Replay-based strategy evaluation

### Phase 2.5 ✅
- [x] Snapshot scheduler (`pmq snapshots run`)
- [x] Data quality validation (gap/duplicate detection)
- [x] Coverage reporting (`pmq snapshots quality/coverage`)
- [x] Backtest run manifests (config hash, git SHA)
- [x] Dashboard snapshot/quality endpoints

### Phase 3 ✅
- [x] Strategy scorecards (evaluate backtest results)
- [x] Approval registry (grant/revoke/list)
- [x] RiskGate enforcement (block unapproved strategies)
- [x] Risk events logging
- [x] Dashboard governance section
- [x] CLI approval commands (`pmq approve`)

### Phase 4 ✅
- [x] Evaluation pipeline (`pmq eval run/list/report/export`)
- [x] Automated quality → backtest → approval flow
- [x] Deterministic go/no-go reports (MD/JSON/CSV)
- [x] Evaluation artifacts persistence
- [x] Dashboard evaluations section
- [x] API endpoints (`/api/evals`)

### Phase 4.1 ✅
- [x] StatArb pairs config validation + schema
- [x] `pmq statarb pairs suggest` - Generate pairs from snapshots
- [x] `pmq statarb pairs validate` - Validate pairs config file
- [x] `pmq statarb explain` - Debug 0 trades issues
- [x] Evaluation pipeline integration (`--pairs` flag)
- [x] Pairs config artifacts in evaluation reports

### Phase 4.2 ✓
- [x] `pmq statarb discover` - Correlation-based pair discovery
- [x] `pmq statarb validate` - Validate overlap in date range
- [x] Deterministic discovery output (same inputs → same outputs)
- [x] Discovery tests (correlation, overlap, determinism)
- [x] Documentation: StatArb Quickstart

### Phase 4.3-4.4 ✓
- [x] Walk-forward evaluation for statarb z-score
- [x] Train/test split with fitted pair parameters
- [x] Scorecard evaluated on TEST only (no data leakage)

### Phase 4.5 ✓
- [x] Gap-aware contiguous data windows
- [x] Prevents old session gaps from penalizing recent healthy data
- [x] Contiguous mode in quality checks and walk-forward

### Phase 4.6 ✓
- [x] Realistic transaction costs (fee_bps, slippage_bps)
- [x] Market constraint filtering (min_liquidity, max_spread)
- [x] Cost assumptions in evaluation reports

### Phase 4.7 ✓
- [x] Quality window alignment with walk-forward evaluation
- [x] `check_explicit_window()` for proper expected/observed computation
- [x] Effective window quality fields in EvaluationResult
- [x] Approval gate uses aligned quality percentage
- [x] Reporter shows effective window quality section

### Phase 4.8 ✓
- [x] Eval pipeline realism: costs + constraints for StatArb
- [x] CLI flags (`--fee-bps`, `--slippage-bps`, `--min-liquidity`, `--max-spread`)
- [x] Precedence rules (CLI > YAML > defaults)
- [x] Constraint filtering with pair counts in reports
- [x] Cost Assumptions section in evaluation reports
- [x] Constraint Filtering section in evaluation reports

### Phase 4.9 ✓
- [x] Microstructure Truth: Real bid/ask spread + top-of-book liquidity from CLOB
- [x] Order book fetching from Polymarket public CLOB API
- [x] Schema extended with microstructure columns (backward compatible)
- [x] Snapshot collection with `--with-orderbook` flag
- [x] Constraints prefer real spread_bps/top_depth_usd over heuristics
- [x] Microstructure section in evaluation reports
- [x] Paper-only: no wallet auth or order placement

### Phase 5.0 ✓
- [x] Market WebSocket client for real-time order book streaming
- [x] `--orderbook-source {rest,wss}` flag for snapshot collection
- [x] In-memory cache with staleness detection
- [x] Automatic reconnection with exponential backoff + jitter
- [x] Fallback to REST when WSS data is stale/missing
- [x] WSS coverage statistics in CLI output
- [x] Thread-safe cache access for concurrent reads
- [x] Paper-only: no auth required (public Market channel only)

### Phase 5.1 ✓
- [x] Continuous snapshot capture daemon (`pmq ops daemon`)
- [x] DaemonRunner with injectable dependencies (clock, sleep, WSS, DAO)
- [x] Coverage tracking per tick (wss_hits, rest_fallbacks, stale, missing)
- [x] Daily UTC rollover exports (CSV, JSON, markdown)
- [x] Graceful shutdown on SIGINT/SIGTERM
- [x] Systemd and Windows Task Scheduler deployment templates
- [x] Unit tests with mocked dependencies

### Phase 5.x (Future)
- [ ] Authenticated CLOB integration
- [ ] Real order placement via py-clob-client
- [ ] Wallet integration (Polygon)
- [ ] Advanced signal strategies

## License

MIT License - see [LICENSE](LICENSE) for details.

## Disclaimer

This software is for educational and research purposes only. Trading prediction markets involves significant risk. The authors are not responsible for any financial losses incurred from using this software. Always do your own research and never trade more than you can afford to lose.
