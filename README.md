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

### Phase 4.2 (Current) ✅
- [x] `pmq statarb discover` - Correlation-based pair discovery
- [x] `pmq statarb validate` - Validate overlap in date range
- [x] Deterministic discovery output (same inputs → same outputs)
- [x] Discovery tests (correlation, overlap, determinism)
- [x] Documentation: StatArb Quickstart

### Phase 5 (Future)
- [ ] Authenticated CLOB integration
- [ ] Real order placement via py-clob-client
- [ ] Wallet integration (Polygon)
- [ ] Advanced signal strategies
- [ ] Real-time WebSocket feeds

## License

MIT License - see [LICENSE](LICENSE) for details.

## Disclaimer

This software is for educational and research purposes only. Trading prediction markets involves significant risk. The authors are not responsible for any financial losses incurred from using this software. Always do your own research and never trade more than you can afford to lose.
