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

#### Snapshot Quality & Coverage (Phase 2.5)

Check snapshot data quality before running backtests:

```powershell
# Check overall snapshot stats
poetry run pmq snapshots summary

# Analyze quality for a time window (detects gaps, duplicates)
poetry run pmq snapshots quality --from 2024-01-01 --to 2024-01-07 --interval 60

# View coverage by market
poetry run pmq snapshots coverage --from 2024-01-01 --to 2024-01-07
```

Quality metrics:
- **Coverage %**: Percentage of expected intervals with data
- **Missing Intervals**: Gaps larger than 50% of expected interval
- **Duplicates**: Same market+timestamp entries
- **Status Badge**: healthy (≥95% coverage), degraded (≥80%), unhealthy (<80%)

#### Backtest Manifests

Every backtest run automatically saves a manifest for reproducibility:

- **Config Hash**: SHA256 of all config parameters
- **Git SHA**: Commit hash at run time
- **Snapshot Range**: First/last snapshot times used
- **Market Filter**: List of market IDs (for stat-arb)

Manifests ensure backtests can be reproduced exactly by recording all inputs.

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

### Stat-Arb Pairs Configuration

Configure market pairs for statistical arbitrage in `config/pairs.yml`:

```yaml
pairs:
  - market_a_id: "0x1234..."
    market_b_id: "0x5678..."
    name: "BTC vs ETH prediction"
    correlation: 1.0  # Same direction

  - market_a_id: "0xabcd..."
    market_b_id: "0xefgh..."
    name: "Inverse pair"
    correlation: -1.0  # Opposite direction
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

### Phase 2.5 (Current) ✅
- [x] Snapshot scheduler (`pmq snapshots run`)
- [x] Data quality validation (gap/duplicate detection)
- [x] Coverage reporting (`pmq snapshots quality/coverage`)
- [x] Backtest run manifests (config hash, git SHA)
- [x] Dashboard snapshot/quality endpoints

### Phase 3 (Future)
- [ ] Authenticated CLOB integration
- [ ] Real order placement via py-clob-client
- [ ] Wallet integration (Polygon)
- [ ] Advanced signal strategies
- [ ] Real-time WebSocket feeds

## License

MIT License - see [LICENSE](LICENSE) for details.

## Disclaimer

This software is for educational and research purposes only. Trading prediction markets involves significant risk. The authors are not responsible for any financial losses incurred from using this software. Always do your own research and never trade more than you can afford to lose.
