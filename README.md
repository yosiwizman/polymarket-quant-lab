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
│   └── storage/
│   │   ├── db.py           # SQLite database
│   │   ├── dao.py          # Data access layer
│   │   └── schema.sql      # Database schema
│   └── strategies/
│       ├── arb.py          # Arbitrage scanner
│       ├── statarb.py      # Stat-arb scanner
│       └── paper.py        # Paper trading ledger
├── tests/
│   ├── test_gamma_client.py
│   ├── test_signals.py
│   └── test_paper_ledger.py
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

### Phase 1 (Current) ✅
- [x] Market data ingestion from Gamma API
- [x] Arbitrage signal detection
- [x] Statistical arbitrage framework
- [x] Paper trading with SQLite storage
- [x] CLI interface
- [x] Safety guardrails
- [x] CI/CD pipeline

### Phase 2 (Future)
- [ ] Authenticated CLOB integration
- [ ] Real order placement via py-clob-client
- [ ] Wallet integration (Polygon)
- [ ] Advanced signal strategies
- [ ] Backtesting framework
- [ ] Real-time WebSocket feeds

## License

MIT License - see [LICENSE](LICENSE) for details.

## Disclaimer

This software is for educational and research purposes only. Trading prediction markets involves significant risk. The authors are not responsible for any financial losses incurred from using this software. Always do your own research and never trade more than you can afford to lose.
