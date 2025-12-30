-- Polymarket Quant Lab Database Schema
-- SQLite database for storing market data, signals, and paper trades

-- Markets table: cached market metadata
CREATE TABLE IF NOT EXISTS markets (
    id TEXT PRIMARY KEY,
    slug TEXT,
    question TEXT,
    condition_id TEXT,
    last_price_yes REAL DEFAULT 0.0,
    last_price_no REAL DEFAULT 0.0,
    liquidity REAL DEFAULT 0.0,
    volume REAL DEFAULT 0.0,
    volume24hr REAL DEFAULT 0.0,
    active INTEGER DEFAULT 1,
    closed INTEGER DEFAULT 0,
    yes_token_id TEXT,
    no_token_id TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_markets_active ON markets(active);
CREATE INDEX IF NOT EXISTS idx_markets_slug ON markets(slug);
CREATE INDEX IF NOT EXISTS idx_markets_volume ON markets(volume24hr DESC);

-- Signals table: detected trading signals
CREATE TABLE IF NOT EXISTS signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    type TEXT NOT NULL,  -- ARBITRAGE, STAT_ARB
    market_ids TEXT NOT NULL,  -- JSON array of market IDs
    payload_json TEXT NOT NULL,  -- Full signal data as JSON
    profit_potential REAL DEFAULT 0.0,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_signals_type ON signals(type);
CREATE INDEX IF NOT EXISTS idx_signals_created ON signals(created_at DESC);

-- Paper trades table: simulated trade executions
CREATE TABLE IF NOT EXISTS paper_trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    strategy TEXT NOT NULL,  -- arb, statarb, manual
    market_id TEXT NOT NULL,
    market_question TEXT,
    side TEXT NOT NULL,  -- BUY, SELL
    outcome TEXT NOT NULL,  -- YES, NO
    price REAL NOT NULL,
    quantity REAL NOT NULL,
    notional REAL NOT NULL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (market_id) REFERENCES markets(id)
);

CREATE INDEX IF NOT EXISTS idx_paper_trades_strategy ON paper_trades(strategy);
CREATE INDEX IF NOT EXISTS idx_paper_trades_market ON paper_trades(market_id);
CREATE INDEX IF NOT EXISTS idx_paper_trades_created ON paper_trades(created_at DESC);

-- Paper positions table: current positions
CREATE TABLE IF NOT EXISTS paper_positions (
    market_id TEXT PRIMARY KEY,
    market_question TEXT,
    yes_quantity REAL DEFAULT 0.0,
    no_quantity REAL DEFAULT 0.0,
    avg_price_yes REAL DEFAULT 0.0,
    avg_price_no REAL DEFAULT 0.0,
    realized_pnl REAL DEFAULT 0.0,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (market_id) REFERENCES markets(id)
);

-- Safety audit log: track all trading operations
CREATE TABLE IF NOT EXISTS audit_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type TEXT NOT NULL,  -- TRADE, SIGNAL, SAFETY_CHECK, ERROR
    market_id TEXT,
    details_json TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_audit_event ON audit_log(event_type);
CREATE INDEX IF NOT EXISTS idx_audit_created ON audit_log(created_at DESC);

-- Rate limit tracking
CREATE TABLE IF NOT EXISTS rate_limits (
    key TEXT PRIMARY KEY,  -- e.g., 'trades_per_hour'
    count INTEGER DEFAULT 0,
    window_start TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);

-- Runtime state: operational status tracking
CREATE TABLE IF NOT EXISTS runtime_state (
    key TEXT PRIMARY KEY,
    value TEXT,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);

-- =============================================================================
-- Backtesting Tables (Phase 2)
-- =============================================================================

-- Market snapshots: immutable time-series data for backtesting
CREATE TABLE IF NOT EXISTS market_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    market_id TEXT NOT NULL,
    yes_price REAL NOT NULL,
    no_price REAL NOT NULL,
    liquidity REAL DEFAULT 0.0,
    volume REAL DEFAULT 0.0,
    snapshot_time TEXT NOT NULL,  -- UTC timestamp
    FOREIGN KEY (market_id) REFERENCES markets(id)
);

CREATE INDEX IF NOT EXISTS idx_snapshots_market ON market_snapshots(market_id);
CREATE INDEX IF NOT EXISTS idx_snapshots_time ON market_snapshots(snapshot_time);
CREATE INDEX IF NOT EXISTS idx_snapshots_market_time ON market_snapshots(market_id, snapshot_time);

-- Backtest runs: metadata for each backtest execution
CREATE TABLE IF NOT EXISTS backtest_runs (
    id TEXT PRIMARY KEY,  -- UUID run_id
    strategy TEXT NOT NULL,
    start_date TEXT NOT NULL,
    end_date TEXT NOT NULL,
    initial_balance REAL NOT NULL,
    final_balance REAL,
    config_json TEXT,  -- Strategy configuration
    status TEXT DEFAULT 'pending',  -- pending, running, completed, failed
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    completed_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_backtest_runs_strategy ON backtest_runs(strategy);
CREATE INDEX IF NOT EXISTS idx_backtest_runs_status ON backtest_runs(status);

-- Backtest trades: simulated trades during backtest
CREATE TABLE IF NOT EXISTS backtest_trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    market_id TEXT NOT NULL,
    side TEXT NOT NULL,  -- BUY, SELL
    outcome TEXT NOT NULL,  -- YES, NO
    price REAL NOT NULL,
    quantity REAL NOT NULL,
    notional REAL NOT NULL,
    trade_time TEXT NOT NULL,  -- Simulated time from snapshot
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (run_id) REFERENCES backtest_runs(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_backtest_trades_run ON backtest_trades(run_id);
CREATE INDEX IF NOT EXISTS idx_backtest_trades_market ON backtest_trades(market_id);

-- Backtest metrics: computed metrics for each run
CREATE TABLE IF NOT EXISTS backtest_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL UNIQUE,
    total_pnl REAL DEFAULT 0.0,
    max_drawdown REAL DEFAULT 0.0,
    win_rate REAL DEFAULT 0.0,
    sharpe_ratio REAL DEFAULT 0.0,
    total_trades INTEGER DEFAULT 0,
    trades_per_day REAL DEFAULT 0.0,
    capital_utilization REAL DEFAULT 0.0,
    metrics_json TEXT,  -- Additional metrics as JSON
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (run_id) REFERENCES backtest_runs(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_backtest_metrics_run ON backtest_metrics(run_id);

-- =============================================================================
-- Phase 2.5: Snapshot Pipeline + Replay Hardening
-- =============================================================================

-- Snapshot quality reports: track data quality over time windows
CREATE TABLE IF NOT EXISTS snapshot_quality_reports (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    window_from TEXT NOT NULL,
    window_to TEXT NOT NULL,
    expected_interval_seconds INTEGER NOT NULL,
    markets_seen INTEGER DEFAULT 0,
    snapshots_written INTEGER DEFAULT 0,
    missing_intervals INTEGER DEFAULT 0,
    largest_gap_seconds REAL DEFAULT 0.0,
    duplicate_count INTEGER DEFAULT 0,
    stale_market_count INTEGER DEFAULT 0,
    coverage_pct REAL DEFAULT 0.0,
    notes_json TEXT
);

CREATE INDEX IF NOT EXISTS idx_quality_reports_window ON snapshot_quality_reports(window_from, window_to);
CREATE INDEX IF NOT EXISTS idx_quality_reports_created ON snapshot_quality_reports(created_at DESC);

-- Backtest manifests: full reproducibility record for each run
CREATE TABLE IF NOT EXISTS backtest_manifests (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL UNIQUE,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    strategy TEXT NOT NULL,
    window_from TEXT NOT NULL,
    window_to TEXT NOT NULL,
    snapshot_interval_seconds INTEGER,
    market_filter_json TEXT,  -- JSON array of market IDs if filtered
    config_hash TEXT,  -- SHA256 of normalized config
    code_git_sha TEXT,  -- Git commit SHA at run time
    snapshot_count INTEGER DEFAULT 0,
    snapshot_time_range_json TEXT,  -- first/last snapshot times
    notes TEXT,
    FOREIGN KEY (run_id) REFERENCES backtest_runs(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_manifests_run ON backtest_manifests(run_id);
CREATE INDEX IF NOT EXISTS idx_manifests_strategy ON backtest_manifests(strategy);
