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
