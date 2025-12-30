"""Data Access Object layer for database operations."""

import json
from datetime import UTC, datetime
from typing import Any

from pmq.logging import get_logger
from pmq.models import (
    ArbitrageSignal,
    GammaMarket,
    Outcome,
    PaperPosition,
    PaperTrade,
    Side,
    SignalType,
    StatArbSignal,
)
from pmq.storage.db import Database, get_database

logger = get_logger("storage.dao")


class DAO:
    """Data Access Object for all database operations."""

    def __init__(self, db: Database | None = None) -> None:
        """Initialize DAO.

        Args:
            db: Database instance (uses global if not provided)
        """
        self._db = db or get_database()

    # =========================================================================
    # Markets
    # =========================================================================

    def upsert_market(self, market: GammaMarket) -> None:
        """Insert or update a market record.

        Args:
            market: Market data to upsert
        """
        now = datetime.now(UTC).isoformat()
        self._db.execute(
            """
            INSERT INTO markets (
                id, slug, question, condition_id, last_price_yes, last_price_no,
                liquidity, volume, volume24hr, active, closed, yes_token_id, no_token_id,
                updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                slug = excluded.slug,
                question = excluded.question,
                condition_id = excluded.condition_id,
                last_price_yes = excluded.last_price_yes,
                last_price_no = excluded.last_price_no,
                liquidity = excluded.liquidity,
                volume = excluded.volume,
                volume24hr = excluded.volume24hr,
                active = excluded.active,
                closed = excluded.closed,
                yes_token_id = excluded.yes_token_id,
                no_token_id = excluded.no_token_id,
                updated_at = excluded.updated_at
            """,
            (
                market.id,
                market.slug,
                market.question,
                market.condition_id,
                market.yes_price,
                market.no_price,
                market.liquidity,
                market.volume,
                market.volume24hr,
                1 if market.active else 0,
                1 if market.closed else 0,
                market.yes_token_id,
                market.no_token_id,
                now,
            ),
        )

    def upsert_markets(self, markets: list[GammaMarket]) -> int:
        """Bulk upsert markets.

        Args:
            markets: List of markets to upsert

        Returns:
            Number of markets upserted
        """
        for market in markets:
            self.upsert_market(market)
        logger.info(f"Upserted {len(markets)} markets")
        return len(markets)

    def get_market(self, market_id: str) -> dict[str, Any] | None:
        """Get a market by ID.

        Args:
            market_id: Market identifier

        Returns:
            Market data dict or None
        """
        row = self._db.fetch_one("SELECT * FROM markets WHERE id = ?", (market_id,))
        return dict(row) if row else None

    def get_active_markets(self, limit: int = 100) -> list[dict[str, Any]]:
        """Get active markets ordered by volume.

        Args:
            limit: Maximum number of results

        Returns:
            List of market dicts
        """
        rows = self._db.fetch_all(
            """
            SELECT * FROM markets
            WHERE active = 1 AND closed = 0
            ORDER BY volume24hr DESC
            LIMIT ?
            """,
            (limit,),
        )
        return [dict(row) for row in rows]

    def get_markets_by_ids(self, market_ids: list[str]) -> list[dict[str, Any]]:
        """Get markets by list of IDs.

        Args:
            market_ids: List of market identifiers

        Returns:
            List of market dicts
        """
        if not market_ids:
            return []
        placeholders = ",".join("?" * len(market_ids))
        rows = self._db.fetch_all(
            f"SELECT * FROM markets WHERE id IN ({placeholders})",
            tuple(market_ids),
        )
        return [dict(row) for row in rows]

    # =========================================================================
    # Signals
    # =========================================================================

    def save_signal(
        self,
        signal_type: SignalType,
        market_ids: list[str],
        payload: dict[str, Any],
        profit_potential: float = 0.0,
    ) -> int:
        """Save a trading signal.

        Args:
            signal_type: Type of signal
            market_ids: Involved market IDs
            payload: Full signal data
            profit_potential: Expected profit

        Returns:
            Signal ID
        """
        cursor = self._db.execute(
            """
            INSERT INTO signals (type, market_ids, payload_json, profit_potential)
            VALUES (?, ?, ?, ?)
            """,
            (
                signal_type.value,
                json.dumps(market_ids),
                json.dumps(payload),
                profit_potential,
            ),
        )
        signal_id = cursor.lastrowid or 0
        logger.debug(f"Saved signal {signal_id}: {signal_type.value}")
        return signal_id

    def save_arb_signal(self, signal: ArbitrageSignal) -> int:
        """Save an arbitrage signal.

        Args:
            signal: Arbitrage signal to save

        Returns:
            Signal ID
        """
        return self.save_signal(
            SignalType.ARBITRAGE,
            [signal.market_id],
            signal.model_dump(mode="json"),
            signal.profit_potential,
        )

    def save_statarb_signal(self, signal: StatArbSignal) -> int:
        """Save a stat-arb signal.

        Args:
            signal: Stat-arb signal to save

        Returns:
            Signal ID
        """
        return self.save_signal(
            SignalType.STAT_ARB,
            [signal.market_a_id, signal.market_b_id],
            signal.model_dump(mode="json"),
            abs(signal.spread),
        )

    def get_recent_signals(
        self,
        signal_type: SignalType | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Get recent signals.

        Args:
            signal_type: Filter by signal type
            limit: Maximum number of results

        Returns:
            List of signal dicts with parsed payload
        """
        if signal_type:
            rows = self._db.fetch_all(
                """
                SELECT * FROM signals WHERE type = ?
                ORDER BY created_at DESC LIMIT ?
                """,
                (signal_type.value, limit),
            )
        else:
            rows = self._db.fetch_all(
                "SELECT * FROM signals ORDER BY created_at DESC LIMIT ?",
                (limit,),
            )

        results = []
        for row in rows:
            data = dict(row)
            data["market_ids"] = json.loads(data["market_ids"])
            data["payload"] = json.loads(data["payload_json"])
            del data["payload_json"]
            results.append(data)
        return results

    # =========================================================================
    # Paper Trades
    # =========================================================================

    def save_paper_trade(self, trade: PaperTrade) -> int:
        """Save a paper trade.

        Args:
            trade: Paper trade to save

        Returns:
            Trade ID
        """
        cursor = self._db.execute(
            """
            INSERT INTO paper_trades (
                strategy, market_id, market_question, side, outcome,
                price, quantity, notional
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                trade.strategy,
                trade.market_id,
                trade.market_question,
                trade.side.value,
                trade.outcome.value,
                trade.price,
                trade.quantity,
                trade.notional,
            ),
        )
        trade_id = cursor.lastrowid or 0
        logger.info(
            f"Paper trade {trade_id}: {trade.side.value} {trade.quantity} "
            f"{trade.outcome.value} @ {trade.price} on {trade.market_id[:8]}..."
        )
        return trade_id

    def get_paper_trades(
        self,
        strategy: str | None = None,
        market_id: str | None = None,
        limit: int = 100,
    ) -> list[PaperTrade]:
        """Get paper trades with optional filters.

        Args:
            strategy: Filter by strategy
            market_id: Filter by market
            limit: Maximum number of results

        Returns:
            List of paper trades
        """
        conditions = []
        params: list[Any] = []

        if strategy:
            conditions.append("strategy = ?")
            params.append(strategy)
        if market_id:
            conditions.append("market_id = ?")
            params.append(market_id)

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        params.append(limit)

        rows = self._db.fetch_all(
            f"SELECT * FROM paper_trades {where} ORDER BY created_at DESC LIMIT ?",
            tuple(params),
        )

        trades = []
        for row in rows:
            trades.append(
                PaperTrade(
                    id=row["id"],
                    strategy=row["strategy"],
                    market_id=row["market_id"],
                    market_question=row["market_question"] or "",
                    side=Side(row["side"]),
                    outcome=Outcome(row["outcome"]),
                    price=row["price"],
                    quantity=row["quantity"],
                    notional=row["notional"],
                    created_at=datetime.fromisoformat(row["created_at"]),
                )
            )
        return trades

    def count_trades_in_window(self, hours: int = 1) -> int:
        """Count paper trades in the last N hours.

        Args:
            hours: Time window in hours

        Returns:
            Number of trades
        """
        row = self._db.fetch_one(
            """
            SELECT COUNT(*) as count FROM paper_trades
            WHERE created_at > datetime('now', ?)
            """,
            (f"-{hours} hours",),
        )
        return row["count"] if row else 0

    # =========================================================================
    # Paper Positions
    # =========================================================================

    def get_position(self, market_id: str) -> PaperPosition | None:
        """Get paper position for a market.

        Args:
            market_id: Market identifier

        Returns:
            Position or None
        """
        row = self._db.fetch_one(
            "SELECT * FROM paper_positions WHERE market_id = ?",
            (market_id,),
        )
        if not row:
            return None

        return PaperPosition(
            market_id=row["market_id"],
            market_question=row["market_question"] or "",
            yes_quantity=row["yes_quantity"],
            no_quantity=row["no_quantity"],
            avg_price_yes=row["avg_price_yes"],
            avg_price_no=row["avg_price_no"],
            realized_pnl=row["realized_pnl"],
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )

    def upsert_position(self, position: PaperPosition) -> None:
        """Insert or update a paper position.

        Args:
            position: Position to upsert
        """
        now = datetime.now(UTC).isoformat()
        self._db.execute(
            """
            INSERT INTO paper_positions (
                market_id, market_question, yes_quantity, no_quantity,
                avg_price_yes, avg_price_no, realized_pnl, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(market_id) DO UPDATE SET
                market_question = excluded.market_question,
                yes_quantity = excluded.yes_quantity,
                no_quantity = excluded.no_quantity,
                avg_price_yes = excluded.avg_price_yes,
                avg_price_no = excluded.avg_price_no,
                realized_pnl = excluded.realized_pnl,
                updated_at = excluded.updated_at
            """,
            (
                position.market_id,
                position.market_question,
                position.yes_quantity,
                position.no_quantity,
                position.avg_price_yes,
                position.avg_price_no,
                position.realized_pnl,
                now,
            ),
        )

    def get_all_positions(self) -> list[PaperPosition]:
        """Get all paper positions.

        Returns:
            List of positions with any holdings
        """
        rows = self._db.fetch_all(
            """
            SELECT * FROM paper_positions
            WHERE yes_quantity != 0 OR no_quantity != 0
            ORDER BY updated_at DESC
            """
        )

        return [
            PaperPosition(
                market_id=row["market_id"],
                market_question=row["market_question"] or "",
                yes_quantity=row["yes_quantity"],
                no_quantity=row["no_quantity"],
                avg_price_yes=row["avg_price_yes"],
                avg_price_no=row["avg_price_no"],
                realized_pnl=row["realized_pnl"],
                updated_at=datetime.fromisoformat(row["updated_at"]),
            )
            for row in rows
        ]

    def count_positions(self) -> int:
        """Count open positions.

        Returns:
            Number of positions with holdings
        """
        row = self._db.fetch_one(
            """
            SELECT COUNT(*) as count FROM paper_positions
            WHERE yes_quantity != 0 OR no_quantity != 0
            """
        )
        return row["count"] if row else 0

    # =========================================================================
    # Audit Log
    # =========================================================================

    def log_audit(
        self,
        event_type: str,
        market_id: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Log an audit event.

        Args:
            event_type: Type of event
            market_id: Related market ID
            details: Additional details
        """
        self._db.execute(
            """
            INSERT INTO audit_log (event_type, market_id, details_json)
            VALUES (?, ?, ?)
            """,
            (event_type, market_id, json.dumps(details) if details else None),
        )

    def get_audit_log(self, limit: int = 100) -> list[dict[str, Any]]:
        """Get recent audit log entries.

        Args:
            limit: Maximum number of results

        Returns:
            List of audit entries
        """
        rows = self._db.fetch_all(
            "SELECT * FROM audit_log ORDER BY created_at DESC LIMIT ?",
            (limit,),
        )
        results = []
        for row in rows:
            data = dict(row)
            if data.get("details_json"):
                data["details"] = json.loads(data["details_json"])
            del data["details_json"]
            results.append(data)
        return results

    # =========================================================================
    # Statistics
    # =========================================================================

    def get_trading_stats(self) -> dict[str, Any]:
        """Get paper trading statistics.

        Returns:
            Dict with trading stats
        """
        trades_row = self._db.fetch_one(
            """
            SELECT
                COUNT(*) as total_trades,
                SUM(notional) as total_notional,
                COUNT(DISTINCT market_id) as unique_markets
            FROM paper_trades
            """
        )

        pnl_row = self._db.fetch_one(
            "SELECT SUM(realized_pnl) as total_realized_pnl FROM paper_positions"
        )

        signals_row = self._db.fetch_one("SELECT COUNT(*) as total_signals FROM signals")

        return {
            "total_trades": trades_row["total_trades"] if trades_row else 0,
            "total_notional": trades_row["total_notional"] or 0 if trades_row else 0,
            "unique_markets": trades_row["unique_markets"] if trades_row else 0,
            "total_realized_pnl": pnl_row["total_realized_pnl"] or 0 if pnl_row else 0,
            "total_signals": signals_row["total_signals"] if signals_row else 0,
            "open_positions": self.count_positions(),
        }

    # =========================================================================
    # Runtime State
    # =========================================================================

    def set_runtime_state(self, key: str, value: str) -> None:
        """Set a runtime state value.

        Args:
            key: State key (e.g., 'last_sync_at')
            value: State value
        """
        now = datetime.now(UTC).isoformat()
        self._db.execute(
            """
            INSERT INTO runtime_state (key, value, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET
                value = excluded.value,
                updated_at = excluded.updated_at
            """,
            (key, value, now),
        )

    def get_runtime_state(self, key: str) -> str | None:
        """Get a runtime state value.

        Args:
            key: State key

        Returns:
            State value or None
        """
        row = self._db.fetch_one(
            "SELECT value FROM runtime_state WHERE key = ?",
            (key,),
        )
        return row["value"] if row else None

    def get_all_runtime_state(self) -> dict[str, str]:
        """Get all runtime state values.

        Returns:
            Dict of key -> value
        """
        rows = self._db.fetch_all("SELECT key, value FROM runtime_state")
        return {row["key"]: row["value"] for row in rows}

    # =========================================================================
    # Export Helpers
    # =========================================================================

    def get_signals_for_export(
        self,
        signal_type: SignalType | None = None,
        limit: int = 1000,
    ) -> list[dict[str, Any]]:
        """Get signals in export-friendly format.

        Args:
            signal_type: Filter by signal type
            limit: Maximum number of results

        Returns:
            List of signal dicts with flattened payload
        """
        if signal_type:
            rows = self._db.fetch_all(
                """
                SELECT id, type, market_ids, payload_json, profit_potential, created_at
                FROM signals WHERE type = ?
                ORDER BY created_at DESC LIMIT ?
                """,
                (signal_type.value, limit),
            )
        else:
            rows = self._db.fetch_all(
                """
                SELECT id, type, market_ids, payload_json, profit_potential, created_at
                FROM signals ORDER BY created_at DESC LIMIT ?
                """,
                (limit,),
            )

        results = []
        for row in rows:
            data = dict(row)
            # Parse JSON fields
            data["market_ids"] = json.loads(data["market_ids"])
            payload = json.loads(data["payload_json"])
            del data["payload_json"]
            # Flatten payload into data
            for k, v in payload.items():
                if k not in data:
                    data[k] = v
            results.append(data)
        return results

    def get_trades_for_export(self, limit: int = 1000) -> list[dict[str, Any]]:
        """Get trades in export-friendly format.

        Args:
            limit: Maximum number of results

        Returns:
            List of trade dicts
        """
        rows = self._db.fetch_all(
            """
            SELECT id, strategy, market_id, market_question, side, outcome,
                   price, quantity, notional, created_at
            FROM paper_trades ORDER BY created_at DESC LIMIT ?
            """,
            (limit,),
        )
        return [dict(row) for row in rows]

    def get_positions_for_export(self) -> list[dict[str, Any]]:
        """Get positions in export-friendly format.

        Returns:
            List of position dicts
        """
        rows = self._db.fetch_all(
            """
            SELECT market_id, market_question, yes_quantity, no_quantity,
                   avg_price_yes, avg_price_no, realized_pnl, created_at, updated_at
            FROM paper_positions
            WHERE yes_quantity != 0 OR no_quantity != 0
            ORDER BY updated_at DESC
            """
        )
        return [dict(row) for row in rows]

    # =========================================================================
    # Market Snapshots (Backtest)
    # =========================================================================

    def save_snapshot(
        self,
        market_id: str,
        yes_price: float,
        no_price: float,
        liquidity: float,
        volume: float,
        snapshot_time: str,
    ) -> int:
        """Save a market snapshot for backtesting.

        Args:
            market_id: Market identifier
            yes_price: YES token price at snapshot time
            no_price: NO token price at snapshot time
            liquidity: Liquidity at snapshot time
            volume: Volume at snapshot time
            snapshot_time: UTC timestamp of snapshot

        Returns:
            Snapshot ID
        """
        cursor = self._db.execute(
            """
            INSERT INTO market_snapshots
            (market_id, yes_price, no_price, liquidity, volume, snapshot_time)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (market_id, yes_price, no_price, liquidity, volume, snapshot_time),
        )
        return cursor.lastrowid or 0

    def save_snapshots_bulk(self, markets: list[GammaMarket], snapshot_time: str) -> int:
        """Save snapshots for multiple markets.

        Args:
            markets: List of markets to snapshot
            snapshot_time: UTC timestamp for all snapshots

        Returns:
            Number of snapshots saved
        """
        count = 0
        for market in markets:
            if market.yes_price > 0 and market.no_price > 0:
                self.save_snapshot(
                    market_id=market.id,
                    yes_price=market.yes_price,
                    no_price=market.no_price,
                    liquidity=market.liquidity,
                    volume=market.volume,
                    snapshot_time=snapshot_time,
                )
                count += 1
        logger.info(f"Saved {count} snapshots at {snapshot_time}")
        return count

    def get_snapshots(
        self,
        start_time: str,
        end_time: str,
        market_ids: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Get snapshots within a time range.

        Args:
            start_time: Start of time range (ISO format)
            end_time: End of time range (ISO format)
            market_ids: Optional list of market IDs to filter

        Returns:
            List of snapshot dicts ordered by time
        """
        if market_ids:
            placeholders = ",".join("?" * len(market_ids))
            rows = self._db.fetch_all(
                f"""
                SELECT * FROM market_snapshots
                WHERE snapshot_time >= ? AND snapshot_time <= ?
                  AND market_id IN ({placeholders})
                ORDER BY snapshot_time ASC, market_id ASC
                """,
                (start_time, end_time, *market_ids),
            )
        else:
            rows = self._db.fetch_all(
                """
                SELECT * FROM market_snapshots
                WHERE snapshot_time >= ? AND snapshot_time <= ?
                ORDER BY snapshot_time ASC, market_id ASC
                """,
                (start_time, end_time),
            )
        return [dict(row) for row in rows]

    def get_snapshot_times(self, start_time: str, end_time: str) -> list[str]:
        """Get distinct snapshot times in a range.

        Args:
            start_time: Start of time range
            end_time: End of time range

        Returns:
            List of distinct snapshot times (sorted)
        """
        rows = self._db.fetch_all(
            """
            SELECT DISTINCT snapshot_time FROM market_snapshots
            WHERE snapshot_time >= ? AND snapshot_time <= ?
            ORDER BY snapshot_time ASC
            """,
            (start_time, end_time),
        )
        return [row["snapshot_time"] for row in rows]

    def count_snapshots(self, start_time: str | None = None, end_time: str | None = None) -> int:
        """Count snapshots, optionally within a time range.

        Args:
            start_time: Optional start of time range
            end_time: Optional end of time range

        Returns:
            Number of snapshots
        """
        if start_time and end_time:
            row = self._db.fetch_one(
                "SELECT COUNT(*) as count FROM market_snapshots WHERE snapshot_time >= ? AND snapshot_time <= ?",
                (start_time, end_time),
            )
        else:
            row = self._db.fetch_one("SELECT COUNT(*) as count FROM market_snapshots")
        return row["count"] if row else 0

    # =========================================================================
    # Backtest Runs
    # =========================================================================

    def create_backtest_run(
        self,
        run_id: str,
        strategy: str,
        start_date: str,
        end_date: str,
        initial_balance: float,
        config_json: str | None = None,
    ) -> str:
        """Create a new backtest run.

        Args:
            run_id: Unique run identifier
            strategy: Strategy name
            start_date: Backtest start date
            end_date: Backtest end date
            initial_balance: Starting balance
            config_json: Optional strategy config JSON

        Returns:
            Run ID
        """
        self._db.execute(
            """
            INSERT INTO backtest_runs
            (id, strategy, start_date, end_date, initial_balance, config_json, status)
            VALUES (?, ?, ?, ?, ?, ?, 'running')
            """,
            (run_id, strategy, start_date, end_date, initial_balance, config_json),
        )
        logger.info(f"Created backtest run {run_id}: {strategy} ({start_date} to {end_date})")
        return run_id

    def complete_backtest_run(
        self,
        run_id: str,
        final_balance: float,
        status: str = "completed",
    ) -> None:
        """Mark a backtest run as completed.

        Args:
            run_id: Run identifier
            final_balance: Final balance after backtest
            status: Final status (completed/failed)
        """
        now = datetime.now(UTC).isoformat()
        self._db.execute(
            """
            UPDATE backtest_runs
            SET final_balance = ?, status = ?, completed_at = ?
            WHERE id = ?
            """,
            (final_balance, status, now, run_id),
        )

    def get_backtest_run(self, run_id: str) -> dict[str, Any] | None:
        """Get a backtest run by ID.

        Args:
            run_id: Run identifier

        Returns:
            Run data or None
        """
        row = self._db.fetch_one(
            "SELECT * FROM backtest_runs WHERE id = ?",
            (run_id,),
        )
        if row:
            data = dict(row)
            if data.get("config_json"):
                data["config"] = json.loads(data["config_json"])
            return data
        return None

    def get_backtest_runs(
        self,
        strategy: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Get backtest runs.

        Args:
            strategy: Optional strategy filter
            limit: Maximum results

        Returns:
            List of run dicts
        """
        if strategy:
            rows = self._db.fetch_all(
                """
                SELECT * FROM backtest_runs
                WHERE strategy = ?
                ORDER BY created_at DESC LIMIT ?
                """,
                (strategy, limit),
            )
        else:
            rows = self._db.fetch_all(
                "SELECT * FROM backtest_runs ORDER BY created_at DESC LIMIT ?",
                (limit,),
            )
        return [dict(row) for row in rows]

    # =========================================================================
    # Backtest Trades
    # =========================================================================

    def save_backtest_trade(
        self,
        run_id: str,
        market_id: str,
        side: str,
        outcome: str,
        price: float,
        quantity: float,
        trade_time: str,
    ) -> int:
        """Save a backtest trade.

        Args:
            run_id: Backtest run ID
            market_id: Market identifier
            side: BUY or SELL
            outcome: YES or NO
            price: Trade price
            quantity: Trade quantity
            trade_time: Simulated trade time

        Returns:
            Trade ID
        """
        notional = price * quantity
        cursor = self._db.execute(
            """
            INSERT INTO backtest_trades
            (run_id, market_id, side, outcome, price, quantity, notional, trade_time)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (run_id, market_id, side, outcome, price, quantity, notional, trade_time),
        )
        return cursor.lastrowid or 0

    def get_backtest_trades(
        self,
        run_id: str,
        limit: int = 1000,
    ) -> list[dict[str, Any]]:
        """Get trades for a backtest run.

        Args:
            run_id: Backtest run ID
            limit: Maximum results

        Returns:
            List of trade dicts
        """
        rows = self._db.fetch_all(
            """
            SELECT * FROM backtest_trades
            WHERE run_id = ?
            ORDER BY trade_time ASC
            LIMIT ?
            """,
            (run_id, limit),
        )
        return [dict(row) for row in rows]

    # =========================================================================
    # Backtest Metrics
    # =========================================================================

    def save_backtest_metrics(
        self,
        run_id: str,
        total_pnl: float,
        max_drawdown: float,
        win_rate: float,
        sharpe_ratio: float,
        total_trades: int,
        trades_per_day: float,
        capital_utilization: float,
        extra_metrics: dict[str, Any] | None = None,
    ) -> int:
        """Save metrics for a backtest run.

        Args:
            run_id: Backtest run ID
            total_pnl: Total profit/loss
            max_drawdown: Maximum drawdown percentage
            win_rate: Win rate (0-1)
            sharpe_ratio: Sharpe-like ratio
            total_trades: Number of trades
            trades_per_day: Average trades per day
            capital_utilization: Average capital utilization
            extra_metrics: Additional metrics as dict

        Returns:
            Metrics record ID
        """
        metrics_json = json.dumps(extra_metrics) if extra_metrics else None
        cursor = self._db.execute(
            """
            INSERT INTO backtest_metrics
            (run_id, total_pnl, max_drawdown, win_rate, sharpe_ratio,
             total_trades, trades_per_day, capital_utilization, metrics_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                total_pnl,
                max_drawdown,
                win_rate,
                sharpe_ratio,
                total_trades,
                trades_per_day,
                capital_utilization,
                metrics_json,
            ),
        )
        return cursor.lastrowid or 0

    def get_backtest_metrics(self, run_id: str) -> dict[str, Any] | None:
        """Get metrics for a backtest run.

        Args:
            run_id: Backtest run ID

        Returns:
            Metrics dict or None
        """
        row = self._db.fetch_one(
            "SELECT * FROM backtest_metrics WHERE run_id = ?",
            (run_id,),
        )
        if row:
            data = dict(row)
            if data.get("metrics_json"):
                data["extra_metrics"] = json.loads(data["metrics_json"])
            return data
        return None
