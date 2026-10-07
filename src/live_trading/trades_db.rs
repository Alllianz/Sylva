use super::types::ClosedTradeRecord;
use rusqlite::{params, Connection, Result};

pub fn init_trades_db(db_path: &str) -> Result<Connection> {
    let conn = Connection::open(db_path)?;
    conn.execute(
        "CREATE TABLE IF NOT EXISTS live_trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timeframe TEXT NOT NULL,
            direction TEXT NOT NULL,
            entry_price REAL NOT NULL,
            exit_price REAL NOT NULL,
            entry_time_utc TEXT NOT NULL,
            exit_time_utc TEXT NOT NULL,
            leverage REAL NOT NULL,
            pnl_percent REAL NOT NULL,
            roe_percent REAL NOT NULL,
            capital_percent REAL NOT NULL,
            pnl_usd REAL NOT NULL,
            total_fees REAL NOT NULL,
            capital_before REAL NOT NULL,
            capital_after REAL NOT NULL,
            is_win INTEGER NOT NULL,
            account_id INTEGER NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )",
        [],
    )?;
    Ok(conn)
}

pub fn save_closed_trade(conn: &Connection, trade: &ClosedTradeRecord) -> Result<i64> {
    conn.execute(
        "INSERT INTO live_trades (
            timeframe, direction, entry_price, exit_price, entry_time_utc, exit_time_utc,
            leverage, pnl_percent, roe_percent, capital_percent, pnl_usd, total_fees,
            capital_before, capital_after, is_win, account_id
        ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16)",
        params![
            trade.timeframe,
            trade.direction,
            trade.entry_price,
            trade.exit_price,
            trade.entry_time_utc,
            trade.exit_time_utc,
            trade.leverage,
            trade.pnl_percent,
            trade.roe_percent,
            trade.capital_percent,
            trade.pnl_usd,
            trade.total_fees,
            trade.capital_before,
            trade.capital_after,
            if trade.is_win { 1 } else { 0 },
            trade.account_id
        ],
    )?;
    Ok(conn.last_insert_rowid())
}

pub fn get_recent_trades(conn: &Connection, tf: &str, limit: usize) -> Result<Vec<ClosedTradeRecord>> {
    let mut stmt = conn.prepare(
        "SELECT timeframe, direction, entry_price, exit_price, entry_time_utc, exit_time_utc,
                leverage, pnl_percent, roe_percent, capital_percent, pnl_usd, total_fees,
                capital_before, capital_after, is_win, account_id
         FROM live_trades 
         WHERE timeframe = ?1 COLLATE NOCASE 
         ORDER BY id DESC LIMIT ?2",
    )?;

    let rows = stmt.query_map(params![tf, limit as i64], |row| {
        Ok(ClosedTradeRecord {
            timeframe: row.get(0)?,
            direction: row.get(1)?,
            entry_price: row.get(2)?,
            exit_price: row.get(3)?,
            entry_time_utc: row.get(4)?,
            exit_time_utc: row.get(5)?,
            leverage: row.get(6)?,
            pnl_percent: row.get(7)?,
            roe_percent: row.get(8)?,
            capital_percent: row.get(9)?,
            pnl_usd: row.get(10)?,
            total_fees: row.get(11)?,
            capital_before: row.get(12)?,
            capital_after: row.get(13)?,
            is_win: row.get::<_, i32>(14)? != 0,
            account_id: row.get(15)?,
        })
    })?;

    let mut res = Vec::new();
    for r in rows {
        res.push(r?);
    }
    Ok(res)
}

pub fn count_trades(conn: &Connection, tf: &str) -> Result<usize> {
    let mut stmt = conn.prepare("SELECT COUNT(*) FROM live_trades WHERE timeframe = ?1 COLLATE NOCASE")?;
    let count: i64 = stmt.query_row(params![tf], |row| row.get(0))?;
    Ok(count as usize)
}
