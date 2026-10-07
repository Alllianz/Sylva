pub use rusqlite::{params, Connection, Result};
use chrono::{TimeZone, Utc};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Kline {
    pub timestamp: i64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}

pub fn init_candles_db(db_path: &str) -> Result<Connection> {
    let conn = Connection::open(db_path)?;
    let _ = conn.pragma_update(None, "journal_mode", "WAL");
    let _ = conn.pragma_update(None, "synchronous", "NORMAL");
    conn.busy_timeout(std::time::Duration::from_secs(60))?;
    conn.execute(
        "CREATE TABLE IF NOT EXISTS klines (
            timeframe TEXT NOT NULL,
            timestamp INTEGER NOT NULL,
            open REAL NOT NULL,
            high REAL NOT NULL,
            low REAL NOT NULL,
            close REAL NOT NULL,
            volume REAL NOT NULL,
            PRIMARY KEY (timeframe, timestamp)
        )",
        [],
    )?;
    Ok(conn)
}

pub fn normalize_timeframe(tf: &str) -> String {
    let clean = tf.trim();
    if clean.is_empty() {
        return "1D".to_string();
    }
    let lower = clean.to_lowercase();
    if lower.ends_with('m') {
        lower
    } else if lower.ends_with('h') || lower.ends_with('d') || lower.ends_with('w') {
        clean.to_uppercase()
    } else {
        clean.to_uppercase()
    }
}

pub fn tf_to_duration_millis(tf: &str) -> i64 {
    let lower = tf.to_lowercase();
    match lower.as_str() {
        "1m" => 60 * 1000,
        "3m" => 3 * 60 * 1000,
        "5m" => 5 * 60 * 1000,
        "15m" => 15 * 60 * 1000,
        "30m" => 30 * 60 * 1000,
        "1h" => 60 * 60 * 1000,
        "2h" => 2 * 60 * 60 * 1000,
        "4h" => 4 * 60 * 60 * 1000,
        "6h" => 6 * 60 * 60 * 1000,
        "8h" => 8 * 60 * 60 * 1000,
        "12h" => 12 * 60 * 60 * 1000,
        "1d" => 24 * 60 * 60 * 1000,
        _ => 24 * 60 * 60 * 1000,
    }
}

pub fn insert_klines(conn: &mut Connection, tf: &str, klines: &[Kline]) -> Result<()> {
    let normalized_tf = normalize_timeframe(tf);
    let tx = conn.transaction()?;
    {
        let mut stmt = tx.prepare(
            "INSERT OR REPLACE INTO klines (timeframe, timestamp, open, high, low, close, volume)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
        )?;
        for k in klines {
            stmt.execute(params![
                normalized_tf,
                k.timestamp,
                k.open,
                k.high,
                k.low,
                k.close,
                k.volume,
            ])?;
        }
    }
    tx.commit()?;
    Ok(())
}

pub fn insert_klines_no_overwrite(conn: &mut Connection, tf: &str, klines: &[Kline]) -> Result<usize> {
    let normalized_tf = normalize_timeframe(tf);
    let tx = conn.transaction()?;
    let mut inserted = 0;
    {
        let mut stmt = tx.prepare(
            "INSERT OR IGNORE INTO klines (timeframe, timestamp, open, high, low, close, volume)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
        )?;
        for k in klines {
            inserted += stmt.execute(params![
                normalized_tf,
                k.timestamp,
                k.open,
                k.high,
                k.low,
                k.close,
                k.volume,
            ])?;
        }
    }
    tx.commit()?;
    Ok(inserted)
}

pub fn get_latest_timestamp(conn: &Connection, tf: &str) -> Result<Option<i64>> {
    let normalized_tf = normalize_timeframe(tf);
    let mut stmt = conn.prepare("SELECT MAX(timestamp) FROM klines WHERE timeframe = ?1 COLLATE NOCASE")?;
    let mut rows = stmt.query(params![normalized_tf])?;
    if let Some(row) = rows.next()? {
        let ts: rusqlite::types::ValueRef<'_> = row.get_ref(0)?;
        if ts != rusqlite::types::ValueRef::Null {
            return Ok(Some(row.get(0)?));
        }
    }
    Ok(None)
}

pub fn get_earliest_timestamp(conn: &Connection, tf: &str) -> Result<Option<i64>> {
    let normalized_tf = normalize_timeframe(tf);
    let mut stmt = conn.prepare("SELECT MIN(timestamp) FROM klines WHERE timeframe = ?1 COLLATE NOCASE")?;
    let mut rows = stmt.query(params![normalized_tf])?;
    if let Some(row) = rows.next()? {
        let ts: rusqlite::types::ValueRef<'_> = row.get_ref(0)?;
        if ts != rusqlite::types::ValueRef::Null {
            return Ok(Some(row.get(0)?));
        }
    }
    Ok(None)
}

pub fn get_candle_count_by_tf(conn: &Connection, tf: &str) -> Result<i64> {
    let normalized_tf = normalize_timeframe(tf);
    let mut stmt = conn.prepare("SELECT COUNT(*) FROM klines WHERE timeframe = ?1 COLLATE NOCASE")?;
    let mut rows = stmt.query(params![normalized_tf])?;
    if let Some(row) = rows.next()? {
        Ok(row.get(0)?)
    } else {
        Ok(0)
    }
}

pub fn get_all_klines_closed_only(conn: &Connection, tf: &str) -> Result<Vec<Kline>> {
    let normalized_tf = normalize_timeframe(tf);
    let now_ms = Utc::now().timestamp_millis();
    let tf_duration = tf_to_duration_millis(&normalized_tf);
    let max_closed_open_ts = (now_ms / tf_duration) * tf_duration - tf_duration;
    let mut stmt = conn.prepare(
        "SELECT timestamp, open, high, low, close, volume FROM klines WHERE timeframe = ?1 COLLATE NOCASE AND timestamp <= ?2 ORDER BY timestamp ASC"
    )?;
    let klines = stmt.query_map(params![normalized_tf, max_closed_open_ts], |row| {
        Ok(Kline {
            timestamp: row.get(0)?,
            open: row.get(1)?,
            high: row.get(2)?,
            low: row.get(3)?,
            close: row.get(4)?,
            volume: row.get(5)?,
        })
    })?
    .collect::<Result<Vec<_>, _>>()?;
    Ok(klines)
}

pub fn print_db_summary(conn: &Connection) -> Result<()> {
    println!("\n=== RESUMEN DE LA BASE DE DATOS DE VELAS (candles.db) ===");
    for tf in ["1D", "12H", "8H", "4H", "1H", "15m", "5m", "1m"] {
        let count = get_candle_count_by_tf(conn, tf)?;
        if count > 0 {
            let min_ts = get_earliest_timestamp(conn, tf)?.unwrap_or(0);
            let max_ts = get_latest_timestamp(conn, tf)?.unwrap_or(0);
            let min_date = Utc.timestamp_millis_opt(min_ts).unwrap();
            let max_date = Utc.timestamp_millis_opt(max_ts).unwrap();
            println!(
                " - {:<5}: {:>8} registros (Desde {} hasta {})",
                tf,
                count,
                min_date.format("%Y-%m-%d %H:%M:%S"),
                max_date.format("%Y-%m-%d %H:%M:%S")
            );
        } else {
            println!(" - {:<5}:        0 registros", tf);
        }
    }
    println!("=========================================================\n");
    Ok(())
}
