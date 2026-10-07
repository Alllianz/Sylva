use rusqlite::{params, Connection, Result};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiConfig {
    pub id: u32,
    pub timeframe: String,
    pub api_key: String,
    pub api_secret: String,
    pub leverage: u32,
    pub exchange: String,
    pub use_testnet: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LiveEngineConfig {
    pub capital_percent: f64,
    pub leverage: f64,
    pub trade_webhook: String,
    pub status_webhook: String,
    pub server_name: String,
    pub status_interval_minutes: u64,
}

impl Default for LiveEngineConfig {
    fn default() -> Self {
        Self {
            capital_percent: 10.0,
            leverage: 10.0,
            trade_webhook: String::new(),
            status_webhook: String::new(),
            server_name: "Sylva-Live-Engine".to_string(),
            status_interval_minutes: 60,
        }
    }
}

pub fn init_config_db(db_path: &str) -> Result<Connection> {
    let conn = Connection::open(db_path)?;
    conn.execute(
        "CREATE TABLE IF NOT EXISTS api_config (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timeframe TEXT NOT NULL,
            api_key TEXT NOT NULL,
            api_secret TEXT NOT NULL,
            leverage INTEGER NOT NULL DEFAULT 10,
            exchange TEXT NOT NULL DEFAULT 'BINGX',
            use_testnet INTEGER NOT NULL DEFAULT 0,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )",
        [],
    )?;

    conn.execute(
        "CREATE TABLE IF NOT EXISTS live_config (
            id INTEGER PRIMARY KEY,
            capital_percent REAL NOT NULL DEFAULT 10.0,
            leverage REAL NOT NULL DEFAULT 10.0,
            trade_webhook TEXT NOT NULL DEFAULT '',
            status_webhook TEXT NOT NULL DEFAULT '',
            server_name TEXT NOT NULL DEFAULT 'Sylva-Live-Engine',
            status_interval_minutes INTEGER NOT NULL DEFAULT 60
        )",
        [],
    )?;

    Ok(conn)
}

pub fn get_live_config(conn: &Connection) -> Result<LiveEngineConfig> {
    let mut stmt = conn.prepare(
        "SELECT capital_percent, leverage, trade_webhook, status_webhook, server_name, status_interval_minutes 
         FROM live_config WHERE id = 1",
    )?;
    let mut rows = stmt.query([])?;
    if let Some(row) = rows.next()? {
        Ok(LiveEngineConfig {
            capital_percent: row.get(0)?,
            leverage: row.get(1)?,
            trade_webhook: row.get(2)?,
            status_webhook: row.get(3)?,
            server_name: row.get(4).unwrap_or_else(|_| "Sylva-Live-Engine".to_string()),
            status_interval_minutes: row.get::<_, i64>(5).unwrap_or(60).max(1) as u64,
        })
    } else {
        Ok(LiveEngineConfig::default())
    }
}

pub fn save_live_config(conn: &Connection, cfg: &LiveEngineConfig) -> Result<()> {
    conn.execute(
        "INSERT OR REPLACE INTO live_config (id, capital_percent, leverage, trade_webhook, status_webhook, server_name, status_interval_minutes)
         VALUES (1, ?1, ?2, ?3, ?4, ?5, ?6)",
        params![
            cfg.capital_percent,
            cfg.leverage,
            cfg.trade_webhook,
            cfg.status_webhook,
            cfg.server_name,
            cfg.status_interval_minutes as i64
        ],
    )?;
    Ok(())
}

pub fn get_all_api_configs(conn: &Connection) -> Result<Vec<ApiConfig>> {
    let mut stmt = conn.prepare(
        "SELECT id, timeframe, api_key, api_secret, leverage, exchange, use_testnet FROM api_config ORDER BY id ASC",
    )?;
    let rows = stmt.query_map([], |row| {
        Ok(ApiConfig {
            id: row.get(0)?,
            timeframe: row.get(1)?,
            api_key: row.get(2)?,
            api_secret: row.get(3)?,
            leverage: row.get(4)?,
            exchange: row.get(5)?,
            use_testnet: row.get::<_, i32>(6)? != 0,
        })
    })?;

    let mut res = Vec::new();
    for r in rows {
        res.push(r?);
    }
    Ok(res)
}

pub fn insert_api_config(conn: &Connection, cfg: &ApiConfig) -> Result<u32> {
    conn.execute(
        "INSERT INTO api_config (timeframe, api_key, api_secret, leverage, exchange, use_testnet)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
        params![
            cfg.timeframe,
            cfg.api_key,
            cfg.api_secret,
            cfg.leverage,
            cfg.exchange,
            if cfg.use_testnet { 1 } else { 0 }
        ],
    )?;
    Ok(conn.last_insert_rowid() as u32)
}

pub fn delete_api_config(conn: &Connection, id: u32) -> Result<()> {
    conn.execute("DELETE FROM api_config WHERE id = ?1", params![id])?;
    Ok(())
}
