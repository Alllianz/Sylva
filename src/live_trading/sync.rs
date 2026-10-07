use crate::data::db::{insert_klines_no_overwrite, Kline};
use crate::data::downloader::download_klines;
use chrono::Utc;
use reqwest::Client;
use rusqlite::Connection;

pub fn tf_to_binance_interval(tf: &str) -> &str {
    match tf {
        "1D" | "1d" => "1d",
        "12H" | "12h" => "12h",
        "8H" | "8h" => "8h",
        "4H" | "4h" => "4h",
        "1H" | "1h" => "1h",
        "15M" | "15m" => "15m",
        "5M" | "5m" => "5m",
        "1M" | "1m" => "1m",
        _ => "1h",
    }
}

pub fn tf_to_duration_millis(tf: &str) -> i64 {
    match tf {
        "1m" | "1M" => 60 * 1000,
        "5m" | "5M" => 5 * 60 * 1000,
        "15m" | "15M" => 15 * 60 * 1000,
        "1H" | "1h" => 60 * 60 * 1000,
        "4H" | "4h" => 4 * 60 * 60 * 1000,
        "8H" | "8h" => 8 * 60 * 60 * 1000,
        "12H" | "12h" => 12 * 60 * 60 * 1000,
        "1D" | "1d" => 24 * 60 * 60 * 1000,
        _ => 60 * 60 * 1000,
    }
}

pub async fn sync_all_missing_klines_from_binance(
    client: &Client,
    conn_candles: &Connection,
    tf: &str,
) {
    let tf_duration = tf_to_duration_millis(tf);
    let now_ms = Utc::now().timestamp_millis();
    let current_candle_start = (now_ms / tf_duration) * tf_duration;
    let max_closed_ts = current_candle_start - tf_duration;

    // Limpiar cualquier vela no cerrada o del futuro
    if let Ok(conn_clean) = Connection::open("candles.db") {
        let _ = conn_clean.execute(
            "DELETE FROM klines WHERE timeframe = ?1 AND timestamp > ?2",
            rusqlite::params![tf, max_closed_ts],
        );
    }

    let api_interval = tf_to_binance_interval(tf);
    let symbol = "BTCUSDT";

    let latest_ts: Option<i64> = conn_candles
        .query_row(
            "SELECT MAX(timestamp) FROM klines WHERE timeframe = ?1",
            rusqlite::params![tf],
            |row| row.get(0),
        )
        .ok()
        .flatten();

    let start_ts = latest_ts.map(|ts| ts + 1);

    if let Some(st) = start_ts {
        println!("  📥 {}: Sincronizando velas pendientes desde {}...", tf, st);
    } else {
        println!("  📥 {}: Sincronizando historial de velas desde Binance...", tf);
    }

    let mut current_start = start_ts;
    let mut total_downloaded = 0;

    loop {
        match download_klines(client, symbol, api_interval, current_start, 1000).await {
            Ok(klines) => {
                if klines.is_empty() {
                    break;
                }
                let valid_klines: Vec<Kline> = klines
                    .into_iter()
                    .filter(|k| k.timestamp <= max_closed_ts)
                    .collect();

                if valid_klines.is_empty() {
                    break;
                }

                let count = valid_klines.len();
                total_downloaded += count;

                if let Ok(mut conn_clone) = Connection::open("candles.db") {
                    let _ = insert_klines_no_overwrite(&mut conn_clone, tf, &valid_klines);
                }

                if let Some(last) = valid_klines.last() {
                    current_start = Some(last.timestamp + 1);
                }

                if count < 1000 {
                    break;
                }

                tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
            }
            Err(e) => {
                eprintln!("  ⚠️ {}: Error sincronizando velas: {}", tf, e);
                break;
            }
        }
    }

    println!("  ✅ {}: Sincronización completa ({} velas descargadas/validadas).", tf, total_downloaded);
}

pub async fn sync_recent_klines_from_binance(
    client: &Client,
    conn_candles: &Connection,
    tf: &str,
    target_closed_ts: i64,
) -> bool {
    let api_interval = tf_to_binance_interval(tf);
    let symbol = "BTCUSDT";

    if let Ok(recent_klines) = download_klines(client, symbol, api_interval, None, 20).await {
        let filtered: Vec<Kline> = recent_klines
            .into_iter()
            .filter(|k| k.timestamp <= target_closed_ts)
            .collect();

        if !filtered.is_empty() {
            if let Ok(mut conn_clone) = Connection::open("candles.db") {
                let _ = insert_klines_no_overwrite(&mut conn_clone, tf, &filtered);
            }
        }
    }

    let last_db_ts: Option<i64> = conn_candles
        .query_row(
            "SELECT MAX(timestamp) FROM klines WHERE timeframe = ?1",
            rusqlite::params![tf],
            |row| row.get(0),
        )
        .ok()
        .flatten();

    last_db_ts.unwrap_or(0) >= target_closed_ts
}
