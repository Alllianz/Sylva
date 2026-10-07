use crate::data::db::{insert_klines, Kline};
use reqwest::Client;
use rusqlite::Connection;
use serde::Deserialize;
use std::error::Error;

#[allow(dead_code)]
#[derive(Debug, Deserialize)]
struct BinanceKlineTuple(
    i64,    // 0: Open time
    String, // 1: Open
    String, // 2: High
    String, // 3: Low
    String, // 4: Close
    String, // 5: Volume
    i64,    // 6: Close time
    String, // 7: Quote asset volume
    i64,    // 8: Number of trades
    String, // 9: Taker buy base asset volume
    String, // 10: Taker buy quote asset volume
    String, // 11: Ignore
);

pub async fn download_klines(
    client: &Client,
    symbol: &str,
    interval: &str,
    start_time: Option<i64>,
    limit: u32,
) -> Result<Vec<Kline>, Box<dyn Error + Send + Sync>> {
    let base_urls = [
        "https://api.binance.com",
        "https://api1.binance.com",
        "https://api2.binance.com",
        "https://api3.binance.com",
        "https://api-gcp.binance.com",
    ];

    let mut last_err = None;

    for &base_url in &base_urls {
        let mut url = format!(
            "{}/api/v3/klines?symbol={}&interval={}&limit={}",
            base_url, symbol, interval, limit
        );

        if let Some(st) = start_time {
            url = format!("{}&startTime={}", url, st);
        }

        match client.get(&url).send().await {
            Ok(res) => match res.text().await {
                Ok(text_res) => match serde_json::from_str::<Vec<BinanceKlineTuple>>(&text_res) {
                    Ok(data) => {
                        let now_ms = chrono::Utc::now().timestamp_millis();
                        let klines: Vec<Kline> = data
                            .into_iter()
                            .filter(|t| t.6 <= now_ms)
                            .map(|t| Kline {
                                timestamp: t.0,
                                open: t.1.parse().unwrap_or(0.0),
                                high: t.2.parse().unwrap_or(0.0),
                                low: t.3.parse().unwrap_or(0.0),
                                close: t.4.parse().unwrap_or(0.0),
                                volume: t.5.parse().unwrap_or(0.0),
                            })
                            .collect();
                        return Ok(klines);
                    }
                    Err(e) => {
                        let err_msg = format!(
                            "Error deserializando respuesta de {}: {}. Respuesta: {}",
                            base_url, e, text_res
                        );
                        last_err = Some(err_msg);
                    }
                },
                Err(e) => {
                    last_err = Some(format!("Error leyendo texto de respuesta de {}: {}", base_url, e));
                }
            },
            Err(e) => {
                last_err = Some(format!("Error de conexión con {}: {}", base_url, e));
            }
        }

        tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;
    }

    Err(Box::new(std::io::Error::other(format!(
        "Todos los endpoints de Binance fallaron al descargar klines. Último error: {:?}",
        last_err
    ))))
}

pub const UPDATE_TIMEFRAMES: [(&str, &str); 6] = [
    ("1D", "1d"),
    ("4H", "4h"),
    ("1H", "1h"),
    ("15m", "15m"),
    ("5m", "5m"),
    ("1m", "1m"),
];

/// Sincroniza progresivamente una única temporalidad desde la última vela registrada o desde 2017-08-17
pub async fn update_single_timeframe(
    conn: &mut Connection,
    client: &Client,
    db_tf: &str,
    api_tf: &str,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    let symbol = "BTCUSDT";
    println!("Actualizando velas en {}...", db_tf);

    loop {
        let last_ts = crate::data::db::get_latest_timestamp(conn, db_tf)?;

        let start_time = match last_ts {
            Some(ts) => Some(ts + 1),
            None => {
                use chrono::{TimeZone, Utc};
                Some(Utc.with_ymd_and_hms(2017, 8, 17, 0, 0, 0).unwrap().timestamp_millis())
            }
        };

        let klines = download_klines(client, symbol, api_tf, start_time, 1000).await?;
        if klines.is_empty() {
            println!("- {} está completamente actualizado.", db_tf);
            break;
        }

        let len = klines.len();
        insert_klines(conn, db_tf, &klines)?;

        if let Some(last) = klines.last() {
            use chrono::{TimeZone, Utc};
            let date = Utc.timestamp_millis_opt(last.timestamp).unwrap();
            println!("  Guardadas {} velas de {}. Última: {}", len, db_tf, date.format("%Y-%m-%d %H:%M:%S"));
        }

        if len < 1000 {
            println!("- {} está completamente actualizado.", db_tf);
            break;
        }

        tokio::time::sleep(tokio::time::Duration::from_millis(150)).await;
    }

    Ok(())
}
