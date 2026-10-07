use crate::data::db::get_candle_count_by_tf;
use crate::data::downloader::{update_single_timeframe, UPDATE_TIMEFRAMES};
use crate::ui::prompts::pause;
use reqwest::Client;
use rusqlite::Connection;
use std::error::Error;
use std::io::{self, Write};

pub async fn show_update_candles_menu(
    conn: &mut Connection,
    client: &Client,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    loop {
        println!("\n             Actualizar Velas desde Binance API (candles.db)\n");
        for (idx, (db_tf, _)) in UPDATE_TIMEFRAMES.iter().enumerate() {
            let count = get_candle_count_by_tf(conn, db_tf).unwrap_or(0);
            println!("  {}) {}  ({} registros actuales)", idx + 1, db_tf, count);
        }
        println!("  7) Actualizar TODAS las temporalidades");
        println!("  0) Volver al menú principal");
        print!("\n  Seleccione la temporalidad a actualizar (1-6, 7 o nombre como '15m'): ");
        io::stdout().flush()?;

        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        let choice = input.trim();

        if choice == "0" || choice.is_empty() {
            break;
        }

        if choice == "7" {
            println!("\nIniciando actualización de TODAS las temporalidades disponibles...");
            for (db_tf, api_tf) in UPDATE_TIMEFRAMES {
                update_single_timeframe(conn, client, db_tf, api_tf).await?;
            }
            println!("\n✅ Sincronización completa de todas las temporalidades finalizada.");
            pause();
            continue;
        }

        let selected = match choice {
            "1" => Some(("1D", "1d")),
            "2" => Some(("4H", "4h")),
            "3" => Some(("1H", "1h")),
            "4" => Some(("15m", "15m")),
            "5" => Some(("5m", "5m")),
            "6" => Some(("1m", "1m")),
            other => UPDATE_TIMEFRAMES
                .iter()
                .copied()
                .find(|&(db_t, _)| db_t.eq_ignore_ascii_case(other)),
        };

        if let Some((db_tf, api_tf)) = selected {
            println!("\nSincronizando únicamente la temporalidad {}...", db_tf);
            update_single_timeframe(conn, client, db_tf, api_tf).await?;
            println!("✅ Actualización de {} finalizada.", db_tf);
            pause();
        } else {
            println!("⚠️ Opción inválida.");
        }
    }

    Ok(())
}
