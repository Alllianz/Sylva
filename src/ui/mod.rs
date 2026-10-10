pub mod candles_menu;
pub mod header;
pub mod prompts;
pub mod session;

pub use candles_menu::show_update_candles_menu;
pub use header::print_header;
pub use prompts::{parse_usize_list, pause, prompt_leverage_for_analysis, prompt_timeframe_for_analysis};
pub use session::run_tree_session;

use crate::data::db::{init_candles_db, print_db_summary};
use crate::features::FeatureMask;
use reqwest::Client;
use std::error::Error;
use std::io::{self, Write};

pub async fn run_interactive_app() -> Result<(), Box<dyn Error + Send + Sync>> {
    let db_path = "candles.db";
    let mut conn = init_candles_db(db_path)?;
    let client = Client::builder()
        .timeout(std::time::Duration::from_secs(10))
        .build()
        .unwrap_or_else(|_| Client::new());

    loop {
        print_header();
        println!("  ── 📊 GESTIÓN DE DATOS Y VELAS ──");
        println!("  [1] 📊 Ver Resumen de Velas en Base de Datos (candles.db)");
        println!("  [2] 📥 Descargar/Actualizar Velas desde Binance API");
        println!();
        println!("  ── 🌲 GBDT ALLIANZ (13 Features Optimizadas - Poda Cuantitativa de 19 Variables) ──");
        println!("  [3] 🚀 Online GBDT Convencional (Streaming Hoeffding Trees & Olvido Exponencial)");
        println!("  [4] 🔬 Árboles Bayesianos Online (Normal-Inverse-Gamma & Incertidumbre Epistémica)");
        println!("  [5] 🧬 Auto-Optimización Evolutiva Sylva EVO (Búsqueda Determinista Multihilo)");
        println!("  [6] ⚡ Auto-Optimización Grid Online (Purged Walk-Forward Causal)");
        println!("  [7] 🌲 GBDT Clásico Batch & Grid Search (Purged K-Fold CV & Multi-Grid)");
        println!("  [8] 📈 Elastic Net Regularizado (L1 Lasso + L2 Ridge Benchmark)");
        println!();
        println!("  ── 🌐 MODELOS GENERALES (32 Variables / Selección Libre de Features) ──");
        println!("  [9] 🚀 Online GBDT Convencional General");
        println!("  [10] 🔬 Árboles Bayesianos Online General");
        println!("  [11] 🧬 Auto-Optimización Evolutiva Sylva EVO General");
        println!("  [12] ⚡ Auto-Optimización Grid Online General");
        println!("  [13] 🌲 GBDT Clásico Batch & Grid Search General");
        println!("  [14] 📈 Elastic Net Regularizado General");
        println!();
        println!("  ── 🔴 MOTOR DE TRADING EN VIVO (BINGX & PAPER TRADING) ──");
        println!("  [15] 🔴 Menú y Ejecución de Trading en Vivo (Live Execution BingX / Simulado)");
        println!();
        println!("  [0] 🚪 Salir");
        print!("\n  👉 Selecciona una opción (0-15) [Por defecto '3']: ");
        io::stdout().flush()?;

        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        let choice = input.trim();

        match choice {
            "0" => {
                println!("\n  👋 Finalizando sesión en Sylva. ¡Hasta pronto, Allianz!\n");
                break;
            }
            "1" => {
                print_db_summary(&conn)?;
                pause();
            }
            "2" => {
                show_update_candles_menu(&mut conn, &client).await?;
            }
            "3" | "" => {
                let mask = FeatureMask::new_allianz_custom();
                run_tree_session(&conn, Some((mask, "GBDT Allianz (13 Features)")), "1")?;
                pause();
            }
            "4" => {
                let mask = FeatureMask::new_allianz_custom();
                run_tree_session(&conn, Some((mask, "GBDT Allianz (13 Features)")), "2")?;
                pause();
            }
            "5" => {
                let mask = FeatureMask::new_allianz_custom();
                run_tree_session(&conn, Some((mask, "GBDT Allianz (13 Features)")), "3")?;
                pause();
            }
            "6" => {
                let mask = FeatureMask::new_allianz_custom();
                run_tree_session(&conn, Some((mask, "GBDT Allianz (13 Features)")), "4")?;
                pause();
            }
            "7" => {
                let mask = FeatureMask::new_allianz_custom();
                run_tree_session(&conn, Some((mask, "GBDT Allianz (13 Features)")), "5")?;
                pause();
            }
            "8" => {
                let mask = FeatureMask::new_allianz_custom();
                run_tree_session(&conn, Some((mask, "GBDT Allianz (13 Features)")), "6")?;
                pause();
            }
            "9" => {
                run_tree_session(&conn, None, "1")?;
                pause();
            }
            "10" => {
                run_tree_session(&conn, None, "2")?;
                pause();
            }
            "11" => {
                run_tree_session(&conn, None, "3")?;
                pause();
            }
            "12" => {
                run_tree_session(&conn, None, "4")?;
                pause();
            }
            "13" => {
                run_tree_session(&conn, None, "5")?;
                pause();
            }
            "14" => {
                run_tree_session(&conn, None, "6")?;
                pause();
            }
            "15" => {
                crate::live_trading::show_live_trading_menu(&conn, &client).await?;
            }
            _ => {
                println!("  ⚠️ Opción no válida. Inténtalo de nuevo.");
            }
        }
    }

    Ok(())
}
