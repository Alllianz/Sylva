use super::account_manager::{fetch_unified_accounts_or_simulated, print_unified_accounts_summary};
use super::config::{delete_api_config, get_all_api_configs, get_live_config, init_config_db, insert_api_config, save_live_config, ApiConfig};
use super::exchange::bingx::test_api_connection;
use super::runner::run_live_trading_loop;
use super::trades_db::{count_trades, get_recent_trades, init_trades_db};
use crate::ui::prompts::{pause, prompt_leverage_for_analysis, prompt_timeframe_for_analysis};
use reqwest::Client;
use rusqlite::Connection;
use std::error::Error;
use std::io::{self, Write};

pub async fn show_live_trading_menu(
    conn_candles: &Connection,
    client: &Client,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    let mut conn_cfg = init_config_db("sylva_config.db")?;

    loop {
        println!("\n================================================================================");
        println!("             🔴 SYLVA LIVE TRADING ENGINE (BingX & Paper Trading)");
        println!("================================================================================");
        println!("  [1] 🚀 Iniciar Trading en Vivo (GBDT Allianz - 13 Features Optimizadas)");
        println!("  [2] 🌐 Iniciar Trading en Vivo (Online GBDT General - 32 Features)");
        println!("  [3] 🔑 Configurar API Keys de BingX (Cuentas Reales / Testnet)");
        println!("  [4] 💵 Consulta de Saldos y Cuentas Unificadas");
        println!("  [5] 📡 Test de Conectividad con API de BingX");
        println!("  [6] 🔔 Configurar Webhooks de Discord para Alertas en Vivo");
        println!("  [7] 📜 Ver Historial de Operaciones en Vivo (trades.db)");
        println!("  [0] ↩️ Volver al menú principal");
        print!("\n  👉 Selecciona una opción (0-7) [Por defecto '1']: ");
        io::stdout().flush()?;

        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        let choice = input.trim();

        match choice {
            "0" => break,
            "1" | "" => {
                let tf = prompt_timeframe_for_analysis();
                let leverage = prompt_leverage_for_analysis();

                print!("  💰 Porcentaje de Capital a operar por posición (%) [Por defecto 10.0]: ");
                io::stdout().flush()?;
                let mut cp_in = String::new();
                io::stdin().read_line(&mut cp_in)?;
                let cp = cp_in.trim().parse::<f64>().unwrap_or(10.0).clamp(1.0, 100.0);

                run_live_trading_loop(
                    client,
                    conn_candles,
                    &conn_cfg,
                    &tf,
                    leverage,
                    cp,
                    true,  // GBDT Allianz 13 features
                    0.0005,
                    -0.0005,
                )
                .await?;
                pause();
            }
            "2" => {
                let tf = prompt_timeframe_for_analysis();
                let leverage = prompt_leverage_for_analysis();

                print!("  💰 Porcentaje de Capital a operar por posición (%) [Por defecto 10.0]: ");
                io::stdout().flush()?;
                let mut cp_in = String::new();
                io::stdin().read_line(&mut cp_in)?;
                let cp = cp_in.trim().parse::<f64>().unwrap_or(10.0).clamp(1.0, 100.0);

                run_live_trading_loop(
                    client,
                    conn_candles,
                    &conn_cfg,
                    &tf,
                    leverage,
                    cp,
                    false, // General 32 features
                    0.0005,
                    -0.0005,
                )
                .await?;
                pause();
            }
            "3" => {
                show_api_keys_sub_menu(&mut conn_cfg)?;
            }
            "4" => {
                let configs = get_all_api_configs(&conn_cfg).unwrap_or_default();
                let pool = fetch_unified_accounts_or_simulated(client, &configs).await?;
                print_unified_accounts_summary(&pool);
                pause();
            }
            "5" => {
                let configs = get_all_api_configs(&conn_cfg).unwrap_or_default();
                if configs.is_empty() {
                    println!("\n  ℹ️ No hay API Keys registradas. El motor opera en modo Paper Trading (Simulado).\n");
                } else {
                    println!("\n  📡 Probando conectividad con BingX API:");
                    for cfg in &configs {
                        print!("  • Cuenta #{} ({} | TF: {}): ", cfg.id, cfg.exchange, cfg.timeframe);
                        io::stdout().flush()?;
                        match test_api_connection(client, &cfg.api_key, &cfg.api_secret, cfg.use_testnet).await {
                            Ok(()) => println!("✅ Conexión EXITOSA"),
                            Err(e) => println!("❌ ERROR: {}", e),
                        }
                    }
                }
                pause();
            }
            "6" => {
                let mut live_cfg = get_live_config(&conn_cfg).unwrap_or_default();
                println!("\n  🔔 Configuración de Webhooks de Discord:");
                println!("  • Webhook Actual: {}", if live_cfg.trade_webhook.is_empty() { "Ninguno (Inactivo)" } else { &live_cfg.trade_webhook });
                print!("  👉 Ingrese nuevo URL del Webhook de Discord (o presione ENTER para conservar): ");
                io::stdout().flush()?;
                let mut wh_in = String::new();
                io::stdin().read_line(&mut wh_in)?;
                let wh_trim = wh_in.trim();
                if !wh_trim.is_empty() {
                    live_cfg.trade_webhook = wh_trim.to_string();
                    save_live_config(&conn_cfg, &live_cfg)?;
                    println!("  ✅ Webhook de Discord guardado correctamente.");
                }
                pause();
            }
            "7" => {
                if let Ok(conn_t) = init_trades_db("trades.db") {
                    let total = count_trades(&conn_t, "1H").unwrap_or(0);
                    println!("\n  📜 Historial de Trades en Vivo (trades.db):");
                    println!("  Total registros: {}", total);
                    let trades = get_recent_trades(&conn_t, "1H", 15).unwrap_or_default();
                    if trades.is_empty() {
                        println!("  (No hay trades cerrados registrados aún en trades.db)");
                    } else {
                        println!("{:-<100}", "");
                        println!("{:<4} | {:<6} | {:<10} | {:<10} | {:<10} | {:<12} | {:<8}",
                            "ID", "TF", "Dirección", "Entrada", "Salida", "PnL ($)", "ROE (%)");
                        println!("{:-<100}", "");
                        for (i, t) in trades.iter().enumerate() {
                            println!("#{:<3} | {:<6} | {:<10} | ${:<9.2} | ${:<9.2} | ${:<11.2} | {:+.2}%",
                                i + 1, t.timeframe, t.direction, t.entry_price, t.exit_price, t.pnl_usd, t.roe_percent);
                        }
                    }
                }
                pause();
            }
            _ => {
                println!("  ⚠️ Opción no válida.");
            }
        }
    }

    Ok(())
}

fn show_api_keys_sub_menu(conn: &mut Connection) -> Result<(), Box<dyn Error + Send + Sync>> {
    loop {
        println!("\n--------------------------------------------------------------------------------");
        println!("             🔑 GESTIÓN DE API KEYS (BINGX FUTURES)");
        println!("--------------------------------------------------------------------------------");
        let list = get_all_api_configs(conn)?;
        if list.is_empty() {
            println!("  ℹ️ No hay API Keys configuradas. El sistema opera en Paper Trading automáticamente.");
        } else {
            for cfg in &list {
                let net = if cfg.use_testnet { "TESTNET" } else { "PROD" };
                let masked = if cfg.api_key.len() > 8 {
                    format!("{}...{}", &cfg.api_key[..4], &cfg.api_key[cfg.api_key.len() - 4..])
                } else {
                    cfg.api_key.clone()
                };
                println!("  #{:<2} | TF: {:<5} | Exchange: {:<6} ({}) | Key: {}", cfg.id, cfg.timeframe, cfg.exchange, net, masked);
            }
        }
        println!("\n  [1] ➕ Agregar nueva API Key de BingX");
        println!("  [2] 🗑️ Eliminar una API Key");
        println!("  [0] ↩️ Volver");
        print!("  👉 Selecciona una opción: ");
        io::stdout().flush()?;

        let mut opt = String::new();
        io::stdin().read_line(&mut opt)?;
        match opt.trim() {
            "0" => break,
            "1" => {
                print!("  • Timeframe asignado (ej. 1H, 4H, 1D, o ALL) [Por defecto ALL]: ");
                io::stdout().flush()?;
                let mut tf_in = String::new();
                io::stdin().read_line(&mut tf_in)?;
                let tf = if tf_in.trim().is_empty() { "ALL".to_string() } else { tf_in.trim().to_uppercase() };

                print!("  • Ingrese API Key: ");
                io::stdout().flush()?;
                let mut key_in = String::new();
                io::stdin().read_line(&mut key_in)?;
                let api_key = key_in.trim().to_string();

                print!("  • Ingrese Secret Key: ");
                io::stdout().flush()?;
                let mut sec_in = String::new();
                io::stdin().read_line(&mut sec_in)?;
                let api_secret = sec_in.trim().to_string();

                print!("  • ¿Usar Testnet VST? (s/N): ");
                io::stdout().flush()?;
                let mut tst_in = String::new();
                io::stdin().read_line(&mut tst_in)?;
                let use_testnet = tst_in.trim().eq_ignore_ascii_case("s");

                if !api_key.is_empty() && !api_secret.is_empty() {
                    let cfg = ApiConfig {
                        id: 0,
                        timeframe: tf,
                        api_key,
                        api_secret,
                        leverage: 10,
                        exchange: "BINGX".to_string(),
                        use_testnet,
                    };
                    insert_api_config(conn, &cfg)?;
                    println!("  ✅ API Key registrada correctamente.");
                }
            }
            "2" => {
                print!("  👉 Ingrese el ID (#) de la clave a eliminar: ");
                io::stdout().flush()?;
                let mut id_in = String::new();
                io::stdin().read_line(&mut id_in)?;
                if let Ok(id) = id_in.trim().parse::<u32>() {
                    delete_api_config(conn, id)?;
                    println!("  🗑️ Clave #{} eliminada.", id);
                }
            }
            _ => (),
        }
    }
    Ok(())
}
