use super::config::ApiConfig;
use super::exchange::bingx::{get_account_details, get_open_positions, parse_bingx_position};
use super::types::{AccountBalanceInfo, SignalType, SimulatedPositionState, UnifiedAccountPool};
use reqwest::Client;
use std::collections::HashMap;
use std::error::Error;
use std::sync::{LazyLock, Mutex};

pub static SIMULATED_POSITIONS: LazyLock<Mutex<HashMap<String, SimulatedPositionState>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

pub fn get_simulated_position(tf: &str) -> Option<SimulatedPositionState> {
    if let Ok(map) = SIMULATED_POSITIONS.lock() {
        map.get(&tf.to_uppercase()).cloned()
    } else {
        None
    }
}

pub fn set_simulated_position(tf: &str, pos: Option<SimulatedPositionState>) {
    if let Ok(mut map) = SIMULATED_POSITIONS.lock() {
        if let Some(p) = pos {
            map.insert(tf.to_uppercase(), p);
        } else {
            map.remove(&tf.to_uppercase());
        }
    }
}

pub async fn fetch_unified_accounts_or_simulated(
    client: &Client,
    configs: &[ApiConfig],
) -> Result<UnifiedAccountPool, Box<dyn Error + Send + Sync>> {
    if configs.is_empty() {
        let mut pool = UnifiedAccountPool::new();
        pool.accounts.push(AccountBalanceInfo {
            id: 1,
            timeframe: "ALL".to_string(),
            api_key: "PAPER_TRADING_KEY".to_string(),
            api_secret: "PAPER_TRADING_SECRET".to_string(),
            leverage: 10,
            exchange: "SIMULATED".to_string(),
            use_testnet: false,
            wallet_balance: 10000.0,
            available_margin: 10000.0,
            user_id: "PAPER_ACCOUNT".to_string(),
        });
        pool.total_wallet_balance = 10000.0;
        pool.total_available_margin = 10000.0;
        pool.last_updated = chrono::Utc::now().timestamp();
        Ok(pool)
    } else {
        fetch_unified_accounts(client, configs).await
    }
}

pub async fn fetch_unified_accounts(
    client: &Client,
    configs: &[ApiConfig],
) -> Result<UnifiedAccountPool, Box<dyn Error + Send + Sync>> {
    let mut pool = UnifiedAccountPool::new();
    let mut total_wallet = 0.0;
    let mut total_avail = 0.0;

    for cfg in configs {
        match get_account_details(client, &cfg.api_key, &cfg.api_secret, cfg.use_testnet).await {
            Ok((w_bal, a_margin, uid)) => {
                total_wallet += w_bal;
                total_avail += a_margin;
                pool.accounts.push(AccountBalanceInfo {
                    id: cfg.id,
                    timeframe: cfg.timeframe.clone(),
                    api_key: cfg.api_key.clone(),
                    api_secret: cfg.api_secret.clone(),
                    leverage: cfg.leverage,
                    exchange: cfg.exchange.clone(),
                    use_testnet: cfg.use_testnet,
                    wallet_balance: w_bal,
                    available_margin: a_margin,
                    user_id: uid,
                });
            }
            Err(e) => {
                eprintln!("  ⚠️ No se pudo obtener balance de cuenta #{}: {}", cfg.id, e);
            }
        }
    }

    pool.total_wallet_balance = total_wallet;
    pool.total_available_margin = total_avail;
    pool.last_updated = chrono::Utc::now().timestamp();

    Ok(pool)
}

pub async fn get_pool_position_state(client: &Client, pool: &UnifiedAccountPool, tf: &str) -> SignalType {
    if pool.accounts.iter().all(|a| a.exchange == "SIMULATED" || a.api_key == "PAPER_TRADING_KEY") {
        return match get_simulated_position(tf) {
            Some(pos) if pos.direction.eq_ignore_ascii_case("LONG") => SignalType::Long,
            Some(pos) if pos.direction.eq_ignore_ascii_case("SHORT") => SignalType::Short,
            _ => SignalType::Flat,
        };
    }

    for acc in &pool.accounts {
        if acc.exchange != "SIMULATED" && acc.api_key != "PAPER_TRADING_KEY" {
            if let Ok(positions) = get_open_positions(client, &acc.api_key, &acc.api_secret, "BTC-USDT", acc.use_testnet).await {
                for pos in &positions {
                    let (dir, amt) = parse_bingx_position(pos);
                    if amt > 0.0 {
                        return dir;
                    }
                }
            }
        }
    }

    SignalType::Flat
}

pub fn print_unified_accounts_summary(pool: &UnifiedAccountPool) {
    println!("\n             Resumen de Cuentas y Capital Unificado en Sylva\n");
    println!("---------------------------------------------------------------------------------------------------------");
    println!("{:<4} | {:<10} | {:<12} | {:<9} | {:<10} | {:<14} | {:<12} | Disponible",
        "Pos", "Cuenta", "UID", "Exchange", "Tipo", "API Key", "Balance"
    );
    println!("---------------------------------------------------------------------------------------------------------");
    if pool.accounts.is_empty() {
        println!("No se detectaron cuentas activas con API Key válida o conexión exitosa.");
    } else {
        for (i, acc) in pool.accounts.iter().enumerate() {
            let net_str = if acc.exchange == "SIMULATED" {
                "PAPER/SIM"
            } else if acc.use_testnet {
                "TESTNET"
            } else {
                "PROD"
            };
            let user_str = if acc.user_id.is_empty() { "N/A" } else { &acc.user_id };
            let masked_key = if acc.api_key.len() > 8 {
                format!("{}...{}", &acc.api_key[..4], &acc.api_key[acc.api_key.len() - 4..])
            } else {
                acc.api_key.clone()
            };
            println!(
                "#{:<3} | TF: {:<6} | {:<12} | {:<9} | {:<10} | {:<14} | ${:<11.2} | ${:.2}",
                i + 1, acc.timeframe, user_str, acc.exchange, net_str, masked_key, acc.wallet_balance, acc.available_margin
            );
        }
    }
    println!("---------------------------------------------------------------------------------------------------------");
    println!("Total Capital Unificado: ${:.2} USDT | Margen Total Disponible: ${:.2} USDT\n", pool.total_wallet_balance, pool.total_available_margin);
}
