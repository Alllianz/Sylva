use super::account_manager::{get_simulated_position, set_simulated_position};
use super::discord::send_trade_alert_webhook;
use super::exchange::bingx::{get_open_positions, parse_bingx_position, place_market_order, set_leverage};
use super::trades_db::{init_trades_db, save_closed_trade};
use super::types::{AccountBalanceInfo, ClosedTradeRecord, LiveSignalEvaluation, SignalType, SimulatedPositionState, UnifiedAccountPool};
use chrono::Utc;
use reqwest::Client;
use std::error::Error;

pub async fn execute_unified_signal(
    client: &Client,
    pool: &UnifiedAccountPool,
    tf: &str,
    eval: &LiveSignalEvaluation,
    capital_percent: f64,
    leverage: f64,
    trade_webhook: &str,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    if pool.accounts.is_empty() {
        return Ok(());
    }

    let matching_accounts: Vec<_> = pool
        .accounts
        .iter()
        .filter(|acc| {
            acc.timeframe.eq_ignore_ascii_case(tf)
                || acc.timeframe.eq_ignore_ascii_case("ALL")
                || acc.timeframe.is_empty()
        })
        .collect();

    let target_accounts = if matching_accounts.is_empty() {
        pool.accounts.iter().collect::<Vec<_>>()
    } else {
        matching_accounts
    };

    for acc in target_accounts {
        if let Err(e) = execute_account_signal(
            client,
            acc,
            tf,
            eval,
            capital_percent,
            leverage,
            trade_webhook,
        )
        .await
        {
            eprintln!("  ⚠️ Error al ejecutar orden en cuenta #{}: {}", acc.id, e);
        }
    }

    Ok(())
}

async fn execute_account_signal(
    client: &Client,
    acc: &AccountBalanceInfo,
    tf: &str,
    eval: &LiveSignalEvaluation,
    capital_percent: f64,
    leverage: f64,
    trade_webhook: &str,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    if acc.exchange == "SIMULATED" || acc.api_key == "PAPER_TRADING_KEY" {
        return execute_simulated_account_signal(
            client,
            acc,
            tf,
            eval,
            capital_percent,
            leverage,
            trade_webhook,
        )
        .await;
    }

    // 0. Sincronizar apalancamiento en BingX Futures
    let lev_u32 = leverage.round().max(1.0) as u32;
    let _ = set_leverage(client, &acc.api_key, &acc.api_secret, "BTC-USDT", lev_u32, "LONG", acc.use_testnet).await;
    let _ = set_leverage(client, &acc.api_key, &acc.api_secret, "BTC-USDT", lev_u32, "SHORT", acc.use_testnet).await;

    // 1. Obtener posición actual en BingX
    let open_positions = get_open_positions(client, &acc.api_key, &acc.api_secret, "BTC-USDT", acc.use_testnet).await?;
    let mut current_direction = SignalType::Flat;
    let mut current_amount = 0.0;

    for pos in &open_positions {
        let (dir, amt) = parse_bingx_position(pos);
        if amt > 0.0 {
            current_direction = dir;
            current_amount = amt;
            break;
        }
    }

    let base_bal = if acc.wallet_balance > 0.0 { acc.wallet_balance } else { 1000.0 };
    let margin_allocated = base_bal * (capital_percent / 100.0);
    let notional = margin_allocated * leverage;
    let order_qty = ((notional / eval.last_price).max(0.0001) * 10000.0).round() / 10000.0;

    match eval.signal {
        SignalType::Long => {
            if current_direction == SignalType::Short {
                // Cerrar Short previo
                let _ = place_market_order(client, &acc.api_key, &acc.api_secret, "BTC-USDT", "BUY", "SHORT", current_amount, acc.use_testnet).await?;
                println!("  🔄 BingX: Cierre SHORT por reversión (Qty: {:.4} BTC)", current_amount);
            }
            if current_direction != SignalType::Long {
                // Abrir Long
                let _ = place_market_order(client, &acc.api_key, &acc.api_secret, "BTC-USDT", "BUY", "LONG", order_qty, acc.use_testnet).await?;
                println!("  🚀 BingX: Orden LONG ejecutada (Qty: {:.4} BTC | ${:.2} | Lev: {:.0}X)", order_qty, eval.last_price, leverage);
                send_trade_alert_webhook(
                    client,
                    trade_webhook,
                    tf,
                    "ABRIR_LONG",
                    "BTC-USDT",
                    eval.last_price,
                    order_qty,
                    notional,
                    leverage,
                    acc.id,
                    &format!("Alpha: {:.4} | DynThr: {:.4}", eval.p_long, eval.dynamic_thr),
                )
                .await;
            }
        }
        SignalType::Short => {
            if current_direction == SignalType::Long {
                // Cerrar Long previo
                let _ = place_market_order(client, &acc.api_key, &acc.api_secret, "BTC-USDT", "SELL", "LONG", current_amount, acc.use_testnet).await?;
                println!("  🔄 BingX: Cierre LONG por reversión (Qty: {:.4} BTC)", current_amount);
            }
            if current_direction != SignalType::Short {
                // Abrir Short
                let _ = place_market_order(client, &acc.api_key, &acc.api_secret, "BTC-USDT", "SELL", "SHORT", order_qty, acc.use_testnet).await?;
                println!("  🔻 BingX: Orden SHORT ejecutada (Qty: {:.4} BTC | ${:.2} | Lev: {:.0}X)", order_qty, eval.last_price, leverage);
                send_trade_alert_webhook(
                    client,
                    trade_webhook,
                    tf,
                    "ABRIR_SHORT",
                    "BTC-USDT",
                    eval.last_price,
                    order_qty,
                    notional,
                    leverage,
                    acc.id,
                    &format!("Alpha: {:.4} | DynThr: {:.4}", eval.p_short, eval.dynamic_thr),
                )
                .await;
            }
        }
        SignalType::Flat => {
            if current_direction == SignalType::Long {
                let _ = place_market_order(client, &acc.api_key, &acc.api_secret, "BTC-USDT", "SELL", "LONG", current_amount, acc.use_testnet).await?;
                println!("  ⏹️ BingX: Cierre LONG a FLAT (Qty: {:.4} BTC)", current_amount);
            } else if current_direction == SignalType::Short {
                let _ = place_market_order(client, &acc.api_key, &acc.api_secret, "BTC-USDT", "BUY", "SHORT", current_amount, acc.use_testnet).await?;
                println!("  ⏹️ BingX: Cierre SHORT a FLAT (Qty: {:.4} BTC)", current_amount);
            }
        }
    }

    Ok(())
}

async fn execute_simulated_account_signal(
    client: &Client,
    acc: &AccountBalanceInfo,
    tf: &str,
    eval: &LiveSignalEvaluation,
    capital_percent: f64,
    leverage: f64,
    trade_webhook: &str,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    let current_pos_opt = get_simulated_position(tf);
    let current_pos_side = match &current_pos_opt {
        Some(p) if p.direction.eq_ignore_ascii_case("LONG") => SignalType::Long,
        Some(p) if p.direction.eq_ignore_ascii_case("SHORT") => SignalType::Short,
        _ => SignalType::Flat,
    };

    let base_bal = if acc.wallet_balance > 0.0 { acc.wallet_balance } else { 10000.0 };
    let margin_allocated = base_bal * (capital_percent / 100.0);
    let notional = margin_allocated * leverage;
    let order_qty = ((notional / eval.last_price).max(0.0001) * 10000.0).round() / 10000.0;

    let fee_rate = 0.0005; // 0.05% taker

    match eval.signal {
        SignalType::Long => {
            if current_pos_side == SignalType::Short {
                // Cerrar Short previo
                if let Some(pos) = current_pos_opt {
                    let gross_pnl = (pos.entry_price - eval.last_price) * pos.amount;
                    let fees = pos.amount * (pos.entry_price + eval.last_price) * fee_rate;
                    let net_pnl = gross_pnl - fees;
                    let initial_margin = (pos.amount * pos.entry_price) / pos.leverage.max(1.0);
                    let roe_pct = if initial_margin > 0.0 { (net_pnl / initial_margin) * 100.0 } else { 0.0 };

                    let rec = ClosedTradeRecord {
                        timeframe: tf.to_string(),
                        direction: "SHORT".to_string(),
                        entry_price: pos.entry_price,
                        exit_price: eval.last_price,
                        entry_time_utc: pos.entry_time_utc,
                        exit_time_utc: Utc::now().format("%Y-%m-%d %H:%M:%S UTC").to_string(),
                        leverage: pos.leverage,
                        pnl_percent: roe_pct * (pos.capital_percent / 100.0),
                        roe_percent: roe_pct,
                        capital_percent: pos.capital_percent,
                        pnl_usd: net_pnl,
                        total_fees: fees,
                        capital_before: base_bal,
                        capital_after: base_bal + net_pnl,
                        is_win: net_pnl > 0.0,
                        account_id: acc.id,
                    };

                    if let Ok(conn_t) = init_trades_db("trades.db") {
                        let _ = save_closed_trade(&conn_t, &rec);
                    }

                    println!(
                        "  🔄 [SIMULADO] Cierre SHORT previo: PnL: ${:.2} (ROE: {:.2}%)",
                        net_pnl, roe_pct
                    );
                }
                set_simulated_position(tf, None);
            }

            if current_pos_side != SignalType::Long {
                set_simulated_position(
                    tf,
                    Some(SimulatedPositionState {
                        timeframe: tf.to_string(),
                        direction: "LONG".to_string(),
                        entry_price: eval.last_price,
                        amount: order_qty,
                        leverage,
                        capital_percent,
                        open_timestamp_ms: Utc::now().timestamp_millis(),
                        entry_time_utc: Utc::now().format("%Y-%m-%d %H:%M:%S UTC").to_string(),
                    }),
                );

                println!(
                    "  🚀 [SIMULADO] Posición LONG abierta (Qty: {:.4} BTC | ${:.2} | Lev: {:.0}X | Cap: {:.1}%)",
                    order_qty, eval.last_price, leverage, capital_percent
                );

                send_trade_alert_webhook(
                    client,
                    trade_webhook,
                    tf,
                    "ABRIR_LONG (SIMULADO)",
                    "BTC-USDT",
                    eval.last_price,
                    order_qty,
                    notional,
                    leverage,
                    acc.id,
                    "Señal LONG detectada en Sylva Trees Engine",
                )
                .await;
            }
        }
        SignalType::Short => {
            if current_pos_side == SignalType::Long {
                // Cerrar Long previo
                if let Some(pos) = current_pos_opt {
                    let gross_pnl = (eval.last_price - pos.entry_price) * pos.amount;
                    let fees = pos.amount * (pos.entry_price + eval.last_price) * fee_rate;
                    let net_pnl = gross_pnl - fees;
                    let initial_margin = (pos.amount * pos.entry_price) / pos.leverage.max(1.0);
                    let roe_pct = if initial_margin > 0.0 { (net_pnl / initial_margin) * 100.0 } else { 0.0 };

                    let rec = ClosedTradeRecord {
                        timeframe: tf.to_string(),
                        direction: "LONG".to_string(),
                        entry_price: pos.entry_price,
                        exit_price: eval.last_price,
                        entry_time_utc: pos.entry_time_utc,
                        exit_time_utc: Utc::now().format("%Y-%m-%d %H:%M:%S UTC").to_string(),
                        leverage: pos.leverage,
                        pnl_percent: roe_pct * (pos.capital_percent / 100.0),
                        roe_percent: roe_pct,
                        capital_percent: pos.capital_percent,
                        pnl_usd: net_pnl,
                        total_fees: fees,
                        capital_before: base_bal,
                        capital_after: base_bal + net_pnl,
                        is_win: net_pnl > 0.0,
                        account_id: acc.id,
                    };

                    if let Ok(conn_t) = init_trades_db("trades.db") {
                        let _ = save_closed_trade(&conn_t, &rec);
                    }

                    println!(
                        "  🔄 [SIMULADO] Cierre LONG previo: PnL: ${:.2} (ROE: {:.2}%)",
                        net_pnl, roe_pct
                    );
                }
                set_simulated_position(tf, None);
            }

            if current_pos_side != SignalType::Short {
                set_simulated_position(
                    tf,
                    Some(SimulatedPositionState {
                        timeframe: tf.to_string(),
                        direction: "SHORT".to_string(),
                        entry_price: eval.last_price,
                        amount: order_qty,
                        leverage,
                        capital_percent,
                        open_timestamp_ms: Utc::now().timestamp_millis(),
                        entry_time_utc: Utc::now().format("%Y-%m-%d %H:%M:%S UTC").to_string(),
                    }),
                );

                println!(
                    "  🔻 [SIMULADO] Posición SHORT abierta (Qty: {:.4} BTC | ${:.2} | Lev: {:.0}X | Cap: {:.1}%)",
                    order_qty, eval.last_price, leverage, capital_percent
                );

                send_trade_alert_webhook(
                    client,
                    trade_webhook,
                    tf,
                    "ABRIR_SHORT (SIMULADO)",
                    "BTC-USDT",
                    eval.last_price,
                    order_qty,
                    notional,
                    leverage,
                    acc.id,
                    "Señal SHORT detectada en Sylva Trees Engine",
                )
                .await;
            }
        }
        SignalType::Flat => {
            if let Some(pos) = current_pos_opt {
                let (gross_pnl, dir_str) = if pos.direction.eq_ignore_ascii_case("LONG") {
                    ((eval.last_price - pos.entry_price) * pos.amount, "LONG")
                } else {
                    ((pos.entry_price - eval.last_price) * pos.amount, "SHORT")
                };
                let fees = pos.amount * (pos.entry_price + eval.last_price) * fee_rate;
                let net_pnl = gross_pnl - fees;
                let initial_margin = (pos.amount * pos.entry_price) / pos.leverage.max(1.0);
                let roe_pct = if initial_margin > 0.0 { (net_pnl / initial_margin) * 100.0 } else { 0.0 };

                let rec = ClosedTradeRecord {
                    timeframe: tf.to_string(),
                    direction: dir_str.to_string(),
                    entry_price: pos.entry_price,
                    exit_price: eval.last_price,
                    entry_time_utc: pos.entry_time_utc,
                    exit_time_utc: Utc::now().format("%Y-%m-%d %H:%M:%S UTC").to_string(),
                    leverage: pos.leverage,
                    pnl_percent: roe_pct * (pos.capital_percent / 100.0),
                    roe_percent: roe_pct,
                    capital_percent: pos.capital_percent,
                    pnl_usd: net_pnl,
                    total_fees: fees,
                    capital_before: base_bal,
                    capital_after: base_bal + net_pnl,
                    is_win: net_pnl > 0.0,
                    account_id: acc.id,
                };

                if let Ok(conn_t) = init_trades_db("trades.db") {
                    let _ = save_closed_trade(&conn_t, &rec);
                }

                println!(
                    "  ⏹️ [SIMULADO] Cierre {} a FLAT: PnL: ${:.2} (ROE: {:.2}%)",
                    dir_str, net_pnl, roe_pct
                );
                set_simulated_position(tf, None);
            }
        }
    }

    Ok(())
}
