use super::account_manager::{fetch_unified_accounts_or_simulated, get_pool_position_state, print_unified_accounts_summary};
use super::config::{get_all_api_configs, get_live_config};
use super::order_executor::execute_unified_signal;
use super::signal_evaluator::evaluate_live_signal;
use super::sync::{sync_all_missing_klines_from_binance, sync_recent_klines_from_binance, tf_to_duration_millis};
use super::telemetry::print_live_tick_banner;
use crate::data::db::get_all_klines_closed_only;
use crate::engine::equation_model::EquationModel;
use crate::features::mask::FeatureMask;
use crate::features::TOTAL_FEATURES;
use crate::trees::online::gbdt::OnlineGbdtModel;
use crate::trees::online::model_wrapper::OnlineGbdtEquationModel;
use crate::trees::online::types::OnlineGbdtConfig;
use chrono::Utc;
use reqwest::Client;
use rusqlite::Connection;
use std::error::Error;
use std::io::{self, Write};

pub async fn run_live_trading_loop(
    client: &Client,
    conn_candles: &Connection,
    conn_config: &Connection,
    tf: &str,
    leverage: f64,
    capital_percent: f64,
    is_allianz_custom: bool,
    threshold_long: f32,
    threshold_short: f32,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    println!("\n========================================================================================");
    println!("             🚀 SYLVA LIVE TRADING ENGINE INICIANDO EN [{}]", tf);
    println!("========================================================================================");
    println!("  • Apalancamiento: {:.0}X | % Capital: {:.1}%", leverage, capital_percent);
    println!("  • Umbral Long: {:.4} | Umbral Short: {:.4}", threshold_long, threshold_short);
    if is_allianz_custom {
        println!("  • Modo: GBDT Allianz (13 Features Cuantitativas Optimizadas - 19 Variables Podadas)");
    } else {
        println!("  • Modo: Online GBDT General (32 Features Cuantitativas Completas)");
    }

    // 1. Sincronización completa de velas históricas desde Binance
    sync_all_missing_klines_from_binance(client, conn_candles, tf).await;

    // 2. Consulta y resumen inicial de cuentas
    let api_configs = get_all_api_configs(conn_config).unwrap_or_default();
    let initial_pool = fetch_unified_accounts_or_simulated(client, &api_configs).await?;
    print_unified_accounts_summary(&initial_pool);

    let live_cfg = get_live_config(conn_config).unwrap_or_default();
    let trade_webhook = live_cfg.trade_webhook;

    // 3. Cargar historial causal para precarga del modelo
    let klines = get_all_klines_closed_only(conn_candles, tf)?;
    if klines.len() < 100 {
        eprintln!("  ⚠️ Historial insuficiente ({} velas). Descargue datos primero.", klines.len());
        return Ok(());
    }

    println!("  🔄 Precargando modelo de Streaming GBDT con {} velas...", klines.len());

    let gbdt_cfg = OnlineGbdtConfig::default();

    let active_features: Vec<usize> = if is_allianz_custom {
        let mask = FeatureMask::new_allianz_custom();
        (0..TOTAL_FEATURES)
            .filter(|&i| mask.is_active(i))
            .collect()
    } else {
        (0..TOTAL_FEATURES).collect()
    };

    let gbdt_model = OnlineGbdtModel::new(gbdt_cfg, &active_features);

    let mut equation_model = OnlineGbdtEquationModel::new(
        gbdt_model,
        threshold_long,
        threshold_short,
        100,
    );

    if is_allianz_custom {
        equation_model = equation_model.with_mask(FeatureMask::new_allianz_custom());
    }

    // Warmup causal de indicadores y modelo con el historial cerrado
    let _ = equation_model.evaluate(&klines);

    println!("  ✅ Modelo preparado y listo para recibir ticks en tiempo real.");
    println!("  ⏱️ Esperando al cierre de cada vela en {}... (Presiona Ctrl+C para detener)", tf);

    let tf_duration = tf_to_duration_millis(tf);
    let mut last_operated_candle_ts: i64 = 0;

    loop {
        let now_ms = Utc::now().timestamp_millis();
        let current_candle_start = (now_ms / tf_duration) * tf_duration;
        let target_closed_ts = current_candle_start - tf_duration;

        // Comprobar si hay una vela recién cerrada que aún no hemos operado
        if now_ms >= current_candle_start && target_closed_ts != last_operated_candle_ts {
            // Sincronizar la vela cerrada desde Binance para consolidar en DB
            let ready = sync_recent_klines_from_binance(client, conn_candles, tf, target_closed_ts).await;
            if ready {
                let fresh_klines = get_all_klines_closed_only(conn_candles, tf)?;
                if let Some(eval) = evaluate_live_signal(&mut equation_model, &fresh_klines, threshold_long, threshold_short) {
                    let pool = fetch_unified_accounts_or_simulated(client, &api_configs).await?;
                    let current_pos = get_pool_position_state(client, &pool, tf).await;

                    print_live_tick_banner(
                        tf,
                        eval.candle_timestamp,
                        eval.last_price,
                        eval.signal,
                        eval.dynamic_thr,
                        current_pos,
                        pool.total_wallet_balance,
                    );

                    // Ejecutar orden si corresponde
                    let _ = execute_unified_signal(
                        client,
                        &pool,
                        tf,
                        &eval,
                        capital_percent,
                        leverage,
                        &trade_webhook,
                    )
                    .await;

                    last_operated_candle_ts = target_closed_ts;
                }
            }
        }

        // Cuenta regresiva visible
        let next_close_ms = current_candle_start + tf_duration;
        let remaining_secs = ((next_close_ms - now_ms) / 1000).max(0);
        let rem_min = remaining_secs / 60;
        let rem_sec = remaining_secs % 60;
        print!("\r  ⏳ Próximo cierre de vela en {:02}m {:02}s... (BTC: Monitoreando)   ", rem_min, rem_sec);
        io::stdout().flush()?;

        tokio::time::sleep(tokio::time::Duration::from_millis(1000)).await;
    }
}
