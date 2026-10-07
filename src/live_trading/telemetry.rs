use super::types::SignalType;
use chrono::{DateTime, Utc};

pub fn print_live_tick_banner(
    tf: &str,
    candle_time: i64,
    price: f64,
    signal: SignalType,
    dynamic_thr: f32,
    current_pos: SignalType,
    total_balance: f64,
) {
    let dt_str = chrono::DateTime::from_timestamp_millis(candle_time)
        .map(|dt: DateTime<Utc>| dt.format("%Y-%m-%d %H:%M:%S UTC").to_string())
        .unwrap_or_else(|| "N/A".to_string());

    let (sig_icon, sig_text) = match signal {
        SignalType::Long => ("🚀", "LONG"),
        SignalType::Short => ("🔻", "SHORT"),
        SignalType::Flat => ("⏹️", "FLAT"),
    };

    let pos_text = match current_pos {
        SignalType::Long => "LONG ACTIVO",
        SignalType::Short => "SHORT ACTIVO",
        SignalType::Flat => "FLAT (Sin posición)",
    };

    println!("\n========================================================================================");
    println!("  🌲 SYLVA LIVE ENGINE | TICK DE DECISIÓN [{}]", tf);
    println!("========================================================================================");
    println!("  • Vela Cerrada: {} | Precio BTC: ${:.2}", dt_str, price);
    println!("  • Señal del Árbol: {} {} (Umbral Dinámico: {:.4})", sig_icon, sig_text, dynamic_thr);
    println!("  • Posición en Cuenta: {} | Capital Unificado: ${:.2} USDT", pos_text, total_balance);
    println!("----------------------------------------------------------------------------------------");
}
