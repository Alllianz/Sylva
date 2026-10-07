use reqwest::Client;
use serde_json::json;

pub async fn send_trade_alert_webhook(
    client: &Client,
    webhook_url: &str,
    tf: &str,
    action: &str,
    symbol: &str,
    price: f64,
    amount: f64,
    notional: f64,
    leverage: f64,
    account_id: u32,
    details: &str,
) {
    if webhook_url.trim().is_empty() {
        return;
    }

    let color = match action {
        a if a.contains("LONG") => 0x2ECC71, // Verde
        a if a.contains("SHORT") => 0xE74C3C, // Rojo
        _ => 0xF39C12,                       // Naranja / Cierre
    };

    let title = format!("🌲 SYLVA LIVE | Operación {} en {}", action, tf);
    let payload = json!({
        "embeds": [{
            "title": title,
            "color": color,
            "fields": [
                { "name": "Par", "value": symbol, "inline": true },
                { "name": "Timeframe", "value": tf, "inline": true },
                { "name": "Precio", "value": format!("${:.2}", price), "inline": true },
                { "name": "Cantidad", "value": format!("{:.4} BTC", amount), "inline": true },
                { "name": "Nocional", "value": format!("${:.2} USDT", notional), "inline": true },
                { "name": "Apalancamiento", "value": format!("{:.0}X", leverage), "inline": true },
                { "name": "Cuenta ID", "value": format!("#{}", account_id), "inline": true },
                { "name": "Info / Alpha", "value": details, "inline": false }
            ],
            "footer": {
                "text": "Sylva Decision Forests & Streaming GBDT • Real-Time Engine"
            },
            "timestamp": chrono::Utc::now().to_rfc3339()
        }]
    });

    let _ = client.post(webhook_url).json(&payload).send().await;
}

pub async fn send_heartbeat_webhook(
    client: &Client,
    webhook_url: &str,
    server_name: &str,
    tf: &str,
    price: f64,
    signal: &str,
    alpha: f32,
    wallet_balance: f64,
) {
    if webhook_url.trim().is_empty() {
        return;
    }

    let payload = json!({
        "embeds": [{
            "title": format!("💓 Latido del Sistema • {}", server_name),
            "color": 0x3498DB,
            "fields": [
                { "name": "Timeframe", "value": tf, "inline": true },
                { "name": "Último Precio BTC", "value": format!("${:.2}", price), "inline": true },
                { "name": "Señal Activa", "value": signal, "inline": true },
                { "name": "Alpha Predicho", "value": format!("{:.5}", alpha), "inline": true },
                { "name": "Capital Total", "value": format!("${:.2} USDT", wallet_balance), "inline": true }
            ],
            "footer": {
                "text": "Sylva Engine Live Heartbeat"
            },
            "timestamp": chrono::Utc::now().to_rfc3339()
        }]
    });

    let _ = client.post(webhook_url).json(&payload).send().await;
}
