use crate::live_trading::types::SignalType;
use hmac::{Hmac, Mac};
use reqwest::Client;
use serde_json::Value;
use sha2::Sha256;
use std::error::Error;

type HmacSha256 = Hmac<Sha256>;

pub fn calculate_signature(secret: &str, query: &str) -> String {
    let mut mac = HmacSha256::new_from_slice(secret.as_bytes())
        .expect("HMAC acepta clave de cualquier tamaño");
    mac.update(query.as_bytes());
    let result = mac.finalize();
    hex::encode(result.into_bytes())
}

pub fn normalize_symbol(symbol: &str) -> String {
    if symbol.contains('-') {
        symbol.to_string()
    } else {
        symbol.replace("USDT", "-USDT")
    }
}

pub async fn test_api_connection(
    client: &Client,
    api_key: &str,
    api_secret: &str,
    use_testnet: bool,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    let base_url = if use_testnet {
        "https://open-api-vst.bingx.com"
    } else {
        "https://open-api.bingx.com"
    };
    let timestamp = chrono::Utc::now().timestamp_millis();
    let query = format!("timestamp={}", timestamp);
    let signature = calculate_signature(api_secret, &query);
    let url = format!("{}/openApi/swap/v2/user/balance?{}&signature={}", base_url, query, signature);

    let res = client
        .get(&url)
        .header("X-BX-APIKEY", api_key)
        .send()
        .await?;

    if res.status().is_success() {
        let text = res.text().await?;
        let json: Value = serde_json::from_str(&text)?;
        if let Some(code) = json.get("code").and_then(|c| c.as_i64()) {
            if code == 0 {
                Ok(())
            } else {
                let msg = json.get("msg").and_then(|m| m.as_str()).unwrap_or("Error desconocido");
                Err(format!("Error BingX (código {}): {}", code, msg).into())
            }
        } else {
            Err("Formato de respuesta inválido de BingX".into())
        }
    } else {
        let err_text = res.text().await?;
        Err(format!("Error HTTP BingX: {}", err_text).into())
    }
}

pub async fn get_account_details(
    client: &Client,
    api_key: &str,
    api_secret: &str,
    use_testnet: bool,
) -> Result<(f64, f64, String), Box<dyn Error + Send + Sync>> {
    let base_url = if use_testnet {
        "https://open-api-vst.bingx.com"
    } else {
        "https://open-api.bingx.com"
    };
    let timestamp = chrono::Utc::now().timestamp_millis();
    let query = format!("timestamp={}", timestamp);
    let signature = calculate_signature(api_secret, &query);
    let url = format!("{}/openApi/swap/v2/user/balance?{}&signature={}", base_url, query, signature);

    let res = client
        .get(&url)
        .header("X-BX-APIKEY", api_key)
        .send()
        .await?;

    if res.status().is_success() {
        let text = res.text().await?;
        let json: Value = serde_json::from_str(&text)?;
        if let Some(data) = json.get("data").and_then(|d| d.get("balance")) {
            let wallet_balance: f64 = data
                .get("balance")
                .and_then(|v| v.as_str())
                .unwrap_or("0")
                .parse()
                .unwrap_or(0.0);
            let available_margin: f64 = data
                .get("availableMargin")
                .and_then(|v| v.as_str())
                .unwrap_or("0")
                .parse()
                .unwrap_or(0.0);
            let user_id = data
                .get("userId")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            Ok((wallet_balance, available_margin, user_id))
        } else {
            Err("Datos de balance no encontrados en BingX".into())
        }
    } else {
        let err_text = res.text().await?;
        Err(format!("Error HTTP al consultar balance: {}", err_text).into())
    }
}

pub async fn set_leverage(
    client: &Client,
    api_key: &str,
    api_secret: &str,
    symbol: &str,
    leverage: u32,
    position_side: &str,
    use_testnet: bool,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    let base_url = if use_testnet {
        "https://open-api-vst.bingx.com"
    } else {
        "https://open-api.bingx.com"
    };
    let normalized = normalize_symbol(symbol);
    let timestamp = chrono::Utc::now().timestamp_millis();
    let query = format!(
        "leverage={}&side={}&symbol={}&timestamp={}",
        leverage, position_side, normalized, timestamp
    );
    let signature = calculate_signature(api_secret, &query);
    let url = format!("{}/openApi/swap/v2/trade/leverage?{}&signature={}", base_url, query, signature);

    let res = client
        .post(&url)
        .header("X-BX-APIKEY", api_key)
        .send()
        .await?;

    if res.status().is_success() {
        let text = res.text().await?;
        let json: Value = serde_json::from_str(&text)?;
        if let Some(code) = json.get("code").and_then(|c| c.as_i64()) {
            if code == 0 {
                Ok(())
            } else {
                let msg = json.get("msg").and_then(|m| m.as_str()).unwrap_or("Error");
                Err(format!("Error BingX al fijar apalancamiento ({}): {}", code, msg).into())
            }
        } else {
            Err("Respuesta no válida al fijar apalancamiento".into())
        }
    } else {
        let err_text = res.text().await?;
        Err(format!("Error HTTP al fijar apalancamiento: {}", err_text).into())
    }
}

pub async fn get_open_positions(
    client: &Client,
    api_key: &str,
    api_secret: &str,
    symbol: &str,
    use_testnet: bool,
) -> Result<Vec<Value>, Box<dyn Error + Send + Sync>> {
    let base_url = if use_testnet {
        "https://open-api-vst.bingx.com"
    } else {
        "https://open-api.bingx.com"
    };
    let normalized = normalize_symbol(symbol);
    let timestamp = chrono::Utc::now().timestamp_millis();
    let query = format!("symbol={}&timestamp={}", normalized, timestamp);
    let signature = calculate_signature(api_secret, &query);
    let url = format!("{}/openApi/swap/v2/user/positions?{}&signature={}", base_url, query, signature);

    let res = client
        .get(&url)
        .header("X-BX-APIKEY", api_key)
        .send()
        .await?;

    if res.status().is_success() {
        let text = res.text().await?;
        let json: Value = serde_json::from_str(&text)?;
        if let Some(code) = json.get("code").and_then(|c| c.as_i64()) {
            if code == 0 {
                if let Some(data) = json.get("data").and_then(|d| d.as_array()) {
                    return Ok(data.clone());
                }
                Ok(vec![])
            } else {
                let msg = json.get("msg").and_then(|m| m.as_str()).unwrap_or("Error");
                Err(format!("Error BingX al consultar posiciones ({}): {}", code, msg).into())
            }
        } else {
            Err("Respuesta no válida de posiciones en BingX".into())
        }
    } else {
        let err_text = res.text().await?;
        Err(format!("Error HTTP al consultar posiciones: {}", err_text).into())
    }
}

pub fn parse_bingx_position(pos: &Value) -> (SignalType, f64) {
    let amt_str = pos
        .get("positionAmt")
        .or_else(|| pos.get("availableAmt"))
        .or_else(|| pos.get("volume"))
        .and_then(|v| {
            if let Some(s) = v.as_str() {
                Some(s.to_string())
            } else {
                v.as_f64().map(|n| n.to_string())
            }
        })
        .unwrap_or_else(|| "0".to_string());

    let amt: f64 = amt_str.parse().unwrap_or(0.0);
    if amt.abs() < 1e-6 {
        return (SignalType::Flat, 0.0);
    }

    let pos_side = pos
        .get("positionSide")
        .and_then(|s| s.as_str())
        .unwrap_or("")
        .to_uppercase();
    let side = pos
        .get("side")
        .and_then(|s| s.as_str())
        .unwrap_or("")
        .to_uppercase();

    if pos_side == "SHORT" || side == "SELL" || amt < 0.0 {
        (SignalType::Short, amt.abs())
    } else if pos_side == "LONG" || side == "BUY" || amt > 0.0 {
        (SignalType::Long, amt.abs())
    } else {
        (SignalType::Flat, 0.0)
    }
}

pub async fn place_market_order(
    client: &Client,
    api_key: &str,
    api_secret: &str,
    symbol: &str,
    side: &str,
    position_side: &str,
    quantity: f64,
    use_testnet: bool,
) -> Result<Value, Box<dyn Error + Send + Sync>> {
    let base_url = if use_testnet {
        "https://open-api-vst.bingx.com"
    } else {
        "https://open-api.bingx.com"
    };
    let normalized = normalize_symbol(symbol);
    let timestamp = chrono::Utc::now().timestamp_millis();
    let qty_str = format!("{:.4}", quantity);
    let qty_val: f64 = qty_str.parse().unwrap_or(0.0);
    if qty_val <= 0.0 {
        return Err("Cantidad inferior al lote mínimo permitido (0.0001 BTC)".into());
    }

    let query = format!(
        "positionSide={}&quantity={}&side={}&symbol={}&timestamp={}&type=MARKET",
        position_side, qty_str, side, normalized, timestamp
    );
    let signature = calculate_signature(api_secret, &query);
    let url = format!("{}/openApi/swap/v2/trade/order?{}&signature={}", base_url, query, signature);

    let res = client
        .post(&url)
        .header("X-BX-APIKEY", api_key)
        .send()
        .await?;

    if res.status().is_success() {
        let text = res.text().await?;
        let json: Value = serde_json::from_str(&text)?;
        if let Some(code) = json.get("code").and_then(|c| c.as_i64()) {
            if code == 0 {
                Ok(json)
            } else {
                let msg = json.get("msg").and_then(|m| m.as_str()).unwrap_or("Error");
                Err(format!("Error BingX al ejecutar orden ({}): {}", code, msg).into())
            }
        } else {
            Err("Respuesta no válida al colocar orden".into())
        }
    } else {
        let err_text = res.text().await?;
        Err(format!("Error HTTP al colocar orden: {}", err_text).into())
    }
}
