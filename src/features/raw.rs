use crate::data::db::Kline;
use crate::features::cached::CachedIndicators;
use crate::features::indicators::compute_markov_state;
use crate::features::names::TOTAL_FEATURES;
use chrono::{Datelike, TimeZone, Utc};
use std::f64::consts::PI;

/// Extrae las 32 características causales para la barra `t`
pub fn compute_raw_features(
    klines: &[Kline],
    t: usize,
    cached: &CachedIndicators,
    current_position_val: f32, // +1.0 Long, -1.0 Short, 0.0 Flat
) -> [f32; TOTAL_FEATURES] {
    let mut out = [0.0f32; TOTAL_FEATURES];
    if t < 19 || t >= klines.len() || t >= cached.volume_sma_10000.len() {
        out[TOTAL_FEATURES - 1] = current_position_val;
        return out;
    }

    let k = &klines[t];
    let open_t = if k.open > 0.0 { k.open } else { 1e-5 };
    let close_t = if k.close > 0.0 { k.close } else { 1e-5 };

    // --- 1. Estructura de Vela y Microestructura Inmediata ---
    // F1: Volumen Relativo a Largo Plazo
    let vol_sma_10k = cached.volume_sma_10000[t];
    let f1 = if vol_sma_10k > 0.0 {
        k.volume / vol_sma_10k
    } else {
        1.0
    };

    // F2: Rango Total Normalizado
    let f2 = (k.high - k.low) / open_t;

    // F3: Mecha Superior
    let f3 = (k.high - open_t) / open_t;

    // F4: Mecha Inferior
    let f4 = (open_t - k.low) / open_t;

    // F5: Cuerpo Real de la Vela
    let f5 = (open_t - close_t) / open_t;

    // F6: Retorno de 10 Velas
    let prev_c10 = klines[t.saturating_sub(10)].close;
    let f6 = if prev_c10 > 0.0 {
        (close_t - prev_c10) / prev_c10
    } else {
        0.0
    };

    // --- 2. Tendencia, Regresión Lineal y Aceleración ---
    // F7: Pendiente MCO 10 velas
    let mut sum_xy = 0.0;
    for i in 0..10 {
        let val = klines[t - 9 + i].close;
        sum_xy += (i as f64 - 4.5) * val;
    }
    let slope = sum_xy / 82.5;
    let f7 = slope / close_t;

    // F8: Posición Estocástica de Rango (10 velas)
    let mut high_10 = f64::NEG_INFINITY;
    let mut low_10 = f64::INFINITY;
    for j in (t - 9)..=t {
        if klines[j].high > high_10 {
            high_10 = klines[j].high;
        }
        if klines[j].low < low_10 {
            low_10 = klines[j].low;
        }
    }
    let range_10 = high_10 - low_10;
    let f8 = if range_10 > 0.0 {
        (close_t - low_10) / range_10
    } else {
        0.5
    };

    // F9: Z-Score de Precio (10 velas)
    let mut sum_10 = 0.0;
    for j in (t - 9)..=t {
        sum_10 += klines[j].close;
    }
    let mean_10 = sum_10 / 10.0;
    let mut sum_sq_diff = 0.0;
    for j in (t - 9)..=t {
        let diff = klines[j].close - mean_10;
        sum_sq_diff += diff * diff;
    }
    let std_10 = (sum_sq_diff / 10.0).sqrt();
    let f9 = if std_10 > 0.0 {
        (close_t - mean_10) / std_10
    } else {
        0.0
    };

    // F18 / F10: Aceleración de la Pendiente
    let mut sum_xy_prev = 0.0;
    let prev_t = t.saturating_sub(1);
    for i in 0..10 {
        let val = klines[prev_t - 9 + i].close;
        sum_xy_prev += (i as f64 - 4.5) * val;
    }
    let slope_prev = sum_xy_prev / 82.5;
    let f18_accel = (slope - slope_prev) / close_t;

    // F22 / F11: Ratio de Eficiencia de Kaufman (10 velas)
    let net_change = (close_t - prev_c10).abs();
    let mut sum_diffs = 0.0;
    for j in (t - 9)..=t {
        sum_diffs += (klines[j].close - klines[j - 1].close).abs();
    }
    let f22_kaufman = if sum_diffs > 0.0 {
        net_change / sum_diffs
    } else {
        0.0
    };

    // --- 3. Régimen de Mercado y Cadenas de Markov ---
    // F11 / F12: Estado Actual de Markov (St / 5.0)
    let s_t = compute_markov_state(klines, t);
    let f11 = s_t / 5.0;

    // F12 / F13: Estado Previo de Markov (St-1 / 5.0)
    let s_prev = compute_markov_state(klines, prev_t);
    let f12 = s_prev / 5.0;

    // --- 4. Medias Móviles y Macro-Estructura ---
    // F17 / F14: Distancia Porcentual a EMA-20
    let ema_20_val = cached.ema_20[t];
    let f17 = if ema_20_val > 0.0 {
        (close_t - ema_20_val) / ema_20_val
    } else {
        0.0
    };

    // F19 / F15: Distancia Porcentual a Macro EMA-200 (HL/2)
    let ema_200_val = cached.ema_200_hl2[t];
    let f19 = if ema_200_val > 0.0 {
        (close_t - ema_200_val) / ema_200_val
    } else {
        0.0
    };

    // F21 / F16: Flag de Macro-Régimen Alcista/Bajista
    let f21 = if close_t > ema_200_val { 1.0 } else { 0.0 };

    // --- 5. Volumen, Flujo y Presión Compradora/Vendedora ---
    // F13 / F17: Volumen Relativo a Mediano Plazo
    let vol_sma_20 = cached.volume_sma_20[t];
    let f13 = if vol_sma_20 > 0.0 {
        k.volume / vol_sma_20
    } else {
        1.0
    };

    // F14 / F18: Ratio de Flujo de Volumen Corto/Mediano (SMA5 / SMA20)
    let vol_sma_5 = cached.volume_sma_5[t];
    let f14 = if vol_sma_20 > 0.0 {
        vol_sma_5 / vol_sma_20
    } else {
        1.0
    };

    // F15 / F19: Amplitud Ajustada por Volumen
    let f15 = f2 / (f13 + 1e-5);

    // F23 / F20: Volume-Price Trend (VPT normalizado 10 velas)
    let mut sum_vpt = 0.0;
    for j in (t - 9)..=t {
        sum_vpt += (klines[j].close - klines[j - 1].close) * klines[j].volume;
    }
    let f23_vpt = if vol_sma_10k > 0.0 {
        sum_vpt / (close_t * vol_sma_10k)
    } else {
        0.0
    };

    // F26 / F21: Fuerza Direccional del Volumen
    let f26_signed_vol = (close_t - open_t).signum() * (k.volume + 1.0).ln();

    // --- 6. Osciladores, Volatilidad y Rango ---
    // F10 / F22: Normalización ATR(10) / Close
    let atr_10_val = cached.atr_10[t];
    let f10 = atr_10_val / close_t;

    // F16 / F23: Histograma MACD Normalizado / Close
    let macd_val = cached.macd_hist[t];
    let f16 = macd_val / close_t;

    // F20 / F24: Ratio de Volatilidad Rápida/Lenta (ATR5 / ATR50)
    let atr_50_val = cached.atr_50[t];
    let f20_ratio = if atr_50_val > 0.0 {
        cached.atr_5[t] / atr_50_val
    } else {
        1.0
    };

    // F24 / F25: Ancho Relativo del Canal (10 velas)
    let f24_width = if mean_10 > 0.0 {
        range_10 / mean_10
    } else {
        0.0
    };

    // F25 / F26: Tasa de Cambio de Volatilidad Intra-vela
    let prev_open_10 = if klines[t.saturating_sub(10)].open > 0.0 {
        klines[t.saturating_sub(10)].open
    } else {
        1.0
    };
    let prev_f2_10 = (klines[t.saturating_sub(10)].high - klines[t.saturating_sub(10)].low) / prev_open_10;
    let f25_f2_roc = if prev_f2_10.abs() > 1e-6 {
        (f2 - prev_f2_10) / prev_f2_10
    } else {
        0.0
    };

    // F27: RSI Centrado en Cero (RSI14 - 0.5)
    let f27_rsi = cached.rsi_14[t] / 100.0 - 0.5;

    // F32 / F28: Volatilidad Extrema de Parkinson
    let mut parkinson_sum = 0.0;
    let ln_2 = 2.0f64.ln();
    for j in (t - 9)..=t {
        let kh = klines[j].high;
        let kl = klines[j].low;
        let ratio = if kl > 0.0 { kh / kl } else { 1.0 };
        let ln_r = ratio.ln();
        parkinson_sum += ln_r * ln_r;
    }
    let f32_parkinson = (parkinson_sum / (40.0 * ln_2)).sqrt();

    // --- 7. Ciclos Semanales y Memoria Fractal ---
    let dt = Utc.timestamp_millis_opt(k.timestamp).unwrap();
    let weekday = dt.weekday().num_days_from_monday() as f64;

    // F29: Ciclo Semanal Seno
    let f29_sin_w = (weekday * 2.0 * PI / 7.0).sin();

    // F30: Ciclo Semanal Coseno
    let f30_cos_w = (weekday * 2.0 * PI / 7.0).cos();

    // F31: Diferenciación Fraccionaria (Memoria Larga d=0.4)
    let weights = [
        1.0, -0.4, -0.12, -0.064, -0.0416, -0.029952, -0.0229632, -0.01837056,
        -0.015155712, -0.0127981568,
    ];
    let log_close = close_t.ln();
    let mut frac_diff = 0.0;
    for i in 0..10 {
        let p = klines[t - i].close;
        frac_diff += weights[i] * (p.ln() - log_close);
    }
    let f31_frac = frac_diff;

    // --- 8. Canal de Estado y Retroalimentación Comercial ---
    let f32_pos = current_position_val as f64;

    out = [
        f1 as f32,
        f2 as f32,
        f3 as f32,
        f4 as f32,
        f5 as f32,
        f6 as f32,
        f7 as f32,
        f8 as f32,
        f9 as f32,
        f18_accel as f32,
        f22_kaufman as f32,
        f11 as f32,
        f12 as f32,
        f17 as f32,
        f19 as f32,
        f21 as f32,
        f13 as f32,
        f14 as f32,
        f15 as f32,
        f23_vpt as f32,
        f26_signed_vol as f32,
        f10 as f32,
        f16 as f32,
        f20_ratio as f32,
        f24_width as f32,
        f25_f2_roc as f32,
        f27_rsi as f32,
        f32_parkinson as f32,
        f29_sin_w as f32,
        f30_cos_w as f32,
        f31_frac as f32,
        f32_pos as f32,
    ];

    out
}
