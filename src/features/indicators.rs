use crate::data::db::Kline;

pub fn calculate_atr(klines: &[Kline], period: usize) -> Vec<f64> {
    let n = klines.len();
    if n == 0 {
        return Vec::new();
    }
    let mut atr = vec![0.0; n];
    if n == 1 {
        atr[0] = klines[0].high - klines[0].low;
        return atr;
    }

    let mut tr = vec![0.0; n];
    tr[0] = klines[0].high - klines[0].low;
    for i in 1..n {
        let hl = klines[i].high - klines[i].low;
        let hc = (klines[i].high - klines[i - 1].close).abs();
        let lc = (klines[i].low - klines[i - 1].close).abs();
        tr[i] = hl.max(hc).max(lc);
    }

    let mut current_sum = 0.0;
    for i in 0..n {
        current_sum += tr[i];
        if i >= period {
            current_sum -= tr[i - period];
            atr[i] = current_sum / (period as f64);
        } else {
            atr[i] = current_sum / ((i + 1) as f64);
        }
    }

    atr
}

pub fn calculate_returns(klines: &[Kline]) -> Vec<f64> {
    let n = klines.len();
    if n == 0 {
        return Vec::new();
    }
    let mut ret = vec![0.0; n];
    for i in 1..n {
        let prev = klines[i - 1].close;
        if prev > 0.0 {
            ret[i] = (klines[i].close - prev) / prev;
        }
    }
    ret
}

pub fn calculate_sma(values: &[f64], period: usize) -> Vec<f64> {
    let n = values.len();
    let mut sma = vec![0.0; n];
    if n == 0 || period == 0 {
        return sma;
    }

    let mut sum = 0.0;
    for i in 0..n {
        sum += values[i];
        if i >= period {
            sum -= values[i - period];
            sma[i] = sum / (period as f64);
        } else {
            sma[i] = sum / ((i + 1) as f64);
        }
    }
    sma
}

pub fn calculate_ema(values: &[f64], period: usize) -> Vec<f64> {
    let n = values.len();
    let mut ema = vec![0.0; n];
    if n == 0 || period == 0 {
        return ema;
    }

    let alpha = 2.0 / (period as f64 + 1.0);
    ema[0] = values[0];
    for i in 1..n {
        ema[i] = alpha * values[i] + (1.0 - alpha) * ema[i - 1];
    }
    ema
}

pub fn calculate_volume_sma_10000(klines: &[Kline]) -> Vec<f64> {
    let mut sma = vec![0.0; klines.len()];
    let mut current_sum = 0.0;
    for idx in 0..klines.len() {
        current_sum += klines[idx].volume;
        if idx >= 10000 {
            current_sum -= klines[idx - 10000].volume;
            sma[idx] = current_sum / 10000.0;
        } else {
            sma[idx] = current_sum / ((idx + 1) as f64);
        }
    }
    sma
}

pub fn calculate_volume_smas(klines: &[Kline]) -> (Vec<f64>, Vec<f64>) {
    let mut sma5 = vec![0.0; klines.len()];
    let mut sma20 = vec![0.0; klines.len()];
    let mut sum5 = 0.0;
    let mut sum20 = 0.0;
    for i in 0..klines.len() {
        sum5 += klines[i].volume;
        if i >= 5 {
            sum5 -= klines[i - 5].volume;
            sma5[i] = sum5 / 5.0;
        } else {
            sma5[i] = sum5 / ((i + 1) as f64);
        }

        sum20 += klines[i].volume;
        if i >= 20 {
            sum20 -= klines[i - 20].volume;
            sma20[i] = sum20 / 20.0;
        } else {
            sma20[i] = sum20 / ((i + 1) as f64);
        }
    }
    (sma5, sma20)
}

pub fn calculate_macd_hist(klines: &[Kline]) -> Vec<f64> {
    let mut hist = vec![0.0; klines.len()];
    if klines.is_empty() {
        return hist;
    }

    let mut ema_12 = klines[0].close;
    let mut ema_26 = klines[0].close;
    let mut macd_line = vec![0.0; klines.len()];

    let alpha_12 = 2.0 / 13.0;
    let alpha_26 = 2.0 / 27.0;
    let alpha_9 = 2.0 / 10.0;

    for i in 0..klines.len() {
        let close = klines[i].close;
        ema_12 = close * alpha_12 + ema_12 * (1.0 - alpha_12);
        ema_26 = close * alpha_26 + ema_26 * (1.0 - alpha_26);
        macd_line[i] = ema_12 - ema_26;
    }

    let mut signal = macd_line[0];
    for i in 0..klines.len() {
        signal = macd_line[i] * alpha_9 + signal * (1.0 - alpha_9);
        hist[i] = macd_line[i] - signal;
    }
    hist
}

pub fn calculate_ema_200_hl2(klines: &[Kline]) -> Vec<f64> {
    let mut ema = vec![0.0; klines.len()];
    if klines.is_empty() {
        return ema;
    }

    let mut current_ema = (klines[0].high + klines[0].low) / 2.0;
    let alpha = 2.0 / 201.0;

    for i in 0..klines.len() {
        let mid = (klines[i].high + klines[i].low) / 2.0;
        current_ema = mid * alpha + current_ema * (1.0 - alpha);
        ema[i] = current_ema;
    }
    ema
}

pub fn calculate_rsi(klines: &[Kline], period: usize) -> Vec<f64> {
    let n = klines.len();
    let mut rsi = vec![50.0; n];
    if n <= period {
        return rsi;
    }

    let mut gains = vec![0.0; n];
    let mut losses = vec![0.0; n];

    for i in 1..n {
        let diff = klines[i].close - klines[i - 1].close;
        if diff > 0.0 {
            gains[i] = diff;
        } else {
            losses[i] = -diff;
        }
    }

    let mut avg_gain: f64 = gains[1..=period].iter().sum::<f64>() / (period as f64);
    let mut avg_loss: f64 = losses[1..=period].iter().sum::<f64>() / (period as f64);

    if avg_loss == 0.0 {
        rsi[period] = 100.0;
    } else {
        let rs = avg_gain / avg_loss;
        rsi[period] = 100.0 - (100.0 / (1.0 + rs));
    }

    for i in (period + 1)..n {
        avg_gain = (avg_gain * (period as f64 - 1.0) + gains[i]) / (period as f64);
        avg_loss = (avg_loss * (period as f64 - 1.0) + losses[i]) / (period as f64);

        if avg_loss == 0.0 {
            rsi[i] = 100.0;
        } else {
            let rs = avg_gain / avg_loss;
            rsi[i] = 100.0 - (100.0 / (1.0 + rs));
        }
    }

    rsi
}

pub fn compute_markov_state(klines: &[Kline], k_idx: usize) -> f64 {
    if k_idx < 19 {
        return 2.0;
    }

    let mut sum_close = 0.0;
    let mut sum_xy = 0.0;
    for i in 0..10 {
        let val = klines[k_idx - 9 + i].close;
        sum_close += val;
        sum_xy += (i as f64 - 4.5) * val;
    }
    let sma_10 = sum_close / 10.0;
    let slope = sum_xy / 82.5;

    let trend = if klines[k_idx].close > sma_10 && slope > 0.0 {
        2.0
    } else if klines[k_idx].close < sma_10 && slope < 0.0 {
        0.0
    } else {
        1.0
    };

    let mut tr_buffer = [0.0f64; 19];
    for i in 0..19 {
        let idx = k_idx - 18 + i;
        let kh = klines[idx].high;
        let kl = klines[idx].low;
        let prev_c = klines[idx - 1].close;
        tr_buffer[i] = (kh - kl).max((kh - prev_c).abs()).max((kl - prev_c).abs());
    }

    let mut natr_sum_10 = 0.0;
    let mut current_natr = 0.0;
    for offset in 0..10 {
        let mut sub_tr_sum = 0.0;
        for j in offset..(offset + 10) {
            sub_tr_sum += tr_buffer[j];
        }
        let target_idx = k_idx - 9 + offset;
        let close_p = klines[target_idx].close;
        let natr_val = if close_p > 0.0 {
            (sub_tr_sum / 10.0) / close_p
        } else {
            0.0
        };
        natr_sum_10 += natr_val;
        if offset == 9 {
            current_natr = natr_val;
        }
    }
    let avg_natr_10 = natr_sum_10 / 10.0;

    let volatility = if current_natr > avg_natr_10 { 1.0 } else { 0.0 };

    match trend as u8 {
        0 => if volatility == 0.0 { 0.0 } else { 1.0 },
        1 => if volatility == 0.0 { 2.0 } else { 3.0 },
        _ => if volatility == 0.0 { 4.0 } else { 5.0 },
    }
}
