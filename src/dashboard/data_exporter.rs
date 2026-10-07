use chrono::{TimeZone, Utc};
use crate::data::db::Kline;

pub struct ExportedChartData {
    pub labels_json: String,
    pub iso_dates_json: String,
    pub timestamps_json: String,
    pub equity_pct_json: String,
    pub equity_nom_json: String,
    pub min_date_str: String,
    pub max_date_str: String,
}

pub fn export_equity_series(
    _klines: &[Kline],
    eq_curve_nom: &[(i64, f64)],
    eq_curve_pct: &[(i64, f64)],
) -> ExportedChartData {
    let mut labels = Vec::new();
    let mut iso_dates = Vec::new();
    let mut timestamps = Vec::new();
    let mut eq_nom = Vec::new();
    let mut eq_pct = Vec::new();

    let len = eq_curve_nom.len().min(eq_curve_pct.len());

    for k in 0..len {
        let (ts_nom, val_nom) = eq_curve_nom[k];
        let (_, val_pct) = eq_curve_pct[k];

        let dt = Utc.timestamp_millis_opt(ts_nom).unwrap();
        labels.push(dt.format("%Y-%m-%d %H:%M").to_string());
        iso_dates.push(dt.format("%Y-%m-%d").to_string());
        timestamps.push(ts_nom);
        eq_nom.push((val_nom * 100.0).round() / 100.0);
        eq_pct.push((val_pct * 100.0).round() / 100.0);
    }

    let min_date_str = iso_dates.first().cloned().unwrap_or_else(|| "2020-04-20".to_string());
    let max_date_str = iso_dates.last().cloned().unwrap_or_else(|| "2026-08-14".to_string());

    ExportedChartData {
        labels_json: serde_json::to_string(&labels).unwrap_or_else(|_| "[]".to_string()),
        iso_dates_json: serde_json::to_string(&iso_dates).unwrap_or_else(|_| "[]".to_string()),
        timestamps_json: serde_json::to_string(&timestamps).unwrap_or_else(|_| "[]".to_string()),
        equity_pct_json: serde_json::to_string(&eq_pct).unwrap_or_else(|_| "[]".to_string()),
        equity_nom_json: serde_json::to_string(&eq_nom).unwrap_or_else(|_| "[]".to_string()),
        min_date_str,
        max_date_str,
    }
}
