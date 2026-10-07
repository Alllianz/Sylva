use crate::data::db::Kline;
use crate::metrics::backtest_report::BacktestReport;
use serde::{Deserialize, Serialize};
use std::error::Error;
use std::fs::{self, File};
use std::io::Write;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GbdtGridCandidateReport {
    pub max_depth: usize,
    pub min_samples_leaf: usize,
    pub n_trees: usize,
    pub cv_mse: f32,
    pub cv_mda: f32,
    pub cv_ic: f32,
    pub best_thr_long: f32,
    pub best_thr_short: f32,
    pub is_astro_fitness: f64,
    pub rank_slope: usize,
    pub rank_cap_dd: usize,
    pub rank_r2: usize,
    pub rank_smoothness: usize,
    pub weighted_avg_rank: f64,
    pub astro_rank_fitness: f64,
    pub raw_slope_ratio: f64,
    pub raw_cap_dd_ratio: f64,
    pub raw_r2_score: f64,
    pub raw_smoothness_score: f64,
    pub report_nom: BacktestReport,
    pub report_pct: BacktestReport,
    pub color: String,
}

pub struct GbdtGridDashboardSummary {
    pub tf: String,
    pub is_start_idx: usize,
    pub oos_start_idx: usize,
    pub initial_capital: f64,
    pub candidates: Vec<GbdtGridCandidateReport>,
}

pub fn generate_gbdt_grid_dashboard(
    summary: &GbdtGridDashboardSummary,
    klines: &[Kline],
) -> Result<String, Box<dyn Error + Send + Sync>> {
    let dashboard_dir = format!("dashboard/{}", summary.tf);
    fs::create_dir_all(&dashboard_dir)?;

    let html_content = render_gbdt_grid_html(summary, klines);
    let filename = format!("gbdt_grid_comparison_{}.html", summary.tf);
    let full_path = format!("{}/{}", dashboard_dir, filename);

    let mut file = File::create(&full_path)?;
    file.write_all(html_content.as_bytes())?;

    println!("  🌐 GBDT Grid Comparison Dashboard generado en: {}", full_path);
    Ok(full_path)
}

fn render_gbdt_grid_html(summary: &GbdtGridDashboardSummary, klines: &[Kline]) -> String {
    let start_idx = summary.is_start_idx;
    let eval_klines = if start_idx < klines.len() {
        &klines[start_idx..]
    } else {
        &klines[..]
    };

    let chart_step = (eval_klines.len() / 1000).max(1);
    let chart_klines: Vec<&Kline> = eval_klines.iter().step_by(chart_step).collect();
    let timestamps_json: Vec<String> = chart_klines
        .iter()
        .map(|k| {
            let dt = chrono::DateTime::from_timestamp(k.timestamp / 1000, 0)
                .map(|t| t.format("%Y-%m-%d %H:%M").to_string())
                .unwrap_or_else(|| format!("{}", k.timestamp));
            format!("\"{}\"", dt)
        })
        .collect();
    let labels_str = timestamps_json.join(",");

    let oos_rel_idx = summary.oos_start_idx.saturating_sub(summary.is_start_idx) / chart_step;

    // Datasets for Chart.js (Nominal and Percent / Compounded)
    // Para no desbordar memoria RAM ni congelar el navegador, seleccionamos únicamente los mejores 20 candidatos
    let mut chart_candidates: Vec<&GbdtGridCandidateReport> = summary.candidates.iter().collect();
    chart_candidates.sort_by(|a, b| {
        if a.weighted_avg_rank > 0.0 && b.weighted_avg_rank > 0.0 {
            a.weighted_avg_rank
                .partial_cmp(&b.weighted_avg_rank)
                .unwrap_or(std::cmp::Ordering::Equal)
        } else {
            b.is_astro_fitness
                .partial_cmp(&a.is_astro_fitness)
                .unwrap_or(std::cmp::Ordering::Equal)
        }
    });
    let top_chart_candidates = &chart_candidates[..chart_candidates.len().min(20)];

    let mut datasets_nom_json = Vec::new();
    let mut datasets_pct_json = Vec::new();
    for cand in top_chart_candidates {
        if cand.report_nom.equity_curve.is_empty() {
            continue;
        }

        let mut points_nom = Vec::with_capacity(chart_klines.len());
        let mut points_pct = Vec::with_capacity(chart_klines.len());

        let nom_len = cand.report_nom.equity_curve.len();
        let pct_len = cand.report_pct.equity_curve.len();

        for (step_i, _k) in chart_klines.iter().enumerate() {
            let nom_idx = (step_i * nom_len / chart_klines.len()).min(nom_len.saturating_sub(1));
            let pct_idx = (step_i * pct_len / chart_klines.len()).min(pct_len.saturating_sub(1));

            let val_nom = if nom_len > 0 { cand.report_nom.equity_curve[nom_idx].1 } else { summary.initial_capital };
            let val_pct = if pct_len > 0 { cand.report_pct.equity_curve[pct_idx].1 } else { summary.initial_capital };

            points_nom.push(format!("{:.2}", val_nom));
            points_pct.push(format!("{:.2}", val_pct));
        }

        let label_nom = format!(
            "Tree (D:{}, Leaf:{}, T:{})",
            cand.max_depth, cand.min_samples_leaf, cand.n_trees
        );
        let label_pct = format!(
            "Tree (D:{}, Leaf:{}, T:{}) (Comp.)",
            cand.max_depth, cand.min_samples_leaf, cand.n_trees
        );

        datasets_nom_json.push(format!(
            r#"{{
                label: "{}",
                data: [{}],
                borderColor: "{}",
                backgroundColor: "{}",
                borderWidth: 2.0,
                pointRadius: 0,
                pointHoverRadius: 4,
                tension: 0.1,
                fill: false
            }}"#,
            label_nom,
            points_nom.join(","),
            cand.color,
            cand.color
        ));

        datasets_pct_json.push(format!(
            r#"{{
                label: "{}",
                data: [{}],
                borderColor: "{}",
                backgroundColor: "{}",
                borderWidth: 2.0,
                pointRadius: 0,
                pointHoverRadius: 4,
                tension: 0.1,
                fill: false
            }}"#,
            label_pct,
            points_pct.join(","),
            cand.color,
            cand.color
        ));
    }
    let datasets_nom_str = datasets_nom_json.join(",\n");
    let datasets_pct_str = datasets_pct_json.join(",\n");

    // Sort candidates by Astro EVO Weighted Rank ascending (or IS Fitness descending if no rank)
    let mut ranked = summary.candidates.clone();
    ranked.sort_by(|a, b| {
        if a.weighted_avg_rank > 0.0 && b.weighted_avg_rank > 0.0 {
            a.weighted_avg_rank
                .partial_cmp(&b.weighted_avg_rank)
                .unwrap_or(std::cmp::Ordering::Equal)
        } else {
            b.is_astro_fitness
                .partial_cmp(&a.is_astro_fitness)
                .unwrap_or(std::cmp::Ordering::Equal)
        }
    });

    // Table rows (mostrar hasta un máximo de 100 mejores para no colapsar la vista web)
    let mut table_rows = String::new();
    for (rank, c) in ranked.iter().take(100).enumerate() {
        let medal = match rank {
            0 => "🥇 1º",
            1 => "🥈 2º",
            2 => "🥉 3º",
            _ => "",
        };

        let net_pnl_class = if c.report_nom.net_profit >= 0.0 {
            "val-pos"
        } else {
            "val-neg"
        };
        let ret_sign = if c.report_nom.net_profit >= 0.0 { "+" } else { "" };

        let rank_display = if c.weighted_avg_rank > 0.0 {
            format!("{:.2} (R²: #{} | Slp: #{} | DD: #{} | Sm: #{})", c.weighted_avg_rank, c.rank_r2, c.rank_slope, c.rank_cap_dd, c.rank_smoothness)
        } else {
            format!("{:.2}", c.is_astro_fitness)
        };

        table_rows.push_str(&format!(
            r#"<tr class="table-row">
                <td class="rank-col"><span class="rank-badge rank-{}">{}</span> {}</td>
                <td class="name-col"><span class="model-badge" style="background-color: {};"></span> <strong>D:{} | Leaf:{} | T:{}</strong></td>
                <td class="fit-col"><strong>{}</strong></td>
                <td>{:.6}</td>
                <td>{:.2}%</td>
                <td>{:+.4}</td>
                <td class="pnl-col {}">{}${:+.2} ({}{:.2}%)</td>
                <td>{:.2}%</td>
                <td>{:.2}</td>
                <td class="val-neg">{:.2}% <span style="font-size: 11px; color:#94a3b8;">(${:.0})</span> | {:.2}% <span style="font-size: 11px; color:#a855f7;">(${:.0})</span></td>
                <td style="color: var(--allianz-gold); font-weight: 600;">{} v</td>
                <td>{} ({}/{})</td>
                <td>{:+.4} / {:+.4}</td>
            </tr>"#,
            rank + 1,
            rank + 1,
            medal,
            c.color,
            c.max_depth,
            c.min_samples_leaf,
            c.n_trees,
            rank_display,
            c.cv_mse,
            c.cv_mda,
            c.cv_ic,
            net_pnl_class,
            ret_sign,
            c.report_nom.net_profit,
            ret_sign,
            c.report_nom.total_return_pct,
            c.report_nom.win_rate_pct,
            c.report_nom.profit_factor,
            c.report_nom.max_drawdown_pct,
            c.report_nom.max_drawdown_amount,
            c.report_pct.max_drawdown_pct,
            c.report_pct.max_drawdown_amount,
            c.report_nom.max_stagnation_bars,
            c.report_nom.total_trades,
            c.report_nom.total_longs,
            c.report_nom.total_shorts,
            c.best_thr_long,
            c.best_thr_short,
        ));
    }

    // Top 3 Podium Cards
    let mut top3_cards = String::new();
    for (i, c) in ranked.iter().take(3).enumerate() {
        let (trophy, border_color) = match i {
            0 => ("🏆 ÁRBOL CAMPEÓN", "#FFD700"),
            1 => ("🥈 2º MEJOR ÁRBOL", "#C0C0C0"),
            _ => ("🥉 3º MEJOR ÁRBOL", "#CD7F32"),
        };
        top3_cards.push_str(&format!(
            r#"<div class="pod top-card" style="border-top: 4px solid {};">
                <div class="top-card-header">
                    <span class="trophy-tag">{}</span>
                    <span class="score-pill">Fitness IS: {:.2}</span>
                </div>
                <h3>Profundidad: {} | Min Leaf: {} | Trees: {}</h3>
                <div class="top-card-metrics">
                    <div class="tc-metric"><span class="lbl">Retorno Total</span><span class="val val-pos">+${:.2} ({:+.2}%)</span></div>
                    <div class="tc-metric"><span class="lbl">Max Drawdown (Nom | Comp)</span><span class="val val-neg">{:.2}% | {:.2}%</span></div>
                    <div class="tc-metric"><span class="lbl">CV OOS MSE</span><span class="val">{:.6}</span></div>
                    <div class="tc-metric"><span class="lbl">Hit Ratio MDA</span><span class="val">{:.2}%</span></div>
                    <div class="tc-metric"><span class="lbl">Profit Factor</span><span class="val">{:.2}</span></div>
                    <div class="tc-metric"><span class="lbl">Win Rate</span><span class="val">{:.2}%</span></div>
                    <div class="tc-metric"><span class="lbl">Máx Estancamiento</span><span class="val" style="color: var(--allianz-gold);">{} v</span></div>
                </div>
            </div>"#,
            border_color,
            trophy,
            c.is_astro_fitness,
            c.max_depth,
            c.min_samples_leaf,
            c.n_trees,
            c.report_nom.net_profit,
            c.report_nom.total_return_pct,
            c.report_nom.max_drawdown_pct,
            c.report_pct.max_drawdown_pct,
            c.cv_mse,
            c.cv_mda,
            c.report_nom.profit_factor,
            c.report_nom.win_rate_pct,
            c.report_nom.max_stagnation_bars,
        ));
    }

    let tf = &summary.tf;

    format!(r#"<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🌲 GBDT Grid Search Comparison - Quant Equation Lab ({tf})</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Montserrat:wght@300;400;500;600;700;800;900&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg-dark: #0b0e11;
            --pod-bg: #15191e;
            --pod-bg-grad: linear-gradient(180deg, #1a2027 0%, #12161b 100%);
            --border-subtle: rgba(255, 255, 255, 0.08);
            --border-highlight: rgba(255, 255, 255, 0.16);
            --text-pure: #FFFFFF;
            --text-light: #EBE6E1;
            --text-mid: #9FAAB5;
            --text-dark: #6C7885;
            --allianz-red: #FF2E4D;
            --allianz-gold: #FFD700;
            --accent-green: #00E676;
            --accent-blue: #00B0FF;
            --accent-purple: #AB47BC;
        }}

        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Montserrat', sans-serif !important;
        }}

        body {{
            background-color: var(--bg-dark);
            color: var(--text-light);
            min-height: 100vh;
            padding-bottom: 60px;
        }}

        .flag-strip {{
            height: 5px;
            width: 100%;
            background: linear-gradient(90deg, #10b981, var(--allianz-gold), var(--accent-blue), var(--allianz-red));
        }}

        .container {{
            max-width: 98%;
            margin: 0 auto;
            padding: 12px 14px;
        }}

        .header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 14px 20px;
            background: var(--pod-bg-grad);
            border: 1px solid var(--border-subtle);
            border-left: 5px solid #10b981;
            border-radius: 12px;
            margin-bottom: 12px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
        }}

        .header-title h1 {{
            font-size: 20px;
            font-weight: 800;
            color: var(--text-pure);
            letter-spacing: -0.5px;
            display: flex;
            align-items: center;
            gap: 10px;
        }}

        .header-title p {{
            font-size: 12px;
            color: var(--text-mid);
            margin-top: 2px;
            font-weight: 500;
        }}

        .badge-tf {{
            background: rgba(16, 185, 129, 0.15);
            color: #10b981;
            border: 1px solid rgba(16, 185, 129, 0.4);
            padding: 6px 14px;
            border-radius: 16px;
            font-size: 13px;
            font-weight: 700;
            letter-spacing: 0.5px;
        }}

        .top3-grid {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 10px;
            margin-bottom: 12px;
        }}

        .pod {{
            background: var(--pod-bg);
            border: 1px solid var(--border-subtle);
            border-radius: 12px;
            padding: 14px 16px;
            box-shadow: 0 8px 24px rgba(0,0,0,0.3);
        }}

        .top-card h3 {{
            font-size: 13.5px;
            font-weight: 700;
            margin: 8px 0 10px;
            color: var(--text-pure);
        }}

        .top-card-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}

        .trophy-tag {{
            font-size: 12px;
            font-weight: 800;
            letter-spacing: 0.5px;
        }}

        .score-pill {{
            background: rgba(255, 215, 0, 0.15);
            color: var(--allianz-gold);
            border: 1px solid rgba(255, 215, 0, 0.3);
            padding: 3px 8px;
            border-radius: 10px;
            font-size: 11px;
            font-weight: 700;
        }}

        .top-card-metrics {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 6px 12px;
            border-top: 1px solid var(--border-subtle);
            padding-top: 10px;
        }}

        .tc-metric {{
            display: flex;
            flex-direction: column;
        }}

        .tc-metric .lbl {{
            font-size: 10px;
            color: var(--text-mid);
            font-weight: 500;
        }}

        .tc-metric .val {{
            font-size: 12px;
            font-weight: 700;
            margin-top: 2px;
        }}

        .val-pos {{ color: var(--accent-green); }}
        .val-neg {{ color: var(--allianz-red); }}

        .chart-pod {{
            padding: 12px 6px 6px 6px;
            margin-bottom: 12px;
        }}

        .chart-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 16px;
        }}

        .chart-header h2 {{
            font-size: 18px;
            font-weight: 800;
            color: var(--text-pure);
        }}

        .chart-controls {{
            display: flex;
            gap: 10px;
        }}

        .btn-ctrl {{
            background: #1f2630;
            border: 1px solid var(--border-subtle);
            color: var(--text-light);
            padding: 6px 14px;
            border-radius: 8px;
            font-size: 12px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s ease;
        }}

        .btn-ctrl:hover {{
            background: #2b3543;
            border-color: var(--border-highlight);
        }}

        .chart-wrapper {{
            position: relative;
            height: 480px;
            width: 100%;
        }}

        .table-pod {{
            margin-bottom: 24px;
            overflow-x: auto;
        }}

        .table-pod h2 {{
            font-size: 18px;
            font-weight: 800;
            color: var(--text-pure);
            margin-bottom: 16px;
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 12.5px;
            text-align: right;
        }}

        th {{
            background: #1a2027;
            color: var(--text-mid);
            font-weight: 700;
            font-size: 11.5px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            padding: 12px 14px;
            border-bottom: 2px solid var(--border-highlight);
            white-space: nowrap;
        }}

        th:first-child, th:nth-child(2) {{
            text-align: left;
        }}

        td {{
            padding: 12px 14px;
            border-bottom: 1px solid var(--border-subtle);
            white-space: nowrap;
            color: var(--text-light);
            font-weight: 500;
        }}

        .table-row:hover {{
            background-color: rgba(255, 255, 255, 0.03);
        }}

        .rank-col {{
            text-align: left;
            font-weight: 700;
        }}

        .rank-badge {{
            display: inline-block;
            width: 24px;
            height: 24px;
            line-height: 24px;
            text-align: center;
            border-radius: 6px;
            font-size: 11px;
            font-weight: 800;
            margin-right: 4px;
            background: #26303c;
        }}

        .rank-1 {{ background: var(--allianz-gold); color: #000; }}
        .rank-2 {{ background: #C0C0C0; color: #000; }}
        .rank-3 {{ background: #CD7F32; color: #000; }}

        .name-col {{
            text-align: left;
            display: flex;
            align-items: center;
            gap: 8px;
        }}

        .model-badge {{
            width: 10px;
            height: 10px;
            border-radius: 50%;
            display: inline-block;
        }}

        .fit-col strong {{
            color: var(--allianz-gold);
            font-size: 13.5px;
        }}

        .equity-toggle-container {{
            display: flex;
            background: rgba(0, 0, 0, 0.4);
            padding: 4px;
            border-radius: 10px;
            border: 1px solid var(--border-subtle);
            gap: 6px;
        }}

        .toggle-btn {{
            background: transparent;
            border: 1px solid transparent;
            color: var(--text-mid);
            padding: 6px 14px;
            border-radius: 6px;
            font-size: 12px;
            font-weight: 700;
            cursor: pointer;
            transition: all 0.2s ease;
        }}

        .toggle-btn:hover {{
            color: var(--text-pure);
            background: rgba(255, 255, 255, 0.05);
        }}

        .toggle-btn.active {{
            background: #10b981;
            color: #0b0e11;
            border-color: #10b981;
            box-shadow: 0 2px 8px rgba(16, 185, 129, 0.3);
        }}

        .footer {{
            text-align: center;
            font-size: 12px;
            color: var(--text-dark);
            margin-top: 30px;
        }}

        @media (max-width: 1200px) {{
            .top3-grid {{
                grid-template-columns: 1fr;
            }}
        }}
    </style>
</head>
<body>
    <div class="flag-strip"></div>
    <div class="container">
        <div class="header">
            <div class="header-title">
                <h1>🌲 GBDT Grid Search Anti-Overfitting Arena</h1>
                <p>Laboratorio Cuantitativo de Allianz | Comparación Exhaustiva de Arquitecturas de Árboles con Purged K-Fold CV & Astro EVO Fitness</p>
            </div>
            <div class="badge-tf">⏱️ Temporalidad: {tf}</div>
        </div>

        <div class="top3-grid">
            {top3_cards}
        </div>

        <div class="pod chart-pod">
            <div class="chart-header">
                <div>
                    <h2>📈 Curvas de Equity Comparativas (IS + OOS)</h2>
                    <div style="font-size: 12px; color: var(--text-mid); margin-top: 4px;">Línea Vertical discontinua = Inicio Out-Of-Sample (0% Look-Ahead)</div>
                </div>
                <div style="display: flex; gap: 14px; align-items: center; flex-wrap: wrap;">
                    <div class="equity-toggle-container">
                        <button id="btn-nom" class="toggle-btn active" onclick="switchGridMode('nominal')">💵 Nominal ($)</button>
                        <button id="btn-pct" class="toggle-btn" onclick="switchGridMode('compuesto')">📈 Porcentual / Compuesto (%)</button>
                    </div>
                    <div class="chart-controls">
                        <button class="btn-ctrl" onclick="showAll()">Mostrar Todos</button>
                        <button class="btn-ctrl" onclick="hideAll()">Ocultar Todos</button>
                        <button class="btn-ctrl" onclick="showTop3()">Solo Top 3</button>
                    </div>
                </div>
            </div>
            <div class="chart-wrapper">
                <canvas id="gridChart"></canvas>
            </div>
        </div>

        <div class="pod table-pod">
            <h2>🏆 Leaderboard de Hiperparámetros (Clasificación por Fitness In-Sample)</h2>
            <table>
                <thead>
                    <tr>
                        <th>Rank</th>
                        <th>Configuración (Depth / Leaf / Trees)</th>
                        <th>IS Astro Fitness</th>
                        <th>CV OOS MSE</th>
                        <th>Hit Ratio MDA</th>
                        <th>Rank IC (Spearman)</th>
                        <th>Retorno Neto PnL</th>
                        <th>Win Rate</th>
                        <th>Profit Factor</th>
                        <th>Max Drawdown (Nom | Comp)</th>
                        <th>Estanc.</th>
                        <th>Trades (L/S)</th>
                        <th>Umbrales (L/S)</th>
                    </tr>
                </thead>
                <tbody>
                    {table_rows}
                </tbody>
            </table>
        </div>

        <div class="footer">
            Quant Equation Lab &copy; 2026 | Sistema Cuantitativo Ultra-Modular en Rust para Allianz | 0% Look-Ahead Bias
        </div>
    </div>

    <script>
        const ctx = document.getElementById('gridChart').getContext('2d');
        const oosIndex = {oos_rel_idx};

        const datasetsNominal = [
            {datasets_nom_str}
        ];
        const datasetsCompuesto = [
            {datasets_pct_str}
        ];

        let currentGridMode = 'nominal';

        const chart = new Chart(ctx, {{
            type: 'line',
            data: {{
                labels: [{labels_str}],
                datasets: datasetsNominal
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                layout: {{
                    padding: {{ left: 0, right: 0, top: 4, bottom: 0 }}
                }},
                interaction: {{
                    mode: 'index',
                    intersect: false,
                }},
                plugins: {{
                    legend: {{
                        position: 'top',
                        labels: {{
                            color: '#EBE6E1',
                            font: {{ family: 'Montserrat', size: 11, weight: '600' }},
                            boxWidth: 14,
                            padding: 12
                        }}
                    }},
                    tooltip: {{
                        backgroundColor: 'rgba(21, 25, 30, 0.95)',
                        titleColor: '#FFFFFF',
                        bodyColor: '#EBE6E1',
                        borderColor: 'rgba(255, 255, 255, 0.1)',
                        borderWidth: 1,
                        padding: 12,
                        callbacks: {{
                            label: function(context) {{
                                let label = context.dataset.label || '';
                                if (label) label += ': ';
                                if (context.parsed.y !== null) {{
                                    label += '$' + context.parsed.y.toLocaleString('en-US', {{ minimumFractionDigits: 2, maximumFractionDigits: 2 }});
                                }}
                                return label;
                            }}
                        }}
                    }}
                }},
                scales: {{
                    x: {{
                        bounds: 'data',
                        offset: false,
                        grid: {{ color: 'rgba(255, 255, 255, 0.05)' }},
                        ticks: {{ color: '#9FAAB5', font: {{ family: 'Montserrat', size: 10 }}, maxTicksLimit: 14 }}
                    }},
                    y: {{
                        grid: {{ color: 'rgba(255, 255, 255, 0.05)' }},
                        ticks: {{
                            color: '#9FAAB5',
                            font: {{ family: 'Montserrat', size: 10 }},
                            callback: function(value) {{ return '$' + value.toLocaleString(); }}
                        }}
                    }}
                }}
            }},
            plugins: [{{
                id: 'oosLine',
                afterDraw: (chart) => {{
                    if (oosIndex > 0 && oosIndex < chart.data.labels.length) {{
                        const meta = chart.getDatasetMeta(0);
                        if (meta.data[oosIndex]) {{
                            const x = meta.data[oosIndex].x;
                            const ctx = chart.ctx;
                            ctx.save();
                            ctx.beginPath();
                            ctx.moveTo(x, chart.chartArea.top);
                            ctx.lineTo(x, chart.chartArea.bottom);
                            ctx.lineWidth = 2;
                            ctx.setLineDash([6, 6]);
                            ctx.strokeStyle = '#FF2E4D';
                            ctx.stroke();

                            ctx.fillStyle = '#FF2E4D';
                            ctx.font = 'bold 11px Montserrat';
                            ctx.textAlign = 'center';
                            ctx.fillText('⚡ INICIO OUT-OF-SAMPLE (OOS)', x, chart.chartArea.top + 18);
                            ctx.restore();
                        }}
                    }}
                }}
            }}]
        }});

        function switchGridMode(mode) {{
            currentGridMode = mode;
            document.getElementById('btn-nom').classList.toggle('active', mode === 'nominal');
            document.getElementById('btn-pct').classList.toggle('active', mode === 'compuesto');
            if (mode === 'nominal') {{
                chart.data.datasets = datasetsNominal;
            }} else {{
                chart.data.datasets = datasetsCompuesto;
            }}
            chart.update();
        }}

        function showAll() {{
            chart.data.datasets.forEach((ds, i) => {{
                chart.setDatasetVisibility(i, true);
            }});
            chart.update();
        }}

        function hideAll() {{
            chart.data.datasets.forEach((ds, i) => {{
                chart.setDatasetVisibility(i, false);
            }});
            chart.update();
        }}

        function showTop3() {{
            chart.data.datasets.forEach((ds, i) => {{
                chart.setDatasetVisibility(i, i < 3);
            }});
            chart.update();
        }}
    </script>
</body>
</html>
"#,
        tf = tf,
        labels_str = labels_str,
        datasets_nom_str = datasets_nom_str,
        datasets_pct_str = datasets_pct_str,
        table_rows = table_rows,
        top3_cards = top3_cards,
        oos_rel_idx = oos_rel_idx,
    )
}
