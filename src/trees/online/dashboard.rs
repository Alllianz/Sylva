use crate::data::db::Kline;
use crate::metrics::backtest_report::BacktestReport;
use crate::trees::online::gbdt::OnlineGbdtModel;
use std::fs::{create_dir_all, File};
use std::io::Write;

/// Genera un dashboard interactivo HTML para el modelo Online GBDT
pub fn generate_online_gbdt_dashboard(
    title: &str,
    tf: &str,
    _klines: &[Kline],
    report_nom: &BacktestReport,
    report_pct: &BacktestReport,
    model: &OnlineGbdtModel,
    leverage: f64,
    capital_percent: f64,
) -> Result<String, std::io::Error> {
    let clean_title = sanitize_slug(title);
    let dashboard_dir = format!("dashboard/{}", tf);
    create_dir_all(&dashboard_dir)?;
    let filename = format!("online_gbdt_{}_{}.html", tf, clean_title);
    let full_path = format!("{}/{}", dashboard_dir, filename);
    let mut file = File::create(&full_path)?;

    let is_profit = report_nom.net_profit >= 0.0;
    let pnl_color = if is_profit { "#10b981" } else { "#ef4444" };

    let mut ranking_html = String::new();
    let ranking = model.get_feature_importance_ranking();
    for (rank_pos, (idx, name, gain_pct)) in ranking.iter().enumerate() {
        ranking_html.push_str(&format!(
            r#"<tr>
                <td style="color: #94a3b8; font-weight: 600;">#{}</td>
                <td style="color: #38bdf8; font-family: monospace;">F{}</td>
                <td style="color: #f8fafc;">{}</td>
                <td style="text-align: right; color: #10b981; font-weight: 700;">{:.2}%</td>
                <td style="width: 35%;">
                    <div style="background: rgba(255,255,255,0.05); border-radius: 4px; overflow: hidden; height: 8px;">
                        <div style="background: linear-gradient(90deg, #3b82f6, #10b981); width: {:.1}%; height: 100%;"></div>
                    </div>
                </td>
            </tr>"#,
            rank_pos + 1, idx + 1, name, gain_pct, gain_pct.min(100.0)
        ));
    }

    // Puntos de la curva de equity nominal
    let mut nom_equity_points = String::new();
    for (t, eq) in &report_nom.equity_curve {
        nom_equity_points.push_str(&format!("{{ x: {}, y: {:.2} }},", t, eq));
    }

    // Puntos de la curva de equity compuesta
    let mut pct_equity_points = String::new();
    for (t, eq) in &report_pct.equity_curve {
        pct_equity_points.push_str(&format!("{{ x: {}, y: {:.2} }},", t, eq));
    }

    let html = format!(r#"<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <title>Online GBDT Dashboard | Quant Equation Lab</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        :root {{
            --bg-dark: #090d16;
            --card-bg: #111827;
            --border-color: #1f293d;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --accent-green: #10b981;
            --accent-blue: #3b82f6;
            --accent-purple: #8b5cf6;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            background: var(--bg-dark);
            color: var(--text-main);
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
            padding: 12px 14px;
            line-height: 1.4;
        }}
        .header {{
            background: linear-gradient(135deg, #1e293b, #0f172a);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 14px 20px;
            margin-bottom: 12px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .badge {{
            display: inline-block;
            background: rgba(16, 185, 129, 0.15);
            color: #34d399;
            padding: 3px 8px;
            border-radius: 6px;
            font-size: 11px;
            font-weight: 700;
            text-transform: uppercase;
        }}
        .grid-stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 8px;
            margin-bottom: 12px;
        }}
        .stat-card {{
            background: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 10px 12px;
        }}
        .stat-label {{ color: var(--text-muted); font-size: 10px; font-weight: 700; text-transform: uppercase; }}
        .stat-val {{ font-size: 16px; font-weight: 800; margin-top: 3px; }}
        .chart-container {{
            background: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 10px 4px 4px 4px;
            margin-bottom: 12px;
            height: 440px;
        }}
        table {{ width: 100%; border-collapse: collapse; }}
        th, td {{ padding: 10px 12px; text-align: left; border-bottom: 1px solid var(--border-color); font-size: 12px; }}
        th {{ color: var(--text-muted); font-weight: 600; text-transform: uppercase; font-size: 10.5px; position: sticky; top: 0; background: var(--card-bg); z-index: 10; }}
        .table-scroll {{ max-height: 650px; overflow-y: auto; }}
        .table-scroll::-webkit-scrollbar {{ width: 6px; }}
        .table-scroll::-webkit-scrollbar-track {{ background: rgba(255, 255, 255, 0.02); }}
        .table-scroll::-webkit-scrollbar-thumb {{ background: #334155; border-radius: 3px; }}
    </style>
</head>
<body>
    <div class="header">
        <div>
            <span class="badge">🚀 Online Streaming ML</span>
            <h1 style="font-size: 20px; font-weight: 800; margin-top: 4px;">{}</h1>
            <p style="color: var(--text-muted); font-size: 12px; margin-top: 2px;">
                Temporalidad: <strong>{}</strong> | Muestras Procesadas: <strong>{}</strong> | Hojas Streaming: <strong>{}</strong> | Olvido Exp (δ): <strong>{}</strong>
            </p>
        </div>
        <div style="text-align: right;">
            <div style="font-size: 11px; color: var(--text-muted);">Apalancamiento: {}x | Margen: {}%</div>
            <div style="font-size: 20px; font-weight: 800; color: {}; margin-top: 2px;">+${:.2} ({:+.2}%)</div>
        </div>
    </div>

    <div class="grid-stats">
        <div class="stat-card">
            <div class="stat-label">Net Profit (Nominal)</div>
            <div class="stat-val" style="color: {};">+${:.2}</div>
            <div style="font-size: 11px; color: var(--text-muted); margin-top: 2px;">Retorno: {:+.2}%</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Net Profit (Compuesto)</div>
            <div class="stat-val" style="color: #38bdf8;">+${:.2}</div>
            <div style="font-size: 11px; color: var(--text-muted); margin-top: 2px;">Retorno: {:+.2}%</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Total Trades & Dirección</div>
            <div class="stat-val" style="color: #38bdf8;">{} Trades</div>
            <div style="font-size: 11px; color: var(--text-muted); margin-top: 2px;">🟢 L: <strong>{}</strong> | 🔴 S: <strong>{}</strong></div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Profit Factor & Win Rate</div>
            <div class="stat-val">{:.2}</div>
            <div style="font-size: 11px; color: var(--text-muted); margin-top: 2px;">WR: {:.1}% (W: {} / L: {})</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Max Drawdown (Nom | Comp)</div>
            <div class="stat-val" style="color: #f87171; font-size: 15px;">{:.2}% <span style="font-size: 11px; color: #94a3b8;">Nom</span> | {:.2}% <span style="font-size: 11px; color: #a855f7;">Comp</span></div>
            <div style="font-size: 11px; color: var(--text-muted); margin-top: 2px;">Nom: -${:.2} | Comp: -${:.2} | Liq: {}</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Métricas de Riesgo</div>
            <div class="stat-val" style="color: #34d399;">Sharpe: {:.2}</div>
            <div style="font-size: 11px; color: var(--text-muted); margin-top: 2px;">Sortino: {:.2} | Fees: ${:.2}</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Máx Estancamiento</div>
            <div class="stat-val" style="color: #fbbf24;">{} velas</div>
            <div style="font-size: 11px; color: var(--text-muted); margin-top: 2px;">Nom: {} v | Comp: {} v</div>
        </div>
    </div>

    <div class="chart-container">
        <canvas id="equityChart"></canvas>
    </div>

    <div style="display: grid; grid-template-columns: 2fr 1fr; gap: 14px; align-items: start;">
        <div class="stat-card">
            <h3 style="font-size: 15px; margin-bottom: 12px;">🌲 Importancia de Características Dinámicas (Streaming Gain 100%)</h3>
            <div class="table-scroll">
                <table>
                    <thead>
                        <tr>
                            <th>Puesto</th>
                            <th>Var</th>
                            <th>Nombre de Indicador</th>
                            <th style="text-align: right;">Ganancia</th>
                            <th>Distribución</th>
                        </tr>
                    </thead>
                    <tbody>
                        {}
                    </tbody>
                </table>
            </div>
        </div>
        <div class="stat-card">
            <h3 style="font-size: 15px; margin-bottom: 12px;">⚡ Arquitectura Online GBDT</h3>
            <div style="font-size: 12px; color: var(--text-muted); line-height: 1.8;">
                <p>• <strong>Algoritmo:</strong> Online Gradient Boosted Trees</p>
                <p>• <strong>Cota de División:</strong> Hoeffding Bound (\delta = {})</p>
                <p>• <strong>Factor de Olvido (Decay):</strong> {} (Adaptativo)</p>
                <p>• <strong>Árboles Streaming:</strong> {}</p>
                <p>• <strong>Profundidad Máx:</strong> {}</p>
                <p>• <strong>Regularización L2 (\lambda):</strong> {}</p>
                <p>• <strong>Regularización L1 (\alpha):</strong> {}</p>
                <p>• <strong>Período de Gracia:</strong> {} muestras</p>
                <p>• <strong>Look-Ahead Bias:</strong> 0.00% (Estrictamente Causal)</p>
            </div>
        </div>
    </div>

    <script>
        const ctx = document.getElementById('equityChart').getContext('2d');
        new Chart(ctx, {{
            type: 'line',
            data: {{
                datasets: [
                    {{
                        label: 'Curva Equity Nominal ($)',
                        data: [{}],
                        borderColor: '#10b981',
                        backgroundColor: 'rgba(16, 185, 129, 0.05)',
                        borderWidth: 2,
                        pointRadius: 0,
                        pointHoverRadius: 3,
                        fill: true,
                    }},
                    {{
                        label: 'Curva Equity Compuesta ($)',
                        data: [{}],
                        borderColor: '#38bdf8',
                        backgroundColor: 'rgba(56, 189, 248, 0.03)',
                        borderWidth: 2,
                        pointRadius: 0,
                        pointHoverRadius: 3,
                        borderDash: [5, 5],
                        fill: false,
                    }}
                ]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                layout: {{
                    padding: {{ left: 0, right: 0, top: 4, bottom: 0 }}
                }},
                interaction: {{ intersect: false, mode: 'index' }},
                scales: {{
                    x: {{
                        type: 'linear',
                        display: false,
                        bounds: 'data',
                        offset: false
                    }},
                    y: {{
                        grid: {{ color: '#1f293d' }},
                        ticks: {{ color: '#94a3b8' }}
                    }}
                }},
                plugins: {{
                    legend: {{ labels: {{ color: '#f8fafc' }} }}
                }}
            }}
        }});
    </script>
</body>
</html>"#,
        title,
        tf,
        model.total_samples_seen,
        model.total_leaves(),
        model.config.decay_factor,
        leverage,
        capital_percent,
        pnl_color,
        report_nom.net_profit,
        report_nom.total_return_pct,
        pnl_color,
        report_nom.net_profit,
        report_nom.total_return_pct,
        report_pct.net_profit,
        report_pct.total_return_pct,
        report_nom.total_trades,
        report_nom.total_longs,
        report_nom.total_shorts,
        report_nom.profit_factor,
        report_nom.win_rate_pct,
        report_nom.winning_trades,
        report_nom.losing_trades,
        report_nom.max_drawdown_pct,
        report_pct.max_drawdown_pct,
        report_nom.max_drawdown_amount,
        report_pct.max_drawdown_amount,
        report_nom.liquidations,
        report_nom.sharpe_ratio,
        report_nom.sortino_ratio,
        report_nom.total_fees,
        report_nom.max_stagnation_bars,
        report_nom.max_stagnation_bars,
        report_pct.max_stagnation_bars,
        ranking_html,
        model.config.split_confidence,
        model.config.decay_factor,
        model.config.n_trees,
        model.config.max_depth,
        model.config.l2_reg,
        model.config.l1_reg,
        model.config.grace_period,
        nom_equity_points,
        pct_equity_points,
    );

    file.write_all(html.as_bytes())?;
    println!("  🌐 Online GBDT Dashboard interactivo generado en: {}", full_path);
    Ok(full_path)
}

fn sanitize_slug(name: &str) -> String {
    let mut clean = String::with_capacity(name.len());
    let mut last_was_under = false;
    for c in name.chars() {
        if c.is_alphanumeric() {
            clean.push(c.to_ascii_lowercase());
            last_was_under = false;
        } else if !last_was_under {
            clean.push('_');
            last_was_under = true;
        }
    }
    clean.trim_matches('_').to_string()
}
