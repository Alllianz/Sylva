use crate::data::db::Kline;
use crate::trees::sylva_tuner::types::SylvaTreeAutoTuningResult;
use std::fs::{create_dir_all, File};
use std::io::Write;

pub use generate_sylva_tree_tuning_dashboard as generate_astro_tree_tuning_dashboard;

/// Genera un dashboard interactivo HTML para la auto-optimización evolutiva Sylva EVO de árboles
pub fn generate_sylva_tree_tuning_dashboard(
    title: &str,
    tf: &str,
    _klines: &[Kline],
    result: &SylvaTreeAutoTuningResult,
    feature_space_label: &str,
) -> Result<String, std::io::Error> {
    let clean_title = sanitize_slug(title);
    let dashboard_dir = format!("dashboard/{}", tf);
    create_dir_all(&dashboard_dir)?;
    let filename = format!("sylva_tree_tuning_{}_{}.html", tf, clean_title);
    let full_path = format!("{}/{}", dashboard_dir, filename);
    let mut file = File::create(&full_path)?;

    let champ = &result.champion_trial;
    let is_profit = champ.report_nom.net_profit >= 0.0;
    let pnl_color = if is_profit { "#10b981" } else { "#ef4444" };

    let mut leaderboard_rows = String::new();
    for (rank, trial) in result.all_trials.iter().take(25).enumerate() {
        let medal = match rank {
            0 => "🥇 1º",
            1 => "🥈 2º",
            2 => "🥉 3º",
            _ => "   ",
        };
        let rep = &trial.report_nom;
        leaderboard_rows.push_str(&format!(
            r#"<tr>
                <td style="font-weight: 700; color: #f8fafc;">{} {}</td>
                <td style="color: #38bdf8; font-family: monospace;">Gen {} (T#{})</td>
                <td style="text-align: center; color: #94a3b8;">{}</td>
                <td style="text-align: center; color: #94a3b8;">{}</td>
                <td style="text-align: center; color: #94a3b8;">{:.4}</td>
                <td style="text-align: center; color: #94a3b8;">{:.3}</td>
                <td style="text-align: center; font-weight: 600; color: #f8fafc;">{} <span style="font-size: 11px; color: #94a3b8;">(L:{} / S:{})</span></td>
                <td style="text-align: center; color: #38bdf8; font-weight: 600;">{:.1}%</td>
                <td style="text-align: center; color: #fbbf24; font-weight: 600;">{:.2}</td>
                <td style="text-align: center; color: #f59e0b; font-weight: 600;">{} v</td>
                <td style="text-align: right; color: #34d399; font-weight: 700;">{:.2}</td>
                <td style="text-align: right; color: #a855f7;">#{}</td>
                <td style="text-align: right; color: #38bdf8;">#{}</td>
                <td style="text-align: right; color: #fbbf24;">#{}</td>
                <td style="text-align: right; color: #f87171;">#{}</td>
                <td style="text-align: right; font-weight: 700; color: {};">+${:.2}</td>
            </tr>"#,
            rank + 1, medal, trial.generation_idx, trial.trial_idx,
            trial.config.max_depth, trial.config.n_trees, trial.config.decay_factor, trial.config.learning_rate,
            rep.total_trades, rep.total_longs, rep.total_shorts,
            rep.win_rate_pct,
            rep.profit_factor,
            rep.max_stagnation_bars,
            trial.candidate_report.weighted_avg_rank,
            trial.candidate_report.rank_r2,
            trial.candidate_report.rank_slope,
            trial.candidate_report.rank_cap_dd,
            trial.candidate_report.rank_smoothness,
            if rep.net_profit >= 0.0 { "#10b981" } else { "#ef4444" },
            rep.net_profit
        ));
    }

    // Curva de equity del campeón
    let mut nom_equity_points = String::new();
    for (t, eq) in &champ.report_nom.equity_curve {
        nom_equity_points.push_str(&format!("{{ x: {}, y: {:.2} }},", t, eq));
    }

    let mut pct_equity_points = String::new();
    for (t, eq) in &champ.report_pct.equity_curve {
        let safe_eq = eq.max(1.0);
        pct_equity_points.push_str(&format!("{{ x: {}, y: {:.2} }},", t, safe_eq));
    }

    let html = format!(r#"<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <title>Sylva EVO Auto-Tuning Dashboard | Quant Equation Lab</title>
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
            background: linear-gradient(135deg, #1e1b4b, #0f172a);
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
            background: rgba(139, 92, 246, 0.20);
            color: #c084fc;
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
        .toggle-btn {{
            background: #1e293b;
            border: 1px solid #334155;
            color: #94a3b8;
            padding: 5px 12px;
            border-radius: 6px;
            font-size: 11px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s;
        }}
        .toggle-btn:hover {{ background: #334155; color: #fff; }}
        .toggle-btn.active-both {{ background: #3b82f6; color: #fff; border-color: #3b82f6; }}
        .toggle-btn.active-nom {{ background: #10b981; color: #fff; border-color: #10b981; }}
        .toggle-btn.active-pct {{ background: #a855f7; color: #fff; border-color: #a855f7; }}
        table {{ width: 100%; border-collapse: collapse; }}
        th, td {{ padding: 8px 10px; text-align: left; border-bottom: 1px solid var(--border-color); font-size: 11px; }}
        th {{ color: var(--text-muted); font-weight: 600; text-transform: uppercase; font-size: 10.5px; }}
    </style>
</head>
<body>
    <div class="header">
        <div>
            <span class="badge">🧬 Sylva EVO Auto-Tuning Engine</span>
            <h1 style="font-size: 20px; font-weight: 800; margin-top: 4px;">{}</h1>
            <p style="color: var(--text-muted); font-size: 12px; margin-top: 2px;">
                Temporalidad: <strong>{}</strong> | Espacio: <strong>{}</strong> | Total Evaluados: <strong>{}</strong>
            </p>
        </div>
        <div style="text-align: right;">
            <div style="font-size: 11px; color: var(--text-muted);">Puesto Sylva EVO: #1 (Pond: {:.2})</div>
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
            <div class="stat-val" style="color: #a855f7;">+${:.2}</div>
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
        <div class="stat-card">
            <div class="stat-label">Campeón: Depth & Trees</div>
            <div class="stat-val" style="color: #38bdf8;">D: {} | T: {}</div>
            <div style="font-size: 11px; color: var(--text-muted); margin-top: 2px;">Min Leaf: {} | Grace: {}</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Campeón: Decay & LR</div>
            <div class="stat-val" style="color: #34d399;">δ: {:.4}</div>
            <div style="font-size: 11px; color: var(--text-muted); margin-top: 2px;">LR: {:.3} | Col: {:.2}</div>
        </div>
    </div>

    <div style="background: var(--card-bg); border: 1px solid var(--border-color); border-radius: 12px; padding: 14px; margin-bottom: 12px;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; flex-wrap: wrap; gap: 8px;">
            <div style="font-size: 14px; font-weight: 700; color: #f8fafc;">📈 CURVAS DE RENDIMIENTO (EQUITY CURVE)</div>
            <div style="display: flex; gap: 8px;">
                <button id="btnBoth" class="toggle-btn active-both" onclick="switchSylvaMode('both')">🌐 Ver Ambas</button>
                <button id="btnNom" class="toggle-btn" onclick="switchSylvaMode('nom')">💵 Nominal ($)</button>
                <button id="btnPct" class="toggle-btn" onclick="switchSylvaMode('pct')">📈 Compuesta (Escala Log)</button>
            </div>
        </div>
        <div class="chart-container" style="height: 400px; margin-bottom: 0;">
            <canvas id="equityChart"></canvas>
        </div>
    </div>

    <div class="stat-card">
        <h3 style="font-size: 15px; margin-bottom: 12px;">🏆 Leaderboard General Sylva EVO (Ranking Multicriterio por Puestos)</h3>
        <table>
            <thead>
                <tr>
                    <th>Puesto</th>
                    <th>Origen</th>
                    <th style="text-align: center;">Depth</th>
                    <th style="text-align: center;">Trees</th>
                    <th style="text-align: center;">Decay (δ)</th>
                    <th style="text-align: center;">LR (η)</th>
                    <th style="text-align: center;">Trades (L / S)</th>
                    <th style="text-align: center;">Win Rate</th>
                    <th style="text-align: center;">PF</th>
                    <th style="text-align: center;">Estanc.</th>
                    <th style="text-align: right;">Puesto Pond.</th>
                    <th style="text-align: right;">R² (2x)</th>
                    <th style="text-align: right;">Slope</th>
                    <th style="text-align: right;">Cap/DD</th>
                    <th style="text-align: right;">Smooth</th>
                    <th style="text-align: right;">Net Profit</th>
                </tr>
            </thead>
            <tbody>
                {}
            </tbody>
        </table>
    </div>

    <script>
        const ctx = document.getElementById('equityChart').getContext('2d');
        const dsNom = {{
            label: 'Campeón: Curva Equity Nominal ($)',
            data: [{}],
            borderColor: '#10b981',
            backgroundColor: 'rgba(16, 185, 129, 0.05)',
            borderWidth: 2,
            pointRadius: 0,
            pointHoverRadius: 3,
            fill: true,
        }};
        const dsPct = {{
            label: 'Campeón: Curva Equity Compuesta ($ - Escala Log)',
            data: [{}],
            borderColor: '#a855f7',
            backgroundColor: 'rgba(168, 85, 247, 0.03)',
            borderWidth: 2,
            pointRadius: 0,
            pointHoverRadius: 3,
            borderDash: [5, 5],
            fill: false,
        }};

        const chart = new Chart(ctx, {{
            type: 'line',
            data: {{
                datasets: [dsNom, dsPct]
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
                        type: 'linear',
                        grid: {{ color: '#1f293d' }},
                        ticks: {{
                            color: '#94a3b8',
                            callback: function(value) {{ return '$' + Number(value).toLocaleString(); }}
                        }}
                    }}
                }},
                plugins: {{
                    legend: {{ labels: {{ color: '#f8fafc' }} }}
                }}
            }}
        }});

        function switchSylvaMode(mode) {{
            const btnBoth = document.getElementById('btnBoth');
            const btnNom = document.getElementById('btnNom');
            const btnPct = document.getElementById('btnPct');
            btnBoth.className = 'toggle-btn';
            btnNom.className = 'toggle-btn';
            btnPct.className = 'toggle-btn';

            if (mode === 'both') {{
                btnBoth.className = 'toggle-btn active-both';
                chart.data.datasets = [dsNom, dsPct];
                chart.options.scales.y.type = 'linear';
            }} else if (mode === 'nom') {{
                btnNom.className = 'toggle-btn active-nom';
                chart.data.datasets = [dsNom];
                chart.options.scales.y.type = 'linear';
            }} else if (mode === 'pct') {{
                btnPct.className = 'toggle-btn active-pct';
                chart.data.datasets = [dsPct];
                chart.options.scales.y.type = 'logarithmic';
            }}
            chart.update();
        }}
    </script>
</body>
</html>"#,
        title,
        tf,
        feature_space_label,
        result.total_evaluated,
        champ.candidate_report.weighted_avg_rank,
        pnl_color,
        champ.report_nom.net_profit,
        champ.report_nom.total_return_pct,
        pnl_color,
        champ.report_nom.net_profit,
        champ.report_nom.total_return_pct,
        champ.report_pct.net_profit,
        champ.report_pct.total_return_pct,
        champ.report_nom.total_trades,
        champ.report_nom.total_longs,
        champ.report_nom.total_shorts,
        champ.report_nom.profit_factor,
        champ.report_nom.win_rate_pct,
        champ.report_nom.winning_trades,
        champ.report_nom.losing_trades,
        champ.report_nom.max_drawdown_pct,
        champ.report_pct.max_drawdown_pct,
        champ.report_nom.max_drawdown_amount,
        champ.report_pct.max_drawdown_amount,
        champ.report_nom.liquidations,
        champ.report_nom.sharpe_ratio,
        champ.report_nom.sortino_ratio,
        champ.report_nom.total_fees,
        champ.report_nom.max_stagnation_bars,
        champ.report_nom.max_stagnation_bars,
        champ.report_pct.max_stagnation_bars,
        champ.config.max_depth,
        champ.config.n_trees,
        champ.config.min_samples_leaf,
        champ.config.grace_period,
        champ.config.decay_factor,
        champ.config.learning_rate,
        champ.config.colsample_bytree,
        leaderboard_rows,
        nom_equity_points,
        pct_equity_points,
    );

    file.write_all(html.as_bytes())?;
    println!("  🌐 Sylva-Evo Tree Tuning Dashboard interactivo generado en: {}", full_path);
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
