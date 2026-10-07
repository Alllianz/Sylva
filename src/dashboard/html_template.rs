use super::data_exporter::ExportedChartData;
use super::models::{DashboardModelParams, DashboardSummaryMetrics};

pub fn render_html_dashboard(
    params: &DashboardModelParams,
    metrics: &DashboardSummaryMetrics,
    chart_data: &ExportedChartData,
) -> String {
    format!(r#"<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Quant Equation Lab - Dashboard Dual (Compuesto vs Nominal) - {tf} {leverage:.0}X</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Montserrat:wght@300;400;500;600;700;800;900&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg-silver: #0d0f12;
            --pod-black: #161a1f;
            --pod-black-grad: linear-gradient(180deg, #1a1e24 0%, #121519 100%);
            --pod-inner: #0f1216;
            --border-subtle: rgba(255, 255, 255, 0.08);
            --border-highlight: rgba(255, 255, 255, 0.16);
            --text-pure: #FFFFFF;
            --text-silverlight: #EBE6E1;
            --text-silvermid: #A0AAB5;
            --text-silverdark: #6C7885;
            --allianz-red: #FF2E4D;
            --allianz-red-glow: rgba(255, 46, 77, 0.35);
            --accent-green: #00E676;
            --accent-blue: #00B0FF;
        }}

        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Montserrat', sans-serif !important;
        }}

        body {{
            background-color: var(--bg-silver);
            color: var(--text-silverlight);
            min-height: 100vh;
            padding-bottom: 60px;
        }}

        .flag-strip {{
            height: 5px;
            width: 100%;
            background: linear-gradient(90deg, var(--allianz-red), #FF7043, var(--accent-blue));
            box-shadow: 0 2px 10px var(--allianz-red-glow);
        }}

        .container {{
            max-width: 98%;
            margin: 0 auto;
            padding: 10px 14px;
        }}

        .header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 12px 20px;
            background: var(--pod-black-grad);
            border: 1px solid var(--border-subtle);
            border-left: 5px solid var(--allianz-red);
            border-radius: 12px;
            margin-bottom: 12px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
        }}

        .header-title h1 {{
            font-size: 20px;
            font-weight: 900;
            color: var(--text-pure);
            letter-spacing: 0.5px;
        }}

        .header-title p {{
            color: var(--text-silvermid);
            font-size: 12px;
            margin-top: 2px;
        }}

        .badge-tricolor {{
            background: #0f1216;
            border: 1px solid var(--border-highlight);
            color: var(--accent-blue);
            padding: 6px 14px;
            border-radius: 8px;
            font-size: 12px;
            font-weight: 800;
        }}

        /* Métricas Duales */
        .metrics-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
            margin-bottom: 12px;
        }}

        .metric-card {{
            background: var(--pod-black-grad);
            border: 1px solid var(--border-subtle);
            border-radius: 12px;
            padding: 12px 14px;
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4);
        }}

        .card-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
            padding-bottom: 6px;
            border-bottom: 1px solid var(--border-subtle);
        }}

        .card-header h2 {{
            font-size: 13px;
            font-weight: 800;
            color: var(--text-pure);
        }}

        .kpi-row {{
            display: grid;
            grid-template-columns: repeat(5, 1fr);
            gap: 6px;
            margin-bottom: 6px;
        }}

        .kpi-box {{
            background: var(--pod-inner);
            border: 1px solid var(--border-subtle);
            border-radius: 8px;
            padding: 7px 8px;
        }}

        .kpi-label {{
            font-size: 9.5px;
            font-weight: 700;
            color: var(--text-silverdark);
            text-transform: uppercase;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }}

        .kpi-value {{
            font-size: 13.5px;
            font-weight: 800;
            color: var(--text-pure);
            margin-top: 2px;
            white-space: nowrap;
        }}

        .val-pos {{ color: var(--accent-green); }}
        .val-neg {{ color: var(--allianz-red); }}

        /* Contenedor del Gráfico */
        .chart-section {{
            background: var(--pod-black-grad);
            border: 1px solid var(--border-subtle);
            border-radius: 12px;
            padding: 12px 4px 6px 4px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
            margin-bottom: 12px;
        }}

        .chart-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 16px;
            flex-wrap: wrap;
            gap: 12px;
        }}

        .chart-header h3 {{
            font-size: 16px;
            font-weight: 800;
            color: var(--text-pure);
        }}

        .equity-toggle-container {{
            display: flex;
            align-items: center;
            gap: 8px;
            background: #0f1216;
            padding: 4px;
            border-radius: 12px;
            border: 1px solid var(--border-subtle);
        }}

        .toggle-btn {{
            background: transparent;
            border: none;
            color: var(--text-silvermid);
            padding: 8px 16px;
            font-size: 12px;
            font-weight: 700;
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.2s ease;
            display: flex;
            align-items: center;
            gap: 6px;
        }}

        .toggle-btn:hover {{
            color: var(--text-pure);
            background: rgba(255, 255, 255, 0.05);
        }}

        .toggle-btn.active-nom {{
            background: var(--accent-blue);
            color: #0d0f12;
            font-weight: 800;
            box-shadow: 0 2px 10px rgba(0, 176, 255, 0.4);
        }}

        .toggle-btn.active-pct {{
            background: var(--accent-green);
            color: #0d0f12;
            font-weight: 800;
            box-shadow: 0 2px 10px rgba(0, 230, 118, 0.4);
        }}

        .toggle-btn.active-both {{
            background: linear-gradient(90deg, var(--accent-blue), var(--accent-green));
            color: #0d0f12;
            font-weight: 800;
            box-shadow: 0 2px 10px rgba(0, 230, 118, 0.3);
        }}

        .chart-canvas-container {{
            position: relative;
            height: 480px;
            width: 100%;
        }}
    </style>
</head>
<body>
    <div class="flag-strip"></div>
    <div class="container">
        <div class="header">
            <div class="header-title">
                <h1>🚀 {model_name} | DASHBOARD DUAL DE BACKTEST</h1>
                <p>Temporalidad: <strong>{tf}</strong> | Apalancamiento: <strong>{leverage:.0}X</strong> | Margen: <strong>{capital_percent:.1}%</strong> | Período: {min_date} a {max_date}</p>
            </div>
            <div class="badge-tricolor">
                ALLIANZ QUANT LAB
            </div>
        </div>

        <div class="metrics-grid">
            <!-- Cuenta Nominal -->
            <div class="metric-card" style="border-top: 4px solid var(--accent-blue);">
                <div class="card-header">
                    <h2>📊 CUENTA NOMINAL (Margen Fijo Base)</h2>
                    <span style="font-size: 12px; color: var(--accent-blue); font-weight: 800;">BASE FIJA $10,000</span>
                </div>
                <div class="kpi-row">
                    <div class="kpi-box">
                        <div class="kpi-label">Balance Final</div>
                        <div class="kpi-value">${capital_final_nom:.2}</div>
                    </div>
                    <div class="kpi-box">
                        <div class="kpi-label">Retorno Neto</div>
                        <div class="kpi-value {nom_ret_class}">{total_return_nom_pct:+.2}%</div>
                    </div>
                    <div class="kpi-box">
                        <div class="kpi-label">Max Drawdown</div>
                        <div class="kpi-value val-neg">{max_dd_nom_pct:.2}% (${max_dd_nom_dollar:.0})</div>
                    </div>
                    <div class="kpi-box">
                        <div class="kpi-label">Profit Factor</div>
                        <div class="kpi-value">{profit_factor_nom:.2}</div>
                    </div>
                    <div class="kpi-box">
                        <div class="kpi-label">Max Estancamiento</div>
                        <div class="kpi-value" style="color: var(--allianz-gold);">{max_stagnation_nom} velas</div>
                    </div>
                </div>
                <div class="kpi-row">
                    <div class="kpi-box">
                        <div class="kpi-label">Win Rate</div>
                        <div class="kpi-value">{winrate_nom:.1}%</div>
                    </div>
                    <div class="kpi-box">
                        <div class="kpi-label">Total Trades</div>
                        <div class="kpi-value">{total_trades_nom} (L: {longs_count_nom} / S: {shorts_count_nom})</div>
                    </div>
                    <div class="kpi-box">
                        <div class="kpi-label">Sharpe Ratio</div>
                        <div class="kpi-value">{sharpe_nom:.2}</div>
                    </div>
                    <div class="kpi-box">
                        <div class="kpi-label">Sortino Ratio</div>
                        <div class="kpi-value">{sortino_nom:.2}</div>
                    </div>
                    <div class="kpi-box">
                        <div class="kpi-label">Liquidaciones</div>
                        <div class="kpi-value {nom_liq_class}">{liquidations_nom}</div>
                    </div>
                </div>
            </div>

            <!-- Cuenta Compuesta -->
            <div class="metric-card" style="border-top: 4px solid var(--accent-green);">
                <div class="card-header">
                    <h2>📈 CUENTA COMPUESTA (Interés Compuesto Dinámico)</h2>
                    <span style="font-size: 12px; color: var(--accent-green); font-weight: 800;">REINVERSIÓN CONTINUA</span>
                </div>
                <div class="kpi-row">
                    <div class="kpi-box">
                        <div class="kpi-label">Balance Final</div>
                        <div class="kpi-value val-pos">${capital_final_pct:.2}</div>
                    </div>
                    <div class="kpi-box">
                        <div class="kpi-label">Retorno Neto</div>
                        <div class="kpi-value {pct_ret_class}">{total_return_pct_pct:+.2}%</div>
                    </div>
                    <div class="kpi-box">
                        <div class="kpi-label">Max Drawdown</div>
                        <div class="kpi-value val-neg">{max_dd_pct_pct:.2}% (${max_dd_pct_dollar:.0})</div>
                    </div>
                    <div class="kpi-box">
                        <div class="kpi-label">Profit Factor</div>
                        <div class="kpi-value">{profit_factor_pct:.2}</div>
                    </div>
                    <div class="kpi-box">
                        <div class="kpi-label">Max Estancamiento</div>
                        <div class="kpi-value" style="color: var(--allianz-gold);">{max_stagnation_pct} velas</div>
                    </div>
                </div>
                <div class="kpi-row">
                    <div class="kpi-box">
                        <div class="kpi-label">Win Rate</div>
                        <div class="kpi-value">{winrate_pct:.1}%</div>
                    </div>
                    <div class="kpi-box">
                        <div class="kpi-label">Total Trades</div>
                        <div class="kpi-value">{total_trades_pct} (L: {longs_count_pct} / S: {shorts_count_pct})</div>
                    </div>
                    <div class="kpi-box">
                        <div class="kpi-label">Sharpe Ratio</div>
                        <div class="kpi-value">{sharpe_pct:.2}</div>
                    </div>
                    <div class="kpi-box">
                        <div class="kpi-label">Sortino Ratio</div>
                        <div class="kpi-value">{sortino_pct:.2}</div>
                    </div>
                    <div class="kpi-box">
                        <div class="kpi-label">Liquidaciones</div>
                        <div class="kpi-value {pct_liq_class}">{liquidations_pct}</div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Curva de Equity con Selector Interactivo -->
        <div class="chart-section">
            <div class="chart-header">
                <h3>📈 CURVAS DE RENDIMIENTO (EQUITY CURVE)</h3>
                <div class="equity-toggle-container">
                    <button class="toggle-btn active-both" id="btnBoth" onclick="switchMode('both')">🌐 Ver Ambas</button>
                    <button class="toggle-btn" id="btnNom" onclick="switchMode('nom')">💵 Nominal ($)</button>
                    <button class="toggle-btn" id="btnPct" onclick="switchMode('pct')">📈 Porcentual / Compuesto (%)</button>
                </div>
            </div>
            <div class="chart-canvas-container">
                <canvas id="equityChart"></canvas>
            </div>
        </div>
    </div>

    <script>
        const labels = {labels_json};
        const eqNom = {equity_nom_json};
        const eqPct = {equity_pct_json};

        const datasetCompuesto = {{
            label: 'Cuenta Compuesta ($)',
            data: eqPct,
            borderColor: '#00E676',
            backgroundColor: 'rgba(0, 230, 118, 0.08)',
            borderWidth: 2.2,
            pointRadius: 0,
            pointHoverRadius: 3,
            tension: 0.1,
            fill: true
        }};

        const datasetNominal = {{
            label: 'Cuenta Nominal ($)',
            data: eqNom,
            borderColor: '#00B0FF',
            backgroundColor: 'transparent',
            borderWidth: 2.0,
            pointRadius: 0,
            pointHoverRadius: 3,
            tension: 0.1,
            borderDash: [4, 4]
        }};

        const ctx = document.getElementById('equityChart').getContext('2d');
        const chart = new Chart(ctx, {{
            type: 'line',
            data: {{
                labels: labels,
                datasets: [datasetCompuesto, datasetNominal]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                layout: {{
                    padding: {{
                        left: 0,
                        right: 0,
                        top: 4,
                        bottom: 0
                    }}
                }},
                interaction: {{
                    mode: 'index',
                    intersect: false
                }},
                plugins: {{
                    legend: {{
                        labels: {{
                            color: '#FFFFFF',
                            font: {{ weight: 'bold' }}
                        }}
                    }},
                    tooltip: {{
                        backgroundColor: '#161A1F',
                        titleColor: '#FFFFFF',
                        bodyColor: '#EBE6E1',
                        borderColor: 'rgba(255, 255, 255, 0.2)',
                        borderWidth: 1
                    }}
                }},
                scales: {{
                    x: {{
                        bounds: 'data',
                        offset: false,
                        grid: {{ color: 'rgba(255, 255, 255, 0.05)' }},
                        ticks: {{ color: '#8C98A5', maxTicksLimit: 14 }}
                    }},
                    y: {{
                        grid: {{ color: 'rgba(255, 255, 255, 0.05)' }},
                        ticks: {{
                            color: '#8C98A5',
                            callback: function(value) {{
                                return '$' + value.toLocaleString();
                            }}
                        }}
                    }}
                }}
            }}
        }});

        function switchMode(mode) {{
            const btnBoth = document.getElementById('btnBoth');
            const btnNom = document.getElementById('btnNom');
            const btnPct = document.getElementById('btnPct');

            btnBoth.className = 'toggle-btn';
            btnNom.className = 'toggle-btn';
            btnPct.className = 'toggle-btn';

            if (mode === 'both') {{
                btnBoth.className = 'toggle-btn active-both';
                chart.data.datasets = [datasetCompuesto, datasetNominal];
            }} else if (mode === 'nom') {{
                btnNom.className = 'toggle-btn active-nom';
                chart.data.datasets = [datasetNominal];
            }} else if (mode === 'pct') {{
                btnPct.className = 'toggle-btn active-pct';
                chart.data.datasets = [datasetCompuesto];
            }}
            chart.update();
        }}
    </script>
</body>
</html>"#,
        model_name = params.model_name,
        tf = params.tf,
        leverage = params.leverage,
        capital_percent = params.capital_percent,
        min_date = chart_data.min_date_str,
        max_date = chart_data.max_date_str,
        capital_final_nom = metrics.capital_final_nom,
        total_return_nom_pct = metrics.total_return_nom_pct,
        max_dd_nom_pct = metrics.max_dd_nom_pct,
        max_dd_nom_dollar = metrics.max_dd_nom_dollar,
        profit_factor_nom = metrics.profit_factor_nom,
        winrate_nom = metrics.winrate_nom,
        total_trades_nom = metrics.total_trades_nom,
        longs_count_nom = metrics.longs_count_nom,
        shorts_count_nom = metrics.shorts_count_nom,
        sharpe_nom = metrics.sharpe_nom,
        sortino_nom = metrics.sortino_nom,
        max_stagnation_nom = metrics.max_stagnation_nom,
        liquidations_nom = metrics.liquidations_nom,
        nom_ret_class = if metrics.total_return_nom_pct >= 0.0 { "val-pos" } else { "val-neg" },
        nom_liq_class = if metrics.liquidations_nom == 0 { "val-pos" } else { "val-neg" },

        capital_final_pct = metrics.capital_final_pct,
        total_return_pct_pct = metrics.total_return_pct_pct,
        max_dd_pct_pct = metrics.max_dd_pct_pct,
        max_dd_pct_dollar = metrics.max_dd_pct_dollar,
        profit_factor_pct = metrics.profit_factor_pct,
        winrate_pct = metrics.winrate_pct,
        total_trades_pct = metrics.total_trades_pct,
        longs_count_pct = metrics.longs_count_pct,
        shorts_count_pct = metrics.shorts_count_pct,
        sharpe_pct = metrics.sharpe_pct,
        sortino_pct = metrics.sortino_pct,
        max_stagnation_pct = metrics.max_stagnation_pct,
        liquidations_pct = metrics.liquidations_pct,
        pct_ret_class = if metrics.total_return_pct_pct >= 0.0 { "val-pos" } else { "val-neg" },
        pct_liq_class = if metrics.liquidations_pct == 0 { "val-pos" } else { "val-neg" },

        labels_json = chart_data.labels_json,
        equity_nom_json = chart_data.equity_nom_json,
        equity_pct_json = chart_data.equity_pct_json,
    )
}
