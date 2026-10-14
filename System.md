# Documentación Técnica del Sistema Cuantitativo Sylva (Manual Exhaustivo)

**Autor:** Allianz & Antigravity  
**Proyecto:** Sylva Quantitative Engine  
**Lenguaje:** 100% Rust (Edición 2024)  
**Entorno de Ejecución:** Multihilo Nativo CPU (Rayon) + GPU Compute (WebGPU/WGSL)  
**Objetivo:** Arquitectura integral, derivaciones matemáticas de segundo orden, algoritmos streaming de Hoeffding, optimización evolutiva multiobjetivo y catálogo exhaustivo de variables causales.

---

## 1. Composición y Anatomía Hiperdetallada del Proyecto

El sistema cuantitativo **Sylva** está diseñado con una arquitectura modular estricta de cero código fantasma (*zero mocks*), desacoplando la ingesta de datos, la ingeniería de características, los modelos de inferencia, la simulación de ejecución y el trading en vivo en módulos independientes.

```
Sylva/
├── Cargo.toml                              # Manifiesto de dependencias y optimizaciones de release
├── candles.db                              # Base de datos SQLite local en modo WAL de series OHLCV
├── System.md                               # Especificación técnica exhaustiva del sistema
├── docs/                                   # Documentación técnica complementaria
│   ├── ESPECIFICACION_ANTI_OVERFITTING_1D_4H.md
│   └── MEJORAS_ONLINE_GBDT_ARQUITECTURA.md
└── src/
    ├── main.rs                             # Punto de entrada binario y configuración de mimalloc
    ├── lib.rs                              # Exposición de la API interna del motor
    ├── types.rs                            # Tipos base transversales
    ├── data/                               # Subsistema de Ingesta y Persistencia de Series Temporales
    │   ├── mod.rs
    │   ├── db.rs                           # Gestión relacional SQLite (rusqlite), esquemas e índices
    │   └── downloader.rs                   # Cliente HTTP (reqwest) para descarga de Binance REST API
    ├── features/                           # Ingeniería de Variables Cuantitativas Causales
    │   ├── mod.rs
    │   ├── names.rs                        # Nomenclatura oficial y constantes de las 32 características
    │   ├── raw.rs                          # Pipeline matemático de extracción causal en t
    │   ├── indicators.rs                   # Indicadores algorítmicos (EMA, ATR, RSI, MACD, Markov, OLS)
    │   ├── cached.rs                       # Estructura de precomputación en memoria para backtest O(1)
    │   ├── mask.rs                         # Máscaras booleanas de selección de features (Allianz 13, etc.)
    │   └── normalizer.rs                   # Normalizadores Z-Score y Min-Max rodantes sin look-ahead
    ├── trees/                              # Núcleo de Machine Learning Cuantitativo
    │   ├── mod.rs
    │   ├── dataset.rs                      # TabularDataset, muestras forward (x_t, y_t) y extracción
    │   ├── online/                         # Streaming GBDT (Árboles aditivos con cotas de Hoeffding)
    │   │   ├── mod.rs
    │   │   ├── types.rs                    # OnlineGbdtConfig, OnlineLeafStats, OnlineMetrics
    │   │   ├── node.rs                     # OnlineTreeNode (Leaf/Internal), Hoeffding bound
    │   │   ├── tree.rs                     # OnlineDecisionTree, submuestreo de variables
    │   │   ├── gbdt.rs                     # OnlineGbdtModel, boosting secuencial update_step
    │   │   ├── model_wrapper.rs            # Adaptador OnlineGbdtEquationModel para simulación
    │   │   ├── auto_tuner.rs               # Búsqueda en cuadrícula causal
    │   │   └── dashboard.rs                # Generación de dashboard HTML de streaming GBDT
    │   ├── bayesian/                       # Streaming Bayesian GBDT con Incertidumbre NIG
    │   │   ├── mod.rs
    │   │   ├── types.rs                    # BayesianTreeConfig, BayesianPrediction
    │   │   ├── leaf.rs                     # BayesianLeaf: conjugada Normal-Inverse-Gamma
    │   │   ├── node.rs                     # Árbol bayesiano y propagación de distribuciones
    │   │   ├── tree.rs                     # Ensamble con muestreo de Thompson
    │   │   ├── model.rs                    # BayesianOnlineGbdtModel (Alpha ajustado por varianza)
    │   │   ├── model_wrapper.rs            # Adaptador bayesiano para el simulador
    │   │   ├── optimizer.rs                # Optimización Bayesiana autónoma con GP
    │   │   ├── dashboard.rs                # Dashboard de árboles bayesianos
    │   │   └── gp/                         # Procesos Gaussianos analíticos
    │   │       ├── mod.rs
    │   │       ├── kernel.rs               # Núcleos de covarianza Matérn 5/2 y RBF
    │   │       ├── gaussian_process.rs     # Inversión de matrices e inferencia GP
    │   │       ├── acquisition.rs          # Función de adquisición Expected Improvement (EI)
    │   │       └── sampler.rs              # Latin Hypercube Sampling (LHS)
    │   ├── batch/                          # GBDT Batch Clásico con Validación Cruzada
    │   │   ├── mod.rs
    │   │   ├── tree.rs                     # Árbol de regresión batch determinista
    │   │   ├── split.rs                    # Búsqueda de corte de segundo orden
    │   │   ├── gbdt.rs                     # Algoritmo Gradient Boosting clásico con L1/L2
    │   │   ├── purged_cv.rs                # Purged K-Fold Cross Validation con Embargo
    │   │   ├── model_wrapper.rs            # Adaptador de inferencia batch
    │   │   └── auto_tuner.rs               # Auto-tuner batch
    │   ├── gpu/                            # Aceleración por Hardware GPU (WebGPU / WGSL)
    │   │   ├── mod.rs
    │   │   ├── context.rs                  # Adaptador WGPU, Device, Queue en Vulkan/DX12
    │   │   ├── types.rs                    # Estructuras GpuNode y GpuTreeMeta
    │   │   ├── flat_tree.rs                # Serialización de árboles jerárquicos a buffers planos
    │   │   ├── inference_pipeline.rs       # Pipeline de cómputo para inferencia masiva
    │   │   ├── threshold_pipeline.rs       # Pipeline para barrido de parejas de umbrales en VRAM
    │   │   ├── trainer.rs                  # Entrenamiento y reducción en GPU
    │   │   ├── engine.rs                   # Interfaz GpuTreeEngine de alto nivel
    │   │   └── shaders/                    # Código fuente en WebGPU Shading Language
    │   │       ├── tree_infer.wgsl
    │   │       ├── tree_train.wgsl
    │   │       └── threshold_sweep.wgsl
    │   ├── linear/                         # Modelo Lineal Regularizado
    │   │   ├── mod.rs
    │   │   ├── elastic_net.rs              # Elastic Net (L1 Lasso + L2 Ridge) vía Coordinate Descent
    │   │   └── model_wrapper.rs            # Adaptador de inferencia lineal
    │   └── sylva_tuner/                    # Motor de Optimización Evolutiva Sylva EVO
    │       ├── mod.rs
    │       ├── types.rs                    # SylvaTreeCandidateConfig, SylvaTreeTrial
    │       ├── sampler.rs                  # Muestreo LHS por bloques y mutación gaussiana
    │       ├── evaluator.rs                # Evaluador causal multihilo/GPU con división 75/25
    │       ├── optimizer.rs                # Bucle genético generacional y ranking por puestos
    │       └── dashboard.rs                # Generación de dashboard interactivo de evolución
    ├── engine/                             # Motor de Simulación y Ejecución Causal
    │   ├── mod.rs
    │   ├── types.rs                        # BacktestConfig, Position, SignalAction, TradeLog
    │   ├── simulator.rs                    # Backtest vela a vela (Nominal y Compuesto con margen)
    │   ├── dynamic_threshold.rs            # Umbrales adaptativos por ratio de volatilidad ATR5/ATR50
    │   ├── threshold_tuner.rs              # Búsqueda en cuadrícula de umbrales Long/Short
    │   ├── partition.rs                    # Partición cronológica estricta (Warm-up / IS / OOS)
    │   ├── equation_model.rs               # Trait EquationModel para desacoplar modelos y simulación
    │   └── precomputed_model.rs            # Modelo de predicciones vectorizadas en memoria
    ├── metrics/                            # Métricas Estadísticas y Financieras Institucionales
    │   ├── mod.rs
    │   ├── backtest_report.rs              # Sharpe, Sortino, Win Rate, Profit Factor, Drawdowns
    │   ├── terminal_report.rs              # Tablas estructuradas para consola
    │   ├── autotuning.rs                   # Consistencia de pendientes IS/OOS y linealidad R²
    │   ├── smoothness.rs                   # Índice de suavidad de curva y Ulcer Index
    │   └── ranking.rs                      # Algoritmo de ranking multicriterio por puestos ordinales
    ├── dashboard/                          # Generación de Reportes Visuales Autónomos (HTML5/Canvas)
    │   ├── mod.rs
    │   ├── html_template.rs                # Plantilla base en modo oscuro con gráficos embebidos
    │   ├── dual_dashboard.rs               # Reporte comparativo Nominal vs Compuesto
    │   ├── grid_dashboard.rs               # Reporte de candidatos de la cuadrícula evolutiva
    │   ├── models.rs                       # Estructuras de datos para exportación JSON
    │   └── data_exporter.rs                # Serializador de curvas de equidad y trades
    ├── live_trading/                       # Motor de Trading en Vivo (BingX & Paper Trading)
    │   ├── mod.rs
    │   ├── types.rs                        # Tipos de órdenes, balances y posiciones
    │   ├── config.rs                       # Almacenamiento seguro de credenciales API en SQLite
    │   ├── exchange/                       # Conectores de exchanges
    │   │   ├── mod.rs
    │   │   └── bingx.rs                    # Conexión REST a futuros de BingX con firma HMAC-SHA256
    │   ├── account_manager.rs              # Monitoreo de balances y gestión de pools de cuentas
    │   ├── order_executor.rs               # Ejecución de órdenes de mercado y límites de riesgo
    │   ├── signal_evaluator.rs             # Inferencia al milisegundo de cierre de vela
    │   ├── runner.rs                       # Bucle sincrónico infinito por tiempo de vela
    │   ├── sync.rs                         # Sincronizador de velas en tiempo real vía Binance
    │   ├── telemetry.rs                    # Banners y logs de consola por tick
    │   ├── discord.rs                      # Notificaciones enriquecidas a webhooks de Discord
    │   ├── trades_db.rs                    # Base de datos SQLite de órdenes y trades reales
    │   └── menu.rs                         # Interfaz TUI de gestión de trading en vivo
    └── ui/                                 # Interfaz de Usuario de Consola (TUI)
        ├── mod.rs                          # Menú principal con opciones interactivas [1] a [15]
        ├── header.rs                       # Banner institucional ASCII
        ├── prompts.rs                      # Validadores de entrada por teclado
        ├── session.rs                      # Orquestador interactivo de sesiones de análisis
        └── candles_menu.rs                 # Menú interactivo de descarga y mantenimiento de velas
```

---

## 2. Arquitectura del Sistema y Principio 0% Look-Ahead Bias

En finanzas cuantitativas, el *Look-Ahead Bias* (sesgo de anticipación) ocurre cuando una decisión en el instante $t$ utiliza directa o indirectamente información de instantes $t' > t$. Sylva elimina este sesgo por diseño a través de mecanismos verificables en el código:

### 2.1 Partición Temporal y Warm-up Causal
* **Ubicación en Código:** [`src/engine/partition.rs#L11-L45`](file:///c:/Users/Allianz/Downloads/Antigravity/Sylva/src/engine/partition.rs#L11-L45) en la función `get_dataset_partition_indices`.
* **Mecanismo:** La constante `MANDATORY_BACKTEST_START_TS = 1587340800000` fija el 20 de abril de 2020 a las 00:00:00 UTC como inicio obligatorio de las operaciones In-Sample. Todas las velas anteriores a esta marca de tiempo se reservan exclusivamente como **período de calentamiento causal (Warm-up)**.
* **Justificación Matemática:** Indicadores de largo plazo como la media móvil exponencial de 200 períodos ($\text{EMA}_{200}$) requieren al menos 100 a 200 barras para que el error del valor inicial amortiguado decaiga por debajo del $0.1\%$:
  $$\text{EMA}_t = \alpha P_t + (1 - \alpha) \text{EMA}_{t-1}, \quad \text{con } \alpha = \frac{2}{N + 1}$$
  Sin un período de warm-up anterior a la primera operación, los primeros valores de la EMA estarían sesgados artificialmente hacia el precio inicial.

### 2.2 Desacoplamiento del Target Forward
* **Ubicación en Código:** [`src/trees/dataset.rs#L73-L85`](file:///c:/Users/Allianz/Downloads/Antigravity/Sylva/src/trees/dataset.rs#L73-L85) en `TabularDataset::extract_from_klines`.
* **Mecanismo:** El target de predicción para la muestra $t$ es el retorno forward de $H$ velas:
  $$y_t = \frac{\text{Close}_{t+H} - \text{Close}_t}{\text{Close}_t}$$
  Esta etiqueta sólo se asocia al vector $x_t$ durante la fase de entrenamiento In-Sample. Durante la inferencia causal vela a vela en el simulador ([`src/engine/simulator.rs#L48-L60`](file:///c:/Users/Allianz/Downloads/Antigravity/Sylva/src/engine/simulator.rs#L48-L60)) o en trading real ([`src/live_trading/signal_evaluator.rs`](file:///c:/Users/Allianz/Downloads/Antigravity/Sylva/src/live_trading/signal_evaluator.rs)), el modelo sólo recibe el vector $x_t$ derivado estrictamente de información $j \le t$ ([`src/features/raw.rs#L8-L20`](file:///c:/Users/Allianz/Downloads/Antigravity/Sylva/src/features/raw.rs#L8-L20)).

### 2.3 Umbrales Dinámicos Adaptativos Causales
* **Ubicación en Código:** [`src/engine/dynamic_threshold.rs#L19-L38`](file:///c:/Users/Allianz/Downloads/Antigravity/Sylva/src/engine/dynamic_threshold.rs#L19-L38) en `calculate_volatility_factor`.
* **Mecanismo:** Para no asumir una volatilidad constante en el futuro, el umbral de disparo de una señal en la vela $t$ se modula por el ratio entre la volatilidad rápida y la volatilidad lenta:
  $$\text{VolFactor}_t = \text{clamp}\left( \frac{\text{ATR}_5(t)}{\text{ATR}_{50}(t)}, \ 0.5, \ 2.5 \right)$$
  $$\theta_{L, t} = \theta_L \times \text{VolFactor}_t, \quad \theta_{S, t} = \theta_S \times \text{VolFactor}_t$$
  Ambos valores de ATR se calculan usando exclusivamente velas pasadas $j \le t$.

---

## 3. Fundamentos Matemáticos de Gradient Boosted Trees (GBDT)

Un **Gradient Boosted Decision Tree (GBDT)** aproxima una función desconocida $y = f(x)$ mediante un ensamble aditivo de $M$ árboles de regresión:

$$\hat{y}_i^{(M)} = \sum_{m=1}^M \eta f_m(x_i)$$

Donde $\eta \in (0, 1]$ es el shrinkage (learning rate) y cada $f_m(x) = w_{q(x)}$ asigna el vector $x$ a una hoja del árbol mediante una función de partición $q: \mathbb{R}^d \to \{1, \dots, T\}$.

### 3.1 Aproximación de Taylor de Segundo Orden
La función de coste regularizada a minimizar en el paso $m$ es:

$$\mathcal{L}^{(m)} = \sum_{i=1}^n L\left(y_i, \hat{y}_i^{(m-1)} + f_m(x_i)\right) + \Omega(f_m)$$

Con regularización estructural elástica (L1 + L2) sobre las $T$ hojas:
$$\Omega(f_m) = \gamma T + \frac{1}{2}\lambda \sum_{j=1}^T w_j^2 + \alpha \sum_{j=1}^T |w_j|$$

Desarrollando la pérdida $L$ en serie de Taylor de segundo orden en torno a la predicción previa $\hat{y}_i^{(m-1)}$:
$$L\left(y_i, \hat{y}_i^{(m-1)} + f_m(x_i)\right) \approx L\left(y_i, \hat{y}_i^{(m-1)}\right) + g_i f_m(x_i) + \frac{1}{2} h_i f_m^2(x_i)$$

Donde:
$$g_i = \left. \frac{\partial L(y_i, \hat{y})}{\partial \hat{y}} \right|_{\hat{y} = \hat{y}_i^{(m-1)}}, \quad h_i = \left. \frac{\partial^2 L(y_i, \hat{y})}{\partial \hat{y}^2} \right|_{\hat{y} = \hat{y}_i^{(m-1)}}$$

Bajo el error cuadrático medio ($L = \frac{1}{2}(y - \hat{y})^2$):
$$g_i = \hat{y}_i^{(m-1)} - y_i, \quad h_i = 1.0$$

Eliminando la constante $L(y_i, \hat{y}_i^{(m-1)})$, la función objetivo simplificada queda:
$$\tilde{\mathcal{L}}^{(m)} = \sum_{j=1}^T \left[ \left(\sum_{i \in I_j} g_i\right) w_j + \frac{1}{2} \left(\sum_{i \in I_j} h_i + \lambda\right) w_j^2 + \alpha |w_j| \right] + \gamma T$$

Definiendo $G_j = \sum_{i \in I_j} g_i$ y $H_j = \sum_{i \in I_j} h_i$:
$$\tilde{\mathcal{L}}^{(m)} = \sum_{j=1}^T \left[ G_j w_j + \frac{1}{2} (H_j + \lambda) w_j^2 + \alpha |w_j| \right] + \gamma T$$

### 3.2 Derivación Analítica del Peso Óptimo de la Hoja
Para optimizar el peso $w_j$ con regularización L1 ($\alpha > 0$) y L2 ($\lambda > 0$), se aplica el operador de soft-thresholding de cálculo subgradiente:

$$\frac{\partial \tilde{\mathcal{L}}}{\partial w_j} = G_j + (H_j + \lambda) w_j + \alpha \cdot \text{sign}(w_j) = 0$$

Despejando $w_j^*$:
$$w_j^* = \begin{cases} 
-\frac{G_j - \alpha}{H_j + \lambda} & \text{si } G_j > \alpha \\
-\frac{G_j + \alpha}{H_j + \lambda} & \text{si } G_j < -\alpha \\
0 & \text{si } |G_j| \le \alpha 
\end{cases}$$

Esta formulación exacta está implementada en [`src/trees/online/types.rs#L128-L145`](file:///c:/Users/Allianz/Downloads/Antigravity/Sylva/src/trees/online/types.rs#L128-L145):
```rust
pub fn compute_weight(&self, l1_reg: f32, l2_reg: f32) -> f32 {
    if self.sum_h + l2_reg <= 1e-7 {
        return 0.0;
    }
    if l1_reg > 0.0 {
        if self.sum_g > l1_reg {
            -(self.sum_g - l1_reg) / (self.sum_h + l2_reg)
        } else if self.sum_g < -l1_reg {
            -(self.sum_g + l1_reg) / (self.sum_h + l2_reg)
        } else {
            0.0
        }
    } else {
        -self.sum_g / (self.sum_h + l2_reg)
    }
}
```

### 3.3 Score de Calidad y Ganancia de Partición (Gain)
Sustituyendo $w_j^*$ en $\tilde{\mathcal{L}}^{(m)}$, el valor óptimo de pérdida que aporta una hoja es:
$$\text{Score}_j = \frac{G_{j, \text{adj}}^2}{H_j + \lambda}$$
Implementado en [`src/trees/online/types.rs#L147-L166`](file:///c:/Users/Allianz/Downloads/Antigravity/Sylva/src/trees/online/types.rs#L147-L166).

Cuando una hoja se divide en dos nodos hijos ($L$ y $R$), la ganancia de segundo orden es:
$$\text{Gain} = \frac{1}{2} \left[ \frac{G_{L, \text{adj}}^2}{H_L + \lambda} + \frac{G_{R, \text{adj}}^2}{H_R + \lambda} - \frac{G_{\text{root}, \text{adj}}^2}{H_{\text{root}} + \lambda} \right] - \gamma$$
Implementado en [`src/trees/online/node.rs#L137-L139`](file:///c:/Users/Allianz/Downloads/Antigravity/Sylva/src/trees/online/node.rs#L137-L139) y [`src/trees/online/node.rs#L242-L261`](file:///c:/Users/Allianz/Downloads/Antigravity/Sylva/src/trees/online/node.rs#L242-L261).

---

## 4. Implementación del Modelo Online (Streaming Hoeffding GBDT)

A diferencia de los modelos por lotes que recalculan árboles desde cero, [`src/trees/online/`](file:///c:/Users/Allianz/Downloads/Antigravity/Sylva/src/trees/online) opera como un flujo continuo (*streaming*) donde cada vela observada actualiza los pesos de las hojas y la topología de los árboles.

### 4.1 Estructura en Memoria de los Nodos
El árbol se define de forma recursiva en [`src/trees/online/node.rs#L5-L18`](file:///c:/Users/Allianz/Downloads/Antigravity/Sylva/src/trees/online/node.rs#L5-L18):
```rust
pub enum OnlineTreeNode {
    Leaf {
        stats: OnlineLeafStats,
        weight: f32,
    },
    Internal {
        feature_idx: usize,
        threshold: f32,
        gain: f32,
        left: Box<OnlineTreeNode>,
        right: Box<OnlineTreeNode>,
    },
}
```

### 4.2 Factor de Olvido Exponencial ($\delta$)
Para evitar que el modelo se f隸ilice con datos de hace años, los acumuladores estadísticos decaen exponencialmente en cada muestra en [`src/trees/online/types.rs#L90-L126`](file:///c:/Users/Allianz/Downloads/Antigravity/Sylva/src/trees/online/types.rs#L90-L126):
$$G_t = G_{t-1} \cdot \delta + g_t$$
$$H_t = H_{t-1} \cdot \delta + h_t$$

La vida media del conocimiento retenido es:
$$\tau_{1/2} = \frac{\ln(0.5)}{\ln(\delta)}$$

Además, para cada una de las 32 características, se mantienen 8 cuantiles adaptativos entre el mínimo y el máximo observado de la variable:
```rust
for bin in 0..8 {
    let threshold = min_v + range * ((bin as f32 + 1.0) / 9.0);
    if val <= threshold {
        self.split_g_left[feat_idx][bin] = self.split_g_left[feat_idx][bin] * decay + g;
        self.split_h_left[feat_idx][bin] = self.split_h_left[feat_idx][bin] * decay + h;
    } else {
        self.split_g_right[feat_idx][bin] = self.split_g_right[feat_idx][bin] * decay + g;
        self.split_h_right[feat_idx][bin] = self.split_h_right[feat_idx][bin] * decay + h;
    }
}
```

### 4.3 Criterio de División por Desigualdad de Hoeffding
Para determinar matemáticamente si una variable candidata $X_1$ es significativamente superior a la segunda mejor $X_2$ sin ver infinitos datos, se utiliza la **Cota de Hoeffding** ([`src/trees/online/node.rs#L169-L198`](file:///c:/Users/Allianz/Downloads/Antigravity/Sylva/src/trees/online/node.rs#L169-L198)).

Sean $\bar{G}(X_1)$ y $\bar{G}(X_2)$ las ganancias estimadas de los dos mejores cortes, con diferencia $\Delta \bar{G} = \bar{G}(X_1) - \bar{G}(X_2)$.
Con probabilidad $1 - \delta_{\text{hoeff}}$, la verdadera diferencia no dista de la observada en más de:

$$\epsilon = \sqrt{\frac{R^2 \ln(1 / \delta_{\text{hoeff}})}{2n}}$$

Donde:
* $R$ es el rango máximo de la métrica de ganancia.
* $n$ es el número acumulado de muestras en la hoja (`stats.sample_count`).
* $\delta_{\text{hoeff}}$ es `cfg.split_confidence` (por defecto $0.05 \implies 95\%$ confianza).

**Regla de División:**
1. Si $\Delta \bar{G} > \epsilon$, la superioridad de $X_1$ está estadísticamente demostrada.
2. Si $\epsilon < \tau$ (`cfg.tie_threshold`, típicamente $0.05$), hay un empate estadístico indiscutible y esperar más datos es fútil.
3. En cualquiera de los dos casos (y si $\bar{G}(X_1) > \gamma$), el nodo `Leaf` se convierte permanentemente en un nodo `Internal` con dos hojas hijas:
```rust
let delta_gain = best_gain - second_best_gain;
let should_split = delta_gain > epsilon || epsilon < cfg.tie_threshold;

if should_split && best_gain > cfg.gamma {
    *self = OnlineTreeNode::Internal {
        feature_idx: best_feat,
        threshold: best_threshold,
        gain: best_gain,
        left: Box::new(OnlineTreeNode::Leaf { stats: best_left_stats, weight: left_weight }),
        right: Box::new(OnlineTreeNode::Leaf { stats: best_right_stats, weight: right_weight }),
    };
    return true;
}
```

### 4.4 Bucle de Boosting Residual Secuencial
Al recibir la muestra $(x_t, y_t)$, el ensamble ejecuta el boosting paso a paso en [`src/trees/online/gbdt.rs#L64-L106`](file:///c:/Users/Allianz/Downloads/Antigravity/Sylva/src/trees/online/gbdt.rs#L64-L106):
```rust
let mut residual = current_pred - target;
let hessian = 1.0f32;
let lr = self.config.learning_rate;

for tree in &mut self.trees {
    let g = residual;
    let _split_occurred = tree.update(features, g, hessian, &self.config);
    let tree_pred = tree.predict(features);
    residual -= lr * tree_pred; // Contracción residual secuencial
}
```

---

## 5. Algoritmo de Búsqueda y Auto-Optimización Evolutiva (Sylva EVO)

El motor evolutivo en [`src/trees/sylva_tuner/`](file:///c:/Users/Allianz/Downloads/Antigravity/Sylva/src/trees/sylva_tuner) resuelve el problema de optimización de hiperparámetros en un espacio continuo no lineal de 14 dimensiones.

### Punto 1: Muestreador Exploratorio por Bloques Coordinados
* **Archivo y Líneas:** [`src/trees/sylva_tuner/sampler.rs#L41-L108`](file:///c:/Users/Allianz/Downloads/Antigravity/Sylva/src/trees/sylva_tuner/sampler.rs#L41-L108) en `sample_exploratory`.
* **Coordinación Estructural:** No genera hiperparámetros de manera independiente (lo que produciría combinaciones absurdas como profundidad 6 con learning rate 0.1 en marcos macro). Los agrupa en 5 bloques coordinados:
  * **Bloque 1 (Capacidad):** Si el horizonte es macro ($H \ge 2$), acota `max_depth` a $1 \dots 3$ (*Decision Stumps*) y `min_samples_leaf` a $35 \dots 75$ velas. Si la profundidad es 1 o 2, eleva `n_trees` a $15 \dots 40$.
  * **Bloque 2 (Dinámica Temporal):** Muestreo de learning rate en espacio logarítmico: $\ln(\eta) \sim \mathcal{U}(-4.8, -2.7) \implies \eta \in [0.008, 0.07]$. Factor de decaimiento acotado en $[0.9980, 0.9999]$ para macro y $[0.9960, 0.9995]$ para intradía.
  * **Bloque 3 (Regularización):** L2 en escala logarítmica: $\ln(\lambda) \sim \mathcal{U}(-1.5, 2.0) \implies \lambda \in [0.2, 10.0]$. L1 con 40% de probabilidad de ser 0.0.
  * **Bloque 4 (Bayesiano):** Precisión a priori $\kappa_0 \in [0.2, 4.0]$ y penalización por incertidumbre $\kappa_{\text{risk}} \in [0.0, 1.8]$.
  * **Bloque 5 (Horizonte Causal Multi-Vela Forward):** Selección de $H \in \{2, 3, 5\}$ en 1D o $\{2, 3, 6, 12\}$ en 4H.

### Punto 2: Evaluación Causal Paralela
* **Archivo y Líneas:** [`src/trees/sylva_tuner/evaluator.rs#L61-L198`](file:///c:/Users/Allianz/Downloads/Antigravity/Sylva/src/trees/sylva_tuner/evaluator.rs#L61-L198) en `evaluate_candidate`.
* **Partición Causal In-Sample (75% Train / 25% Validación):** [`evaluator.rs#L78-L83`](file:///c:/Users/Allianz/Downloads/Antigravity/Sylva/src/trees/sylva_tuner/evaluator.rs#L78-L83). El modelo se entrena sobre `train_dataset` y se evalúa en el 25% posterior sin Look-Ahead Bias.
* **Calibración de Umbrales:** [`evaluator.rs#L115-L130`](file:///c:/Users/Allianz/Downloads/Antigravity/Sylva/src/trees/sylva_tuner/evaluator.rs#L115-L130) con `ThresholdTuner` sobre la malla bidimensional de umbrales Long y Short.
* **Simulación Dual:** Simula simultáneamente en modo **Nominal** (capital fijo) y **Compuesto** (interés compuesto institucional con apalancamiento) en [`evaluator.rs#L145-L146`](file:///c:/Users/Allianz/Downloads/Antigravity/Sylva/src/trees/sylva_tuner/evaluator.rs#L145-L146) y [`#L189-L190`](file:///c:/Users/Allianz/Downloads/Antigravity/Sylva/src/trees/sylva_tuner/evaluator.rs#L189-L190).

### Punto 3: Selección de Padres y Elitismo Estricto
* **Archivo y Líneas:** [`src/trees/sylva_tuner/optimizer.rs#L180-L186`](file:///c:/Users/Allianz/Downloads/Antigravity/Sylva/src/trees/sylva_tuner/optimizer.rs#L180-L186).
* Al final de cada generación, toda la población histórica acumulada se clasifica y se seleccionan los **Top 4 Progenitores Élite**. El mejor candidato histórico (*Best Ever*) se mantiene intacto, asegurando una evolución no destructiva.

### Punto 4: Mutación Genética Adaptativa
* **Archivo y Líneas:** [`src/trees/sylva_tuner/sampler.rs#L111-L174`](file:///c:/Users/Allianz/Downloads/Antigravity/Sylva/src/trees/sylva_tuner/sampler.rs#L111-L174) en `mutate_parent`.
* **Distribución de Mutación:**
  * 70% de probabilidad: Selecciona uno de los 5 bloques funcionales del padre y aplica una perturbación gaussiana $\mathcal{N}(0, \sigma^2)$ exclusivamente a ese bloque, manteniendo intacta la sinergia de los otros 4 bloques.
  * 15% de probabilidad: Clona al padre directamente para explorar variaciones estocásticas de umbrales.
  * 15% de probabilidad: Genera un individuo exploratorio nuevo (LHS) en [`optimizer.rs#L200-L202`](file:///c:/Users/Allianz/Downloads/Antigravity/Sylva/src/trees/sylva_tuner/optimizer.rs#L200-L202) para preservar la diversidad y escapar de óptimos locales.

### Punto 5: Ranking Multicriterio por Puestos Ordinales (Sylva EVO Max)
* **Archivo y Líneas:** [`src/metrics/ranking.rs#L23-L135`](file:///c:/Users/Allianz/Downloads/Antigravity/Sylva/src/metrics/ranking.rs#L23-L135) en `compute_multicriteria_rankings`.
* Para evitar sesgos monetarios (donde una estrategia con alta volatilidad o un trade de suerte supera a modelos estables), los candidatos son ordenados por su **puesto ordinal relativo (1º a N-ésimo)** en 4 dimensiones independientes:
  1. **Consistencia de Pendiente IS/OOS (*Slope Ratio*):** [`#L39-L55`](file:///c:/Users/Allianz/Downloads/Antigravity/Sylva/src/metrics/ranking.rs#L39-L55). Ordena por cercanía a 1.0 de $|\text{Slope}_{\text{val}} / \text{Slope}_{\text{train}} - 1.0|$.
  2. **Ratio Capital / Max Drawdown:** [`#L57-L69`](file:///c:/Users/Allianz/Downloads/Antigravity/Sylva/src/metrics/ranking.rs#L57-L69). Ordena de mayor a menor $\text{Capital Final} / \text{Max DD Dólares}$.
  3. **Linealidad $R^2$ Compuesta (Ponderación 2x):** [`#L71-L84`](file:///c:/Users/Allianz/Downloads/Antigravity/Sylva/src/metrics/ranking.rs#L71-L84). Evalúa la cercanía a una línea recta ideal, penalizando curvas con escalones o serruchos.
  4. **Suavidad y Resistencia (Ulcer Index):** [`#L86-L99`](file:///c:/Users/Allianz/Downloads/Antigravity/Sylva/src/metrics/ranking.rs#L86-L99). Penaliza caídas profundas y estancamiento prolongado.
* **Fórmula de Ponderación Final:** [`#L110-L114`](file:///c:/Users/Allianz/Downloads/Antigravity/Sylva/src/metrics/ranking.rs#L110-L114):
  $$\text{Weighted Rank} = \frac{1 \cdot R_{\text{slope}} + 1 \cdot R_{\text{cap\_dd}} + 2 \cdot R_{R^2} + 1 \cdot R_{\text{smooth}}}{5.0}$$
* El **Campeón Absoluto** es el candidato que minimiza `Weighted Rank` ([`src/trees/sylva_tuner/optimizer.rs#L264-L272`](file:///c:/Users/Allianz/Downloads/Antigravity/Sylva/src/trees/sylva_tuner/optimizer.rs#L264-L272)).

---

## 6. Catálogo Exhaustivo de las 32 Features Cuantitativas

Declaradas en [`src/features/names.rs#L4-L37`](file:///c:/Users/Allianz/Downloads/Antigravity/Sylva/src/features/names.rs#L4-L37) y calculadas en [`src/features/raw.rs#L25-L297`](file:///c:/Users/Allianz/Downloads/Antigravity/Sylva/src/features/raw.rs#L25-L297):

```rust
// Código fuente en src/features/raw.rs:
let k = &klines[t];
let open_t = if k.open > 0.0 { k.open } else { 1e-5 };
let close_t = if k.close > 0.0 { k.close } else { 1e-5 };
```

### F1: Long-term Rel Volume (SMA 10k)
* **Código:** [`src/features/raw.rs#L26-L32`](file:///c:/Users/Allianz/Downloads/Antigravity/Sylva/src/features/raw.rs#L26-L32)
* **Fórmula:** $F_1 = \frac{\text{Volume}_t}{\overline{\text{Volume}}_{10000}}$
* **Significado:** Ratio del volumen actual respecto a la media de 10.000 barras. Identifica inyecciones masivas de liquidez institucional.

### F2: Normalized Total Range
* **Código:** [`src/features/raw.rs#L34-L35`](file:///c:/Users/Allianz/Downloads/Antigravity/Sylva/src/features/raw.rs#L34-L35)
* **Fórmula:** $F_2 = \frac{\text{High}_t - \text{Low}_t}{\text{Open}_t}$
* **Significado:** Volatilidad total intra-barra normalizada por el precio de apertura.

### F3: Upper Wick Ratio
* **Código:** [`src/features/raw.rs#L37-L38`](file:///c:/Users/Allianz/Downloads/Antigravity/Sylva/src/features/raw.rs#L37-L38)
* **Fórmula:** $F_3 = \frac{\text{High}_t - \text{Open}_t}{\text{Open}_t}$
* **Significado:** Mecha superior. Mide el rechazo en zonas altas y la presión de oferta de los vendedores.

### F4: Lower Wick Ratio
* **Código:** [`src/features/raw.rs#L40-L41`](file:///c:/Users/Allianz/Downloads/Antigravity/Sylva/src/features/raw.rs#L40-L41)
* **Fórmula:** $F_4 = \frac{\text{Open}_t - \text{Low}_t}{\text{Open}_t}$
* **Significado:** Mecha inferior. Mide la absorción de ventas por compradores agresivos en soporte.

### F5: Real Candle Body
* **Código:** [`src/features/raw.rs#L43-L44`](file:///c:/Users/Allianz/Downloads/Antigravity/Sylva/src/features/raw.rs#L43-L44)
* **Fórmula:** $F_5 = \frac{\text{Open}_t - \text{Close}_t}{\text{Open}_t}$
* **Significado:** Magnitud y polaridad del cuerpo real de la vela. Positivo en barras rojas, negativo en verdes.

### F6: 10-Bar Return Momentum
* **Código:** [`src/features/raw.rs#L46-L52`](file:///c:/Users/Allianz/Downloads/Antigravity/Sylva/src/features/raw.rs#L46-L52)
* **Fórmula:** $F_6 = \frac{\text{Close}_t - \text{Close}_{t-10}}{\text{Close}_{t-10}}$
* **Significado:** Retorno acumulado en 10 velas. Cuantifica la velocidad direccional a medio plazo.

### F7: OLS Linear Slope (10 Bars)
* **Código:** [`src/features/raw.rs#L54-L63`](file:///c:/Users/Allianz/Downloads/Antigravity/Sylva/src/features/raw.rs#L54-L63)
* **Fórmula:** $F_7 = \frac{1}{\text{Close}_t} \left( \frac{\sum_{i=0}^9 (i - 4.5) \cdot \text{Close}_{t-9+i}}{82.5} \right)$
* **Significado:** Pendiente por regresión lineal por mínimos cuadrados ordinarios sobre las últimas 10 barras.

### F8: Donchian Stochastic Pos (10 Bars)
* **Código:** [`src/features/raw.rs#L64-L80`](file:///c:/Users/Allianz/Downloads/Antigravity/Sylva/src/features/raw.rs#L64-L80)
* **Fórmula:** $F_8 = \frac{\text{Close}_t - \min_{10}(\text{Low})}{\max_{10}(\text{High}) - \min_{10}(\text{Low})}$
* **Significado:** Oscilador estocástico en $[0, 1]$. Detecta compresión de precios en los extremos del canal.

### F9: Price Z-Score (10 Bars)
* **Código:** [`src/features/raw.rs#L82-L98`](file:///c:/Users/Allianz/Downloads/Antigravity/Sylva/src/features/raw.rs#L82-L98)
* **Fórmula:** $F_9 = \frac{\text{Close}_t - \mu_{10}}{\sigma_{10}}$
* **Significado:** Desviaciones estándar que separan al precio actual de su media móvil de 10 barras.

### F10: Slope Acceleration
* **Código:** [`src/features/raw.rs#L100-L109`](file:///c:/Users/Allianz/Downloads/Antigravity/Sylva/src/features/raw.rs#L100-L109)
* **Fórmula:** $F_{10} = \frac{\text{Slope}_t - \text{Slope}_{t-1}}{\text{Close}_t}$
* **Significado:** Segunda derivada temporal del precio. Mide la aceleración o agotamiento de la tendencia.

### F11: Kaufman Efficiency Ratio
* **Código:** [`src/features/raw.rs#L110-L120`](file:///c:/Users/Allianz/Downloads/Antigravity/Sylva/src/features/raw.rs#L110-L120)
* **Fórmula:** $F_{11} = \frac{|\text{Close}_t - \text{Close}_{t-10}|}{\sum_{j=t-9}^t |\text{Close}_j - \text{Close}_{j-1}|}$
* **Significado:** Eficiencia fractal en $[0, 1]$. Discierne tendencias directas ($ER \approx 1$) de oscilaciones erráticas ($ER \approx 0$).

### F12: Markov Current State ($S_t / 5.0$)
* **Código:** [`src/features/raw.rs#L123-L126`](file:///c:/Users/Allianz/Downloads/Antigravity/Sylva/src/features/raw.rs#L123-L126)
* **Fórmula:** $F_{12} = \frac{S_t}{5.0}, \quad S_t \in \{0 \dots 5\}$
* **Significado:** Estado discreto actual de la Cadena de Markov (retorno, volatilidad y volumen combinados).

### F13: Markov Previous State ($S_{t-1} / 5.0$)
* **Código:** [`src/features/raw.rs#L127-L130`](file:///c:/Users/Allianz/Downloads/Antigravity/Sylva/src/features/raw.rs#L127-L130)
* **Fórmula:** $F_{13} = \frac{S_{t-1}}{5.0}$
* **Significado:** Estado discreto de la vela precedente, permitiendo aprender matrices de transición de régimen.

### F14: Dist to EMA-20
* **Código:** [`src/features/raw.rs#L132-L139`](file:///c:/Users/Allianz/Downloads/Antigravity/Sylva/src/features/raw.rs#L132-L139)
* **Fórmula:** $F_{14} = \frac{\text{Close}_t - \text{EMA}_{20}(t)}{\text{EMA}_{20}(t)}$
* **Significado:** Desviación porcentual con respecto a la media móvil exponencial de 20 períodos.

### F15: Dist to Macro EMA-200 (HL/2)
* **Código:** [`src/features/raw.rs#L140-L147`](file:///c:/Users/Allianz/Downloads/Antigravity/Sylva/src/features/raw.rs#L140-L147)
* **Fórmula:** $F_{15} = \frac{\text{Close}_t - \text{EMA}_{200}(\frac{H+L}{2})}{\text{EMA}_{200}}$
* **Significado:** Desviación porcentual al centro de gravedad institucional de largo plazo.

### F16: Macro Regime Flag (Close > EMA200)
* **Código:** [`src/features/raw.rs#L148-L150`](file:///c:/Users/Allianz/Downloads/Antigravity/Sylva/src/features/raw.rs#L148-L150)
* **Fórmula:** $F_{16} = \mathbb{I}(\text{Close}_t > \text{EMA}_{200}(t))$
* **Significado:** Variable booleana ($1.0$ alcista, $0.0$ bajista) que define el sesgo macro del mercado.

### F17: Medium-term Rel Volume (SMA 20)
* **Código:** [`src/features/raw.rs#L152-L159`](file:///c:/Users/Allianz/Downloads/Antigravity/Sylva/src/features/raw.rs#L152-L159)
* **Fórmula:** $F_{17} = \frac{\text{Volume}_t}{\overline{\text{Volume}}_{20}}$
* **Significado:** Volumen relativo normalizado por la actividad de los últimos 20 períodos.

### F18: Volume Flow Ratio (SMA5 / SMA20)
* **Código:** [`src/features/raw.rs#L160-L167`](file:///c:/Users/Allianz/Downloads/Antigravity/Sylva/src/features/raw.rs#L160-L167)
* **Fórmula:** $F_{18} = \frac{\overline{\text{Volume}}_5}{\overline{\text{Volume}}_{20}}$
* **Significado:** Aceleración o contracción del flujo de volumen a corto plazo frente a su media de mediano plazo.

### F19: Volume Adjusted Amplitude
* **Código:** [`src/features/raw.rs#L168-L170`](file:///c:/Users/Allianz/Downloads/Antigravity/Sylva/src/features/raw.rs#L168-L170)
* **Fórmula:** $F_{19} = \frac{F_2}{F_{17} + 10^{-5}}$
* **Significado:** Rango intra-vela por unidad de volumen relativo. Detecta falsas rupturas sin liquidez.

### F20: Volume-Price Trend (VPT 10)
* **Código:** [`src/features/raw.rs#L171-L181`](file:///c:/Users/Allianz/Downloads/Antigravity/Sylva/src/features/raw.rs#L171-L181)
* **Fórmula:** $F_{20} = \frac{\sum_{j=t-9}^t (\text{Close}_j - \text{Close}_{j-1}) \cdot \text{Volume}_j}{\text{Close}_t \cdot \overline{\text{Volume}}_{10000}}$
* **Significado:** Tendencia de volumen-precio acumulada en 10 barras, capturando divergencias de flujo.

### F21: Directional Signed Volume Force
* **Código:** [`src/features/raw.rs#L182-L184`](file:///c:/Users/Allianz/Downloads/Antigravity/Sylva/src/features/raw.rs#L182-L184)
* **Fórmula:** $F_{21} = \text{sign}(\text{Close}_t - \text{Open}_t) \cdot \ln(\text{Volume}_t + 1)$
* **Significado:** Presión neta de compradores o vendedores con compresión logarítmica.

### F22: ATR(10) Normalized / Close
* **Código:** [`src/features/raw.rs#L186-L189`](file:///c:/Users/Allianz/Downloads/Antigravity/Sylva/src/features/raw.rs#L186-L189)
* **Fórmula:** $F_{22} = \frac{\text{ATR}_{10}(t)}{\text{Close}_t}$
* **Significado:** Average True Range de 10 barras expresado como fracción del precio (volatilidad normalizada).

### F23: MACD Histogram / Close
* **Código:** [`src/features/raw.rs#L190-L193`](file:///c:/Users/Allianz/Downloads/Antigravity/Sylva/src/features/raw.rs#L190-L193)
* **Fórmula:** $F_{23} = \frac{\text{MACD\_Hist}_{12, 26, 9}(t)}{\text{Close}_t}$
* **Significado:** Histograma MACD libre de escala respecto al nivel absoluto de precio.

### F24: Volatility Ratio (ATR5 / ATR50)
* **Código:** [`src/features/raw.rs#L194-L201`](file:///c:/Users/Allianz/Downloads/Antigravity/Sylva/src/features/raw.rs#L194-L201)
* **Fórmula:** $F_{24} = \frac{\text{ATR}_5(t)}{\text{ATR}_{50}(t)}$
* **Significado:** Ratio de compresión/expansión de volatilidad rápida frente a lenta.

### F25: Channel Width (10 Bars)
* **Código:** [`src/features/raw.rs#L202-L208`](file:///c:/Users/Allianz/Downloads/Antigravity/Sylva/src/features/raw.rs#L202-L208)
* **Fórmula:** $F_{25} = \frac{\max_{10}(\text{High}) - \min_{10}(\text{Low})}{\mu_{10}(\text{Close})}$
* **Significado:** Amplitud relativa de las bandas de Donchian respecto a la media de precios.

### F26: Intra-Candle Volatility ROC
* **Código:** [`src/features/raw.rs#L209-L221`](file:///c:/Users/Allianz/Downloads/Antigravity/Sylva/src/features/raw.rs#L209-L221)
* **Fórmula:** $F_{26} = \frac{F_{2, t} - F_{2, t-10}}{F_{2, t-10}}$
* **Significado:** Tasa de cambio de la volatilidad intra-vela en 10 barras.

### F27: RSI-14 Centered ([-0.5, +0.5])
* **Código:** [`src/features/raw.rs#L222-L224`](file:///c:/Users/Allianz/Downloads/Antigravity/Sylva/src/features/raw.rs#L222-L224)
* **Fórmula:** $F_{27} = \frac{\text{RSI}_{14}(t)}{100.0} - 0.5$
* **Significado:** Índice de Fuerza Relativa centrado en cero para balancear las divisiones del árbol.

### F28: Parkinson Extreme Volatility
* **Código:** [`src/features/raw.rs#L225-L236`](file:///c:/Users/Allianz/Downloads/Antigravity/Sylva/src/features/raw.rs#L225-L236)
* **Fórmula:** $F_{28} = \sqrt{\frac{\sum_{j=t-9}^t \ln^2(\text{High}_j / \text{Low}_j)}{40 \ln(2)}}$
* **Significado:** Estimador de volatilidad basado en extremos High/Low, con menor varianza muestral que el desvío estándar de cierres.

### F29: Weekly Cycle Sine
* **Código:** [`src/features/raw.rs#L241-L243`](file:///c:/Users/Allianz/Downloads/Antigravity/Sylva/src/features/raw.rs#L241-L243)
* **Fórmula:** $F_{29} = \sin\left(\frac{\text{Weekday}_t \cdot 2\pi}{7}\right)$
* **Significado:** Componente armónica sinusoidal para capturar estacionalidad de día de semana.

### F30: Weekly Cycle Cosine
* **Código:** [`src/features/raw.rs#L244-L246`](file:///c:/Users/Allianz/Downloads/Antigravity/Sylva/src/features/raw.rs#L244-L246)
* **Fórmula:** $F_{30} = \cos\left(\frac{\text{Weekday}_t \cdot 2\pi}{7}\right)$
* **Significado:** Componente armónica cosinusoidal complementaria.

### F31: Fractional Differentiation (Memory $d=0.40$)
* **Código:** [`src/features/raw.rs#L247-L259`](file:///c:/Users/Allianz/Downloads/Antigravity/Sylva/src/features/raw.rs#L247-L259)
* **Fórmula:** $F_{31} = \sum_{k=0}^9 w_k (\ln(\text{Close}_{t-k}) - \ln(\text{Close}_t))$
* **Pesos:** $[1.0, -0.4, -0.12, -0.064, -0.0416, -0.029952, -0.0229632, -0.01837056, -0.015155712, -0.0127981568]$.
* **Significado:** Diferenciación Fraccionaria ($d = 0.40$) según López de Prado (2018). Convierte la serie en estacionaria preservando la memoria histórica de largo plazo.

### F32: Account Commercial Position State
* **Código:** [`src/features/raw.rs#L261-L262`](file:///c:/Users/Allianz/Downloads/Antigravity/Sylva/src/features/raw.rs#L261-L262)
* **Fórmula:** $F_{32} = \text{Posición Actual} \in \{+1.0, -1.0, 0.0\}$
* **Significado:** Estado de exposición de la cuenta, permitiendo al árbol condicionar señales de cierre o reversión.

---

### 6.1 Poda Selectiva Cuantitativa: Modelo GBDT Allianz (13 Activas vs 19 Podadas)
Implementada en [`src/features/mask.rs#L58-L87`](file:///c:/Users/Allianz/Downloads/Antigravity/Sylva/src/features/mask.rs#L58-L87) en `FeatureMask::new_allianz_custom`:

* **19 Variables Podadas:** `F1, F2, F3, F7, F8, F12, F15, F19, F21, F22, F23, F24, F25, F26, F27, F28, F29, F30, F32`.
  * *Razón Técnica:* En marcos macro y swing (1D y 4H), la microestructura intra-vela (mecha superior F3, rango F2, amplitude F19) y los osciladores simples desfasados (F7, F8, F25) agregan ruido idiosincrático y causan sobreajuste severo en muestras pequeñas ($N \approx 2.500$ barras en 1D).
* **13 Variables Activas Retenidas:**
  1. `F4`: Lower Wick Ratio (Absorción de ventas en suelo).
  2. `F5`: Real Candle Body (Dirección neta de la vela).
  3. `F6`: 10-Bar Return Momentum (Inercia direccional de medio plazo).
  4. `F9`: Price Z-Score (Desviación estadística respecto a la media).
  5. `F10`: Slope Acceleration (Segunda derivada de la tendencia).
  6. `F11`: Kaufman Efficiency Ratio (Calidad fractal de la tendencia vs rango).
  7. `F13`: Markov Previous State (Memoria del régimen estocástico previo).
  8. `F14`: Dist to EMA-20 (Elasticidad de reversión a la media).
  9. `F16`: Macro Regime Flag (Filtro institucional Close > EMA200).
  10. `F17`: Medium-term Rel Volume (Validación de volumen contemporáneo).
  11. `F18`: Volume Flow Ratio (Aceleración de liquidez SMA5 / SMA20).
  12. `F20`: Volume-Price Trend (Divergencias volumen-precio).
  13. `F31`: Fractional Differentiation (Memoria persistente estacionaria con $d=0.40$).

---

## 7. Módulos Adicionales y Ecosistema de Producción

### 7.1 Árboles Bayesianos Online (Incertidumbre NIG)
* **Ubicación:** [`src/trees/bayesian/leaf.rs#L1-L71`](file:///c:/Users/Allianz/Downloads/Antigravity/Sylva/src/trees/bayesian/leaf.rs#L1-L71) y [`src/trees/bayesian/model.rs#L8-L149`](file:///c:/Users/Allianz/Downloads/Antigravity/Sylva/src/trees/bayesian/model.rs#L8-L149).
* Cada hoja modela la distribución de retornos mediante una conjugada Normal-Inverse-Gamma (NIG) parametrizada por $(\mu, \kappa, \alpha, \beta)$ ([`leaf.rs#L2-L14`](file:///c:/Users/Allianz/Downloads/Antigravity/Sylva/src/trees/bayesian/leaf.rs#L2-L14)).
* En cada paso streaming, los parámetros se actualizan analíticamente con decaimiento ([`leaf.rs#L28-L50`](file:///c:/Users/Allianz/Downloads/Antigravity/Sylva/src/trees/bayesian/leaf.rs#L28-L50)).
* **Ajuste del Alpha por Incertidumbre:** [`model.rs#L67-L72`](file:///c:/Users/Allianz/Downloads/Antigravity/Sylva/src/trees/bayesian/model.rs#L67-L72):
  $$\alpha_{\text{adj}} = \text{sign}(\mu) \cdot \max(|\mu| - \kappa_{\text{risk}} \cdot \sigma, \ 0)$$
  Si la varianza epistémica (falta de datos) o aleatoria (ruido de mercado) es alta, la señal decae automáticamente a cero.

### 7.2 Aceleración por Hardware en GPU (WebGPU / WGSL)
* **Ubicación:** [`src/trees/gpu/`](file:///c:/Users/Allianz/Downloads/Antigravity/Sylva/src/trees/gpu).
* Serialización a buffers lineales contiguos de 32 bytes (`GpuNode`) en [`src/trees/gpu/flat_tree.rs#L37-L58`](file:///c:/Users/Allianz/Downloads/Antigravity/Sylva/src/trees/gpu/flat_tree.rs#L37-L58).
* Inferencia vectorial en paralelo en compute shader WGSL con grupos de trabajo de 64 hilos en [`src/trees/gpu/shaders/tree_infer.wgsl#L35-L68`](file:///c:/Users/Allianz/Downloads/Antigravity/Sylva/src/trees/gpu/shaders/tree_infer.wgsl#L35-L68).
* Barrido masivo de parejas de umbrales en VRAM en [`src/trees/gpu/shaders/threshold_sweep.wgsl`](file:///c:/Users/Allianz/Downloads/Antigravity/Sylva/src/trees/gpu/shaders/threshold_sweep.wgsl).

### 7.3 Motor de Trading en Vivo y Conector BingX
* **Ubicación:** [`src/live_trading/`](file:///c:/Users/Allianz/Downloads/Antigravity/Sylva/src/live_trading).
* **Bucle de Ejecución en Tiempo Real:** [`src/live_trading/runner.rs#L20-L148`](file:///c:/Users/Allianz/Downloads/Antigravity/Sylva/src/live_trading/runner.rs#L20-L148). Monitorea el cierre exacto de cada vela vía Binance ([`sync.rs#L15-L65`](file:///c:/Users/Allianz/Downloads/Antigravity/Sylva/src/live_trading/sync.rs#L15-L65)), evalúa la señal forward del modelo online y ejecuta en BingX Perpetual Futures ([`exchange/bingx.rs#L15-L95`](file:///c:/Users/Allianz/Downloads/Antigravity/Sylva/src/live_trading/exchange/bingx.rs#L15-L95)) mediante autenticación criptográfica HMAC-SHA256.
* **Telemetría y Resiliencia:** Persistencia de órdenes en SQLite local ([`trades_db.rs#L10-L50`](file:///c:/Users/Allianz/Downloads/Antigravity/Sylva/src/live_trading/trades_db.rs#L10-L50)) y alertas en tiempo real a webhooks de Discord ([`discord.rs#L10-L45`](file:///c:/Users/Allianz/Downloads/Antigravity/Sylva/src/live_trading/discord.rs#L10-L45)).
