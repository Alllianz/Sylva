# Especificación Técnica Cuantitativa: Regularización y Poda Macro para 1D y 4H en Sylva

**Autor:** Allianz & Antigravity  
**Proyecto:** Sylva Quantitative Engine  
**Objetivo:** Arquitectura anti-overfitting, regularización de hiperparámetros (B) y poda de micro-ruido (C) para temporalidades macro/swing (1D y 4H) con 0% Look-Ahead Bias.

---

## 1. Contexto y Diagnóstico Teórico

En trading cuantitativo y machine learning aplicado a series temporales financieras (López de Prado, 2018), los timeframes **1D** y **4H** difieren drásticamente del intradía de alta frecuencia (1m a 15m):

1. **Restricción de Volumen de Muestras ($N$):**
   - En **1D**: ~2.583 muestras In-Sample (~7 años).
   - En **4H**: ~15.500 muestras In-Sample (~7 años).
   - En comparación, en **1m** se cuenta con >300.000 muestras.
2. **Capacidad del Modelo vs. Grados de Libertad:**
   - Un árbol con `max_depth >= 4` en 1D o `max_depth >= 5` en 4H tiene la capacidad de memorizar eventos históricos idiosincráticos (e.g., caídas de volatilidad de fines de semana o shocks puntuales de liquidez), generando un ajuste perfecto In-Sample y colapso Out-of-Sample.
3. **Decaimiento de Memoria Temporal ($\delta$):**
   - La vida media de una observación bajo decaimiento exponencial es:
     $$\tau_{1/2} = \frac{\ln(0.5)}{\ln(\delta)}$$
   - Con $\delta = 0.9950$, $\tau_{1/2} \approx 138$ barras:
     - En 1m: 2.3 horas.
     - En 4H: 23 días (*olvida el régimen del mes anterior*).
     - En 1D: 4.5 meses (*olvida el año anterior*).
   - Para temporalidades swing, $\delta$ debe situarse en $[0.9980, 0.9999]$ para conservar la memoria de regímenes macro.

---

## 2. Sección B: Arquitectura de Regularización en el Muestreador Evolutivo (Sylva EVO)

### 2.1 Hiperparámetros Acotados por Régimen Temporal

| Hiperparámetro | Perfil 1D (Macro Swing) | Perfil 4H (Session Swing) | Intradía Estándar (1m / 5m / 15m) |
| :--- | :--- | :--- | :--- |
| **`max_depth`** | **$1 \dots 2$** (*Decision Stumps*) | **$2 \dots 3$** (Compact Trees) | $2 \dots 6$ |
| **`min_samples_leaf`** | **$40 \dots 80$ velas** (~1.5 a 3 meses) | **$30 \dots 60$ velas** (~5 a 10 días) | $15 \dots 35$ velas |
| **`colsample_bytree`** | **$0.50 \dots 0.65$** | **$0.55 \dots 0.70$** | $0.70 \dots 0.95$ |
| **`learning_rate`** | **$0.010 \dots 0.035$** | **$0.015 \dots 0.040$** | $0.008 \dots 0.120$ |
| **`decay_factor` ($\delta$)** | **$0.9990 \dots 0.9999$** | **$0.9980 \dots 0.9998$** | $0.9940 \dots 0.9985$ |
| **`l2_reg`** | **$1.0 \dots 10.0$** | **$0.5 \dots 8.0$** | $0.1 \dots 5.0$ |
| **`l1_reg`** | **$0.01 \dots 0.10$** | **$0.005 \dots 0.05$** | $0.0 \dots 0.01$ |
| **`n_trees`** | **$15 \dots 35$** | **$20 \dots 45$** | $10 \dots 60$ |

### 2.2 Diseño Modular en Rust (`src/trees/sylva_tuner/regime.rs`)

```rust
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TimeframeRegime {
    HighFrequency, // 1m, 5m
    Intraday,      // 15m, 1H
    SwingMacro,    // 4H, 1D
}

pub struct RegimeHyperparamBounds {
    pub min_depth: usize,
    pub max_depth: usize,
    pub min_samples_leaf_range: (usize, usize),
    pub colsample_range: (f32, f32),
    pub lr_range: (f32, f32),
    pub decay_range: (f32, f32),
    pub default_n_trees: (usize, usize),
}

impl RegimeHyperparamBounds {
    pub fn for_timeframe(tf: &str) -> Self {
        match tf {
            "1d" | "1D" => Self {
                min_depth: 1,
                max_depth: 2,
                min_samples_leaf_range: (40, 80),
                colsample_range: (0.50, 0.65),
                lr_range: (0.010, 0.035),
                decay_range: (0.9990, 0.9999),
                default_n_trees: (15, 35),
            },
            "4h" | "4H" => Self {
                min_depth: 2,
                max_depth: 3,
                min_samples_leaf_range: (30, 60),
                colsample_range: (0.55, 0.70),
                lr_range: (0.015, 0.040),
                decay_range: (0.9980, 0.9998),
                default_n_trees: (20, 45),
            },
            _ => Self {
                min_depth: 2,
                max_depth: 6,
                min_samples_leaf_range: (15, 45),
                colsample_range: (0.70, 0.95),
                lr_range: (0.008, 0.120),
                decay_range: (0.9940, 0.9985),
                default_n_trees: (10, 60),
            },
        }
    }
}
```

---

## 3. Sección C: Poda Selectiva de Variables Macro (0% Micro-Ruido)

### 3.1 Justificación Teórica de la Poda
- **Microestructura (F1..F5):** Las relaciones entre mecha superior/inferior y cuerpo de vela en marcos de 4 horas o 1 día no representan desbalances de órdenes agresivas instantáneas, sino consolidaciones intradía condensadas. Aportan ruido espurio al árbol.
- **Osciladores de Muy Corto Plazo (F7, F8, F9, F10):** En 1D y 4H, los rangos de 10 barras simples sin normalización adaptativa crean sobreajuste a falsos rompimientos.
- **Variables Críticas de Régimen y Memoria:**
  - **F16 (Filtro Macro EMA200):** $Close > EMA200$. El indicador institucional primario para definir el sesgo de largo plazo.
  - **F15 (Distancia porcentual a EMA200):** Mide la elasticidad de reversión a la media.
  - **F11 (Kaufman Efficiency Ratio):** Ratio entre el cambio neto y el recorrido total de la serie. Permite al árbol discernir entre tendencias sostenidas y rangos erráticos.
  - **F31 (Diferenciación Fraccionaria):** Permite retener la información histórica del nivel de precios (memoria larga) mientras transforma la serie en estacionaria para evitar regresiones espurias.
  - **F6 (Momentum 10 Barras):** Inercia direccional acumulada.
  - **F18 (Volume Flow Ratio SMA5 / SMA20):** Expansión vs. contracción del flujo de volumen institucional.
  - **F20 (Volume-Price Trend VPT 10):** Divergencia acumulada entre el flujo de volumen y el precio.
  - **F14 (Distancia a EMA20):** Retracción intermedia hacia la media de corto-medio plazo.

### 3.2 Máscara Propuesta (`FeatureMask::new_macro_swing()`)
- **Variables Podadas (24):** F1, F2, F3, F4, F5, F7, F8, F9, F10, F12, F13, F17, F19, F21, F22, F23, F24, F25, F26, F27, F28, F29, F30, F32.
- **Variables Activas (8):**
  1. `F6`: 10-Bar Return Momentum
  2. `F11`: Kaufman Efficiency Ratio
  3. `F14`: Dist to EMA-20
  4. `F15`: Dist to Macro EMA-200 (HL/2)
  5. `F16`: Macro Regime Flag (Close > EMA200)
  6. `F18`: Volume Flow Ratio (SMA5 / SMA20)
  7. `F20`: Volume-Price Trend (VPT 10)
  8. `F31`: Fractional Differentiation (Memory)

---

## 4. Estado de Implementación

- [x] **A. Horizontes Causales Multi-Vela ($H$):** **IMPLEMENTADO EN RUST** en [`src/ui/session.rs`](../src/ui/session.rs).
  - 1D: $[2, 3, 5]$ velas ($48h$, $72h$, $120h$).
  - 4H: $[2, 3, 6, 12]$ velas ($8h$, $12h$, $24h$, $48h$).
  - Ruido de $H=1$ eliminado en ambos.
- [ ] **B. Arquitectura de Regularización en Sylva EVO:** Documentada en esta especificación para integración modular.
- [ ] **C. Poda Selectiva Macro:** Documentada en esta especificación para integración en `FeatureMask`.
