# 🌲 Especificación Técnica: Mejoras Arquitectónicas y Cuantitativas para Online GBDT en Sylva

**Destinatario:** Allianz | Laboratorio Cuantitativo Sylva  
**Fecha de Elaboración:** Octubre 2026  
**Dominio:** Finanzas Cuantitativas, Streaming Causal y Machine Learning No Estacionario  
**Lenguaje y Entorno:** 100% Rust Nativo | Cero Look-Ahead Bias | Complejidad Temporal $O(1)$ por Vela  

---

## 1. Diagnóstico y Contexto del Motor Actual

El motor actual de **Online Gradient Boosted Decision Trees** ([`src/trees/online/`](../src/trees/online/)) implementa un ensamble aditivo de árboles streaming basado en las cotas de Hoeffding (*Very Fast Decision Trees / VFDT*) y un factor de decaimiento exponencial ($\delta \in (0, 1]$).

### Fortalezas del Motor Actual
1. **Streaming Causal Puro:** Procesa vela a vela en tiempo real sin requerir reentrenamientos en bloque (*zero look-ahead bias*).
2. **Eficiencia en Memoria y Cómputo:** Complejidad espacial y temporal acotada en $O(N_{trees} \cdot \text{profundidad})$, con latencia inferior a microsegundos por vela.
3. **Optimización Taylor de Segundo Orden:** Integra gradientes y hessianos con regularización $L_1$ y $L_2$ estilo XGBoost.

### Limitaciones Identificadas en Series Financieras
1. **Vulnerabilidad a Colas Pesadas (*Fat Tails*):** La función de pérdida actual es el Error Cuadrático Medio puro (MSE), donde $g_i = \hat{y} - y$ y $h_i = 1.0$. Un mechazo de liquidación de $5\sigma$ distorsiona agresivamente las hojas.
2. **Fosilización Estructural de Nodos:** Cuando una hoja se divide mediante la cota de Hoeffding en un nodo interno (`Internal`), esa división queda fija para siempre, acumulando reglas obsoletas de regímenes macro superados.
3. **Deriva de Cuantiles en Hojas:** Los 8 bins de división se calculan en base a $\text{min\_v}$ y $\text{max\_v}$ acumulados. Si el rango se expande, los cortes cambian numéricamente pero arrastran gradientes de cortes anteriores.
4. **Indiferencia al Régimen de Volatilidad:** Todas las barras aportan el mismo peso al hessiano ($h_i = 1.0$), provocando que las velas de alta volatilidad dominen el 80% de la capacidad de aprendizaje.
5. **Inexistencia de Reciclaje de Árboles:** El ensamble mantiene los mismos $N$ árboles fijos a lo largo de cientos de miles de velas.
6. **Inestabilidad en Hojas Jóvenes:** Una hoja recién creada aplica la misma tasa de aprendizaje $\eta$ completa que una hoja veterana con miles de observaciones.

---

## 2. Detalle Exhaustivo de las 6 Mejoras

---

### Mejora 1: Función de Pérdida Robusta Pseudo-Huber / Huber Adaptativa

#### 1.1 Fundamento Matemático
Las distribuciones de retornos financieros en activos criptográficos presentan curtosis extrema ($\text{Kurtosis} \gg 3$). Bajo MSE ($L = \frac{1}{2}(\hat{y} - y)^2$), el gradiente escala linealmente con el error, lo que provoca que un único evento de liquidación mueva los pesos de las hojas más que 200 velas de tendencia normal.

La **Pérdida Huber** combina la precisión cuadrática en errores pequeños con la robustez lineal en errores grandes:
$$L_\delta(e) = \begin{cases} 
\frac{1}{2} e^2 & \text{si } |e| \le \delta_{huber} \\ 
\delta_{huber} |e| - \frac{1}{2} \delta_{huber}^2 & \text{si } |e| > \delta_{huber} 
\end{cases}$$

Donde $e = \hat{y} - y$ es el residuo.

#### 1.2 Gradiente y Hessiano de Segundo Orden
- **Primer Orden (Gradiente $g$):**
  $$g = \frac{\partial L}{\partial \hat{y}} = \begin{cases} 
  e & \text{si } |e| \le \delta_{huber} \\ 
  \delta_{huber} \cdot \text{sign}(e) & \text{si } |e| > \delta_{huber} 
  \end{cases}$$
- **Segundo Orden (Hessiano $h$):**
  $$h = \frac{\partial^2 L}{\partial \hat{y}^2} = \begin{cases} 
  1.0 & \text{si } |e| \le \delta_{huber} \\ 
  \epsilon_{floor} & \text{si } |e| > \delta_{huber} \quad (\text{ej. } \epsilon_{floor} = 0.05) 
  \end{cases}$$

#### 1.3 Calibración Adaptativa de $\delta_{huber}$
En lugar de un valor estático, $\delta_{huber}$ se escala de forma causal según la volatilidad del target:
$$\delta_{huber, t} = 1.5 \cdot \sigma_{target, t}$$
Donde $\sigma_{target, t}$ es la desviación estándar rodante ponderada exponencialmente ($EMA$) del target en In-Sample.

#### 1.4 Diseño en Rust
```rust
#[inline(always)]
pub fn calculate_huber_gradient_hessian(residual: f32, delta_huber: f32) -> (f32, f32) {
    let abs_err = residual.abs();
    if abs_err <= delta_huber {
        (residual, 1.0)
    } else {
        let g = delta_huber * residual.signum();
        let h = 0.05; // Regularización de segundo orden en cola
        (g, h)
    }
}
```

---

### Mejora 2: Poda Dinámica y Fusión de Nodos (*Tree Rejuvenation & Pruning*)

#### 2.1 El Problema de la Fosilización Estructural
En mercados de criptomonedas, un cambio de régimen (ej. de mercado alcista parabólico a mercado lateral en rango de baja volatilidad) invalida divisiones previas. En el motor actual, un nodo `Internal` jamás vuelve a ser `Leaf`.

#### 2.2 Algoritmo de Fusión de Nodos (*Leaf Merging*)
Cada nodo interno almacena estadísticas amortiguadas de la ganancia efectiva que sus dos sub-árboles están generando en el flujo actual.

1. **Monitoreo de Ganancia Dinámica del Nodo:**
   $$Gain_{actual} = \frac{1}{2} \left[ \frac{G_L^2}{H_L + \lambda} + \frac{G_R^2}{H_R + \lambda} - \frac{(G_L + G_R)^2}{H_L + H_R + \lambda} \right] - \gamma$$
2. **Criterio de Colapso (Poda Inversa):**
   Si $Gain_{actual} < -\gamma_{prune}$ durante más de $N_{grace\_prune}$ muestras (ej. 100 velas) o si el flujo relativo de una de las ramas cae por debajo del 5% del flujo total del nodo:
   $$\frac{\min(H_L, H_R)}{H_L + H_R} < 0.05$$
   El nodo `Internal` colapsa y se convierte en un nodo `Leaf` unificado:
   $$G_{new} = G_L + G_R, \quad H_{new} = H_L + H_R$$
   $$W_{new} = -\frac{G_{new}}{H_{new} + \lambda}$$

#### 2.3 Estructura Rust Propuesta
```rust
impl OnlineTreeNode {
    pub fn try_prune(&mut self, cfg: &OnlineGbdtConfig) -> bool {
        match self {
            OnlineTreeNode::Internal { left, right, .. } => {
                let left_h = left.total_hessian();
                let right_h = right.total_hessian();
                let total_h = left_h + right_h;
                
                // Si una rama quedó completamente desierta o la ganancia es destructiva
                if total_h > 20.0 && (left_h / total_h < 0.05 || right_h / total_h < 0.05) {
                    let merged_stats = left.collect_stats().merge(&right.collect_stats());
                    let merged_weight = merged_stats.compute_weight(cfg.l1_reg, cfg.l2_reg);
                    *self = OnlineTreeNode::Leaf {
                        stats: merged_stats,
                        weight: merged_weight,
                    };
                    return true;
                }
                false
            }
            OnlineTreeNode::Leaf { .. } => false,
        }
    }
}
```

---

### Mejora 3: Cuantiles Adaptativos Estables en Hojas (*Drift-Resistant Bins*)

#### 3.1 El Problema de Deriva en Bins Lineales
Actualmente, los 8 umbrales de división en cada hoja se calculan como:
$$\text{threshold}_b = \text{min\_v} + (\text{max\_v} - \text{min\_v}) \cdot \frac{b + 1}{9}$$
Si $\text{max\_v}$ se duplica por un movimiento extremo, el bin $b=3$ pasa de representar el valor $0.002$ al valor $0.004$, pero contiene gradientes acumulados bajo la premisa de $0.002$.

#### 3.2 Solución: Centros de Masa Cuantiles Normalizados
1. **Normalización por Z-Score Rodante:** Cada variable $x_i$ ingresa pre-normalizada mediante media y varianza rodante causal ($\mu_{t-1}, \sigma_{t-1}$), situándose aproximadamente en $[-3.0, +3.0]$.
2. **Partición en Cuantiles Estables Fijados:** En lugar de buscar cortes en el espacio numérico crudo, los 8 cuantiles se fijan en percentiles estándar de la distribución empírica estandarizada:
   $$\text{Cortes: } \Phi^{-1}([0.10, 0.20, 0.35, 0.50, 0.65, 0.80, 0.90])$$
3. **Beneficio Cuantitativo:** Los acumuladores de gradientes `split_g_left` y `split_h_left` siempre representan la misma región relativa de la distribución, garantizando que el decaimiento exponencial refleje probabilidades estadísticas coherentes.

---

### Mejora 4: Ponderación de Muestras Inversa a la Volatilidad ($w_t = 1/\sigma_t$)

#### 4.1 Justificación Institucional (Espacio Ratio Sharpe)
En backtest y trading real, el objetivo no es maximizar el error cuadrático en dólares, sino maximizar el **retorno ajustado por volatilidad (Sharpe / Sortino)**.

Si el mercado atraviesa una fase de volatilidad del 4% diario, los errores residuales serán 4 veces mayores que en una fase de compresión del 1%. El GBDT estándar centrará toda su capacidad en el período volátil y generará ruido en fases de compresión.

#### 4.2 Formulación Matemática
Definimos el peso causal de la muestra $t$ como:
$$w_t = \text{clamp}\left( \frac{\overline{\text{ATR}}_{200}}{\text{ATR}_t}, \ 0.25, \ 4.0 \right)$$
Donde:
- $\text{ATR}_t$ es el Average True Range de la vela actual (calculado estrictamente con velas pasadas $t-1$).
- $\overline{\text{ATR}}_{200}$ es la media móvil de 200 períodos del ATR.

#### 4.3 Inyección en el Gradient Boosting
En el paso de actualización de la muestra $t$:
$$g_t^{weighted} = w_t \cdot g_t$$
$$h_t^{weighted} = w_t \cdot h_t$$
Esto normaliza la superficie de optimización, logrando que cada vela aporte exactamente una unidad estándar de información matemática independiente del régimen de volatilidad.

---

### Mejora 5: Reciclaje y Reemplazo Continuo de Árboles (*Sliding Ensemble / Tree Dropout*)

#### 5.1 Envejecimiento del Ensamble Fijo
En un ensamble de 30 árboles fijos, el Árbol 1 se formó con las primeras 1,000 velas y puede arrastrar sesgos de un mercado alcista de años anteriores. Aunque el factor de olvido $\delta$ atenúa los pesos, la topología del árbol (sus divisiones) sigue condicionada por el pasado lejano.

#### 5.2 Algoritmo de Reemplazo Deslizante (*Tree Recycling*)
1. **Período de Evaluación de Rejuvenecimiento:** Cada $K = 2,500$ velas.
2. **Métrica de Contribución Marginal:** Para cada árbol $m \in \{1 \dots M\}$, calcular su contribución a la reducción de error en las últimas 500 barras:
   $$\text{Utility}_m = \sum_{t=T-500}^T \left( |y_t - \hat{y}^{(m-1)}_t| - |y_t - \hat{y}^{(m)}_t| \right)$$
3. **Poda y Regeneración:**
   - Se identifica el árbol con menor $\text{Utility}_m$.
   - Se reinicializa como una hoja raíz en blanco (`OnlineTreeNode::new_leaf()`).
   - Se rota al final de la cola del ensamble para ajustarse a los residuos contemporáneos más frescos.
4. **Resultado:** El ensamble opera como una cinta transportadora viva: combina árboles "ancianos" con estructura macro sólida y árboles "jóvenes" hiper-adaptados al régimen actual.

---

### Mejora 6: Shrinkage Adaptativo por Nivel de Evidencia en Hoja

#### 6.1 Problema de Varianza en Hojas Nuevas
Cuando una hoja se acaba de dividir, tiene una masa de datos pequeña ($N \approx \text{min\_samples\_split}$). Asignarle la tasa de aprendizaje $\eta$ completa (ej. $0.03$) inyecta alta varianza a la predicción global del ensamble.

#### 6.2 Amortiguación Bayesiana de la Tasa de Aprendizaje
Definimos el número efectivo de muestras de una hoja como:
$$N_{eff} = \frac{H_j}{1 - \delta}$$
La tasa de aprendizaje efectiva de la hoja $j$ se escala suavemente según su madurez estadística:
$$\eta_j = \eta_{base} \cdot \sqrt{\frac{N_{eff}}{N_{eff} + N_{prior}}}$$
Donde $N_{prior}$ es una constante de amortiguación (ej. $50$ muestras).

- **Hoja recién nacida ($N_{eff} \approx 5$):** $\eta_j \approx 0.3 \cdot \eta_{base}$ (suave, no arriesga capital).
- **Hoja madura ($N_{eff} \ge 200$):** $\eta_j \approx 1.0 \cdot \eta_{base}$ (aplica el 100% de la convicción).

---

## 3. Matriz Comparativa y Hoja de Ruta

| ID | Mejora Propuesta | Impacto en Sharpe / PnL | Impacto en Overfitting | Complejidad Rust | Prioridad Recomendada |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **M1** | **Huber / Pseudo-Huber Loss** | ⭐⭐⭐⭐⭐ (Protección contra mechas) | ⭐⭐⭐⭐⭐ (Elimina outliers destructivos) | Baja (1 archivo) | **Fase 1 (Inmediata)** |
| **M4** | **Ponderación por $1/\text{ATR}$** | ⭐⭐⭐⭐⭐ (Señales consistentes) | ⭐⭐⭐⭐ (Equilibra volatilidad) | Baja (2 archivos) | **Fase 1 (Inmediata)** |
| **M6** | **Shrinkage por Evidencia** | ⭐⭐⭐⭐ (Reduce varianza) | ⭐⭐⭐⭐ (Evita saltos bruscos) | Baja (1 archivo) | **Fase 2** |
| **M3** | **Cuantiles Adaptativos Estables** | ⭐⭐⭐⭐ (Mayor precisión de corte) | ⭐⭐⭐ (Menor deriva) | Media (2 archivos) | **Fase 2** |
| **M2** | **Poda Dinámica de Nodos** | ⭐⭐⭐⭐⭐ (Adaptabilidad de régimen) | ⭐⭐⭐⭐⭐ (Árboles compactos) | Media (2 archivos) | **Fase 3** |
| **M5** | **Reciclaje Deslizante de Árboles** | ⭐⭐⭐⭐⭐ (Cero fosilización) | ⭐⭐⭐⭐ (Rejuvenecimiento continuo) | Media (2 archivos) | **Fase 3** |

---

## 4. Compromisos de Calidad y Restricciones
- **100% Rust:** Todo el código futuro de estas mejoras se implementará de forma modular, compilable sin advertencias y sin dependencias externas innecesarias.
- **Causalidad Estricta:** Ninguna de estas mejoras utiliza información de velas futuras ($t+1$). Todas las métricas de ATR, Huber y cuantiles operan estrictamente sobre el flujo causal $t \le T$.
- **Cero Simulación:** Cada algoritmo contiene lógica de producción exacta y medible en los dashboards de Sylva.
