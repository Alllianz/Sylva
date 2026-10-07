use crate::features::FeatureMask;
use crate::features::TOTAL_FEATURES;
use crate::trees::dataset::TabularDataset;

/// Configuración para el ajuste de Elastic Net (L1 Lasso + L2 Ridge)
#[derive(Clone, Debug)]
pub struct ElasticNetConfig {
    pub l1_ratio: f32, // alpha in [0.0, 1.0]: 1.0 es Lasso puro, 0.0 es Ridge puro
    pub lambda: f32,   // Fuerza de penalización global
    pub max_iter: usize,
    pub tol: f32,
}

impl Default for ElasticNetConfig {
    fn default() -> Self {
        Self {
            l1_ratio: 0.5,
            lambda: 0.005,
            max_iter: 1000,
            tol: 1e-6,
        }
    }
}

/// Modelo lineal regularizado Elastic Net
#[derive(Clone, Debug)]
pub struct ElasticNetModel {
    pub weights: [f32; TOTAL_FEATURES],
    pub bias: f32,
    pub active_mask: [bool; TOTAL_FEATURES],
}

impl ElasticNetModel {
    #[inline(always)]
    pub fn predict(&self, features: &[f32; TOTAL_FEATURES]) -> f32 {
        let mut score = self.bias;
        for i in 0..TOTAL_FEATURES {
            if self.active_mask[i] {
                score += self.weights[i] * features[i];
            }
        }
        score
    }

    pub fn active_features_count(&self) -> usize {
        self.active_mask.iter().filter(|&&act| act).count()
    }
}

/// Entrenador mediante Descenso por Coordenadas Cíclico (Cyclic Coordinate Descent)
pub struct ElasticNetTrainer {
    pub config: ElasticNetConfig,
}

impl ElasticNetTrainer {
    pub fn new(config: ElasticNetConfig) -> Self {
        Self { config }
    }

    /// Operador de umbral suave (Soft-thresholding operator)
    #[inline(always)]
    fn soft_threshold(z: f64, gamma: f64) -> f64 {
        if z > gamma {
            z - gamma
        } else if z < -gamma {
            z + gamma
        } else {
            0.0
        }
    }

    /// Ajusta los pesos de Elastic Net usando Coordinate Descent
    pub fn fit(
        &self,
        dataset: &TabularDataset,
        mask: Option<&FeatureMask>,
    ) -> Result<ElasticNetModel, &'static str> {
        let n_samples = dataset.len();
        if n_samples < TOTAL_FEATURES + 10 {
            return Err("Muestras insuficientes para Elastic Net");
        }

        let l1_pen = (self.config.lambda * self.config.l1_ratio) as f64;
        let l2_pen = (self.config.lambda * (1.0 - self.config.l1_ratio)) as f64;

        // Centrar targets
        let mean_y = dataset.samples.iter().map(|s| s.target as f64).sum::<f64>() / (n_samples as f64);

        // Precalcular sum(x_j^2) y normas de columnas
        let mut sum_x_sq = [0.0f64; TOTAL_FEATURES];
        for s in &dataset.samples {
            for j in 0..TOTAL_FEATURES {
                let xj = s.features[j] as f64;
                sum_x_sq[j] += xj * xj;
            }
        }

        let mut w = [0.0f64; TOTAL_FEATURES];
        let mut residuals: Vec<f64> = dataset.samples.iter().map(|s| (s.target as f64) - mean_y).collect();

        // Iteraciones de Coordinate Descent
        for _iter in 0..self.config.max_iter {
            let mut max_change = 0.0f64;

            for j in 0..TOTAL_FEATURES {
                if let Some(m) = mask {
                    if !m.is_active(j) {
                        w[j] = 0.0;
                        continue;
                    }
                }

                if sum_x_sq[j] <= 1e-9 {
                    w[j] = 0.0;
                    continue;
                }

                let old_wj = w[j];

                // Calcular rho_j = sum(x_ij * (r_i + old_wj * x_ij))
                let mut rho_j = 0.0f64;
                for (i, s) in dataset.samples.iter().enumerate() {
                    let xj = s.features[j] as f64;
                    rho_j += xj * (residuals[i] + old_wj * xj);
                }

                // Actualización cerrada mediante soft-thresholding
                let denom = sum_x_sq[j] + l2_pen * (n_samples as f64);
                let new_wj = Self::soft_threshold(rho_j, l1_pen * (n_samples as f64)) / denom;

                // Actualizar residuos si hubo cambio
                let delta_w = new_wj - old_wj;
                if delta_w.abs() > 1e-12 {
                    for (i, s) in dataset.samples.iter().enumerate() {
                        let xj = s.features[j] as f64;
                        residuals[i] -= delta_w * xj;
                    }
                }

                max_change = max_change.max(delta_w.abs());
                w[j] = new_wj;
            }

            if max_change < self.config.tol as f64 {
                break;
            }
        }

        let mut final_weights = [0.0f32; TOTAL_FEATURES];
        let mut active_mask = [false; TOTAL_FEATURES];

        for i in 0..TOTAL_FEATURES {
            let val = w[i] as f32;
            if val.abs() > 1e-7 {
                final_weights[i] = val;
                active_mask[i] = true;
            }
        }

        Ok(ElasticNetModel {
            weights: final_weights,
            bias: mean_y as f32,
            active_mask,
        })
    }
}
