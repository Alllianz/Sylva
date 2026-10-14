use crate::trees::sylva_tuner::types::{SylvaTreeCandidateConfig, SylvaTreeModelType};

/// Muestreador y Mutador Evolutivo por Bloques Coordinados (Inspirado en el motor de Sylva EVO)
pub struct SylvaTreeSampler {
    pub seed: u64,
    pub model_type: SylvaTreeModelType,
    pub candidate_horizons: Vec<usize>,
}

pub type AstroTreeSampler = SylvaTreeSampler;

impl SylvaTreeSampler {
    pub fn new(seed: u64, model_type: SylvaTreeModelType, candidate_horizons: Vec<usize>) -> Self {
        let candidate_horizons = if candidate_horizons.is_empty() {
            vec![1, 2, 4]
        } else {
            candidate_horizons
        };
        Self { seed, model_type, candidate_horizons }
    }

    #[inline(always)]
    pub fn next_u32(&mut self) -> u32 {
        self.seed = self.seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.seed >> 32) as u32
    }

    #[inline(always)]
    pub fn next_f32(&mut self) -> f32 {
        (self.next_u32() as f64 / u32::MAX as f64) as f32
    }

    #[inline(always)]
    pub fn next_gaussian(&mut self) -> f32 {
        let u1 = self.next_f32().max(1e-7);
        let u2 = self.next_f32();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
    }

    /// Genera una muestra exploratoria inicial respetando la relación estructural entre bloques
    pub fn sample_exploratory(&mut self) -> SylvaTreeCandidateConfig {
        // Bloque 1: Arquitectura y Capacidad del Árbol (Profundidad, Árboles, Hojas Mínimas)
        // Evita árboles profundos (depth 5-6) que memorizan ruido histórico
        let is_macro = self.candidate_horizons.iter().any(|&h| h >= 2);
        let max_depth = if is_macro {
            (1 + (self.next_u32() % 3)) as usize // Profundidad 1..3 (Decision Stumps y árboles compactos)
        } else {
            (2 + (self.next_u32() % 3)) as usize // Profundidad 2..4
        };

        let n_trees = match max_depth {
            1 | 2 => (15 + (self.next_u32() % 25)) as usize, // 15..39
            3 => (15 + (self.next_u32() % 20)) as usize,     // 15..34
            _ => (10 + (self.next_u32() % 15)) as usize,     // 10..24
        };
        let min_samples_leaf = if is_macro {
            (35 + (self.next_u32() % 40)) as usize // 35..74 velas de evidencia mínima
        } else {
            (25 + (self.next_u32() % 30)) as usize // 25..54
        };

        // Bloque 2: Dinámica Temporal y Tasa de Aprendizaje (Log-space)
        let log_lr = match max_depth {
            1 | 2 => -4.2 + (self.next_f32() * 1.5), // e^-4.2 .. e^-2.7 (0.015 .. 0.067)
            _ => -4.8 + (self.next_f32() * 1.4),     // e^-4.8 .. e^-3.4 (0.008 .. 0.033)
        };
        let learning_rate = log_lr.exp().clamp(0.008, 0.07);

        // Factor de olvido exponencial robusto para preservar memoria de ciclos
        let decay_factor = if is_macro {
            0.9980 + (self.next_f32() * 0.0019) // 0.9980..0.9999
        } else {
            0.9960 + (self.next_f32() * 0.0035) // 0.9960..0.9995
        };

        // Bloque 3: Regularización L1 / L2 y Período de Gracia
        let log_l2 = -1.5 + (self.next_f32() * 3.5); // 0.22 .. 7.38
        let l2_reg = log_l2.exp().clamp(0.1, 10.0);
        let l1_reg = if self.next_f32() < 0.4 { 0.0 } else { 0.001 * (1.0 + self.next_f32() * 9.0) };

        let grace_period = (10 + (self.next_u32() % 25)) as usize; // 10..34
        let split_confidence = 0.01 + (self.next_f32() * 0.09); // 0.01..0.10
        let colsample_bytree = if max_depth >= 4 { 0.70 + self.next_f32() * 0.25 } else { 0.80 + self.next_f32() * 0.20 };

        // Bloque 4: Inferencia Bayesiana & Incertidumbre
        let prior_precision = 0.2 + (self.next_f32() * 3.8); // 0.2..4.0
        let uncertainty_penalty_kappa = self.next_f32() * 1.8; // 0.0..1.8

        let h_idx = (self.next_u32() as usize) % self.candidate_horizons.len();
        let target_horizon = self.candidate_horizons[h_idx];

        SylvaTreeCandidateConfig {
            model_type: self.model_type,
            max_depth,
            n_trees,
            min_samples_leaf,
            learning_rate,
            decay_factor,
            l2_reg,
            l1_reg,
            grace_period,
            split_confidence,
            colsample_bytree,
            prior_precision,
            uncertainty_penalty_kappa,
            target_horizon,
        }
    }

    /// Mutación Adaptativa por Bloques Coordinados de un Candidato Padre Exitoso
    pub fn mutate_parent(&mut self, parent: &SylvaTreeCandidateConfig) -> SylvaTreeCandidateConfig {
        let mut child = parent.clone();
        let block_to_mutate = (self.next_u32() % 5) as usize;

        match block_to_mutate {
            0 => {
                // Mutación Bloque 1: Arquitectura (Depth, Trees, Min Leaf)
                let is_macro = self.candidate_horizons.iter().any(|&h| h >= 2);
                let delta_d = (self.next_gaussian() * 0.8).round() as i32;
                child.max_depth = if is_macro {
                    ((child.max_depth as i32) + delta_d).clamp(1, 3) as usize
                } else {
                    ((child.max_depth as i32) + delta_d).clamp(2, 4) as usize
                };

                let delta_t = (self.next_gaussian() * 4.0).round() as i32;
                child.n_trees = ((child.n_trees as i32) + delta_t).clamp(10, 45) as usize;

                let delta_leaf = (self.next_gaussian() * 4.0).round() as i32;
                child.min_samples_leaf = if is_macro {
                    ((child.min_samples_leaf as i32) + delta_leaf).clamp(30, 80) as usize
                } else {
                    ((child.min_samples_leaf as i32) + delta_leaf).clamp(20, 60) as usize
                };
            }
            1 => {
                // Mutación Bloque 2: Dinámica Temporal (Learning Rate & Olvido Exponencial en espacio log)
                let log_lr = child.learning_rate.ln() + self.next_gaussian() * 0.25;
                child.learning_rate = log_lr.exp().clamp(0.005, 0.08);

                // Mutación de decay factor
                let delta_decay = self.next_gaussian() * 0.0008;
                child.decay_factor = (child.decay_factor + delta_decay).clamp(0.9960, 0.9999);
            }
            2 => {
                // Mutación Bloque 3: Regularización (L1, L2, Colsample, Grace)
                let log_l2 = child.l2_reg.ln() + self.next_gaussian() * 0.35;
                child.l2_reg = log_l2.exp().clamp(0.05, 15.0);

                if self.next_f32() < 0.3 {
                    child.l1_reg = (child.l1_reg + self.next_gaussian() * 0.005).max(0.0);
                }

                let delta_grace = (self.next_gaussian() * 3.0).round() as i32;
                child.grace_period = ((child.grace_period as i32) + delta_grace).clamp(8, 45) as usize;
            }
            3 => {
                // Mutación Bloque 4: Parámetros Bayesianos & Incertidumbre
                let delta_prec = self.next_gaussian() * 0.4;
                child.prior_precision = (child.prior_precision + delta_prec).clamp(0.1, 5.0);

                let delta_kappa = self.next_gaussian() * 0.25;
                child.uncertainty_penalty_kappa = (child.uncertainty_penalty_kappa + delta_kappa).clamp(0.0, 2.5);
            }
            _ => {
                // Mutación Bloque 5: Horizonte Causal Multi-Vela
                let h_idx = (self.next_u32() as usize) % self.candidate_horizons.len();
                child.target_horizon = self.candidate_horizons[h_idx];
            }
        }

        child
    }
}
