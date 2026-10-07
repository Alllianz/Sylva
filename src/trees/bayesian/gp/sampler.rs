/// Límites continuos [min, max] para cada parámetro a optimizar
#[derive(Debug, Clone, Copy)]
pub struct ContinuousBound {
    pub min: f64,
    pub max: f64,
}

impl ContinuousBound {
    pub fn new(min: f64, max: f64) -> Self {
        Self { min, max }
    }

    #[inline]
    pub fn clamp(&self, val: f64) -> f64 {
        val.max(self.min).min(self.max)
    }

    #[inline]
    pub fn span(&self) -> f64 {
        self.max - self.min
    }
}

/// Generador pseudoaleatorio determinista de 64 bits (LCG / XorShift)
pub struct FastRng {
    state: u64,
}

impl FastRng {
    pub fn new(seed: u64) -> Self {
        let initial_state = if seed == 0 { 0x853c49e6748fea9b } else { seed };
        Self { state: initial_state }
    }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.state
    }

    #[inline]
    pub fn next_f64(&mut self) -> f64 {
        // [0.0, 1.0)
        let bits = (self.next_u64() >> 11) as f64;
        bits * (1.0 / (1u64 << 53) as f64)
    }

    #[inline]
    pub fn gen_range_f64(&mut self, min: f64, max: f64) -> f64 {
        min + (max - min) * self.next_f64()
    }

    #[inline]
    pub fn gen_range_usize(&mut self, min: usize, max: usize) -> usize {
        if min >= max {
            return min;
        }
        let count = max - min;
        min + (self.next_u64() as usize % count)
    }
}

/// Muestreador de Hipercubo Latino (Latin Hypercube Sampling - LHS)
pub struct LatinHypercubeSampler;

impl LatinHypercubeSampler {
    /// Genera `n_samples` puntos estratificados en el espacio acotado por `bounds`
    pub fn sample(bounds: &[ContinuousBound], n_samples: usize, seed: u64) -> Vec<Vec<f64>> {
        let dim = bounds.len();
        let mut rng = FastRng::new(seed);
        let mut result = vec![vec![0.0f64; dim]; n_samples];

        for (d, bound) in bounds.iter().enumerate() {
            let mut intervals: Vec<usize> = (0..n_samples).collect();
            // Barajar aleatoriamente los estratos para la dimensión d (Fisher-Yates)
            for i in (1..n_samples).rev() {
                let j = rng.gen_range_usize(0, i + 1);
                intervals.swap(i, j);
            }

            let step = bound.span() / (n_samples as f64);
            for (i, &stratum) in intervals.iter().enumerate() {
                let offset = rng.next_f64();
                let val = bound.min + (stratum as f64 + offset) * step;
                result[i][d] = bound.clamp(val);
            }
        }

        result
    }

    /// Genera candidatos para maximizar la función de adquisición:
    /// Combina exploración global (LHS / uniforme) con perturbaciones locales alrededor de los mejores puntos observados.
    pub fn generate_acquisition_candidates(
        bounds: &[ContinuousBound],
        observed_x: &[Vec<f64>],
        n_candidates: usize,
        mutation_scale: f64,
        seed: u64,
    ) -> Vec<Vec<f64>> {
        let dim = bounds.len();
        let mut rng = FastRng::new(seed);
        let mut candidates = Vec::with_capacity(n_candidates);

        // 1. 40% candidatos de exploración global pura (LHS)
        let n_global = (n_candidates as f64 * 0.40) as usize;
        let global_samples = Self::sample(bounds, n_global.max(10), seed.wrapping_add(101));
        candidates.extend(global_samples);

        // 2. 60% candidatos de refinamiento local (mutaciones gaussianas/uniformes alrededor de puntos observados)
        if !observed_x.is_empty() {
            let n_local = n_candidates.saturating_sub(candidates.len());
            for _ in 0..n_local {
                let base_idx = rng.gen_range_usize(0, observed_x.len());
                let base = &observed_x[base_idx];
                let mut cand = vec![0.0f64; dim];

                for d in 0..dim {
                    let span = bounds[d].span();
                    let std_dev = span * mutation_scale;
                    let noise = rng.gen_range_f64(-std_dev, std_dev);
                    cand[d] = bounds[d].clamp(base[d] + noise);
                }
                candidates.push(cand);
            }
        }

        candidates
    }
}
