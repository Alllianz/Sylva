use std::fmt::Debug;

pub trait Kernel: Send + Sync + Debug {
    fn compute(&self, x1: &[f64], x2: &[f64]) -> f64;
    fn variance(&self) -> f64;
    fn clone_box(&self) -> Box<dyn Kernel>;
}

impl Clone for Box<dyn Kernel> {
    fn clone(&self) -> Box<dyn Kernel> {
        self.clone_box()
    }
}

/// Núcleo Matérn 5/2 (Recomendado estándar en Optimización Bayesiana para finanzas)
/// k(x, x') = σ_f² (1 + √5 d + 5/3 d²) exp(-√5 d)
#[derive(Debug, Clone)]
pub struct Matern52Kernel {
    pub sigma_f_sq: f64,
    pub length_scales: Vec<f64>,
}

impl Matern52Kernel {
    pub fn new(dim: usize, length_scale: f64, signal_var: f64) -> Self {
        Self {
            sigma_f_sq: signal_var.max(1e-6),
            length_scales: vec![length_scale.max(1e-6); dim],
        }
    }

    pub fn with_ard(length_scales: Vec<f64>, signal_var: f64) -> Self {
        Self {
            sigma_f_sq: signal_var.max(1e-6),
            length_scales,
        }
    }
}

impl Kernel for Matern52Kernel {
    fn compute(&self, x1: &[f64], x2: &[f64]) -> f64 {
        let mut d_sq = 0.0;
        let dim = x1.len().min(x2.len()).min(self.length_scales.len());
        for i in 0..dim {
            let diff = (x1[i] - x2[i]) / self.length_scales[i];
            d_sq += diff * diff;
        }

        let d = d_sq.sqrt();
        let sqrt5_d = 5.0_f64.sqrt() * d;
        let term = 1.0 + sqrt5_d + (5.0 / 3.0) * d_sq;
        self.sigma_f_sq * term * (-sqrt5_d).exp()
    }

    fn variance(&self) -> f64 {
        self.sigma_f_sq
    }

    fn clone_box(&self) -> Box<dyn Kernel> {
        Box::new(self.clone())
    }
}

/// Núcleo Gaussiano RBF / Squared Exponential
/// k(x, x') = σ_f² exp(-0.5 * d²)
#[derive(Debug, Clone)]
pub struct RbfKernel {
    pub sigma_f_sq: f64,
    pub length_scales: Vec<f64>,
}

impl RbfKernel {
    pub fn new(dim: usize, length_scale: f64, signal_var: f64) -> Self {
        Self {
            sigma_f_sq: signal_var.max(1e-6),
            length_scales: vec![length_scale.max(1e-6); dim],
        }
    }
}

impl Kernel for RbfKernel {
    fn compute(&self, x1: &[f64], x2: &[f64]) -> f64 {
        let mut d_sq = 0.0;
        let dim = x1.len().min(x2.len()).min(self.length_scales.len());
        for i in 0..dim {
            let diff = (x1[i] - x2[i]) / self.length_scales[i];
            d_sq += diff * diff;
        }

        self.sigma_f_sq * (-0.5 * d_sq).exp()
    }

    fn variance(&self) -> f64 {
        self.sigma_f_sq
    }

    fn clone_box(&self) -> Box<dyn Kernel> {
        Box::new(self.clone())
    }
}
