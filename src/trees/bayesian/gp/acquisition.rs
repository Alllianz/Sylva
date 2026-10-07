use std::f64::consts::PI;

/// Funciones de Adquisición para Optimización Bayesiana
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AcquisitionType {
    ExpectedImprovement,
    UpperConfidenceBound,
    ProbabilityOfImprovement,
}

/// Densidad de Probabilidad Normal Estándar φ(z)
#[inline]
pub fn standard_normal_pdf(z: f64) -> f64 {
    (1.0 / (2.0 * PI).sqrt()) * (-0.5 * z * z).exp()
}

/// Función de Error erf(x) - Aproximación de Abramowitz & Stegun (7.1.26)
/// Error absoluto máximo < 1.5e-7
pub fn erf(x: f64) -> f64 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let abs_x = x.abs();

    let p = 0.3275911;
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;

    let t = 1.0 / (1.0 + p * abs_x);
    let poly = ((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t;
    let y = 1.0 - poly * (-abs_x * abs_x).exp();

    sign * y
}

/// Función de Distribución Acumulada Normal Estándar Φ(z)
#[inline]
pub fn standard_normal_cdf(z: f64) -> f64 {
    0.5 * (1.0 + erf(z / 2.0_f64.sqrt()))
}

/// Evalúa la función de adquisición seleccionada
pub fn evaluate_acquisition(
    acq_type: AcquisitionType,
    mean: f64,
    variance: f64,
    current_best_y: f64,
    exploration_xi: f64,
    kappa_ucb: f64,
) -> f64 {
    let std_dev = variance.sqrt();
    if std_dev < 1e-8 {
        return 0.0;
    }

    match acq_type {
        AcquisitionType::ExpectedImprovement => {
            // EI(x) = (μ(x) - f* - ξ) Φ(Z) + σ(x) φ(Z)
            let improvement = mean - current_best_y - exploration_xi;
            let z = improvement / std_dev;
            let phi_z = standard_normal_pdf(z);
            let big_phi_z = standard_normal_cdf(z);

            let ei = improvement * big_phi_z + std_dev * phi_z;
            ei.max(0.0)
        }
        AcquisitionType::UpperConfidenceBound => {
            // UCB(x) = μ(x) + κ * σ(x)
            mean + kappa_ucb * std_dev
        }
        AcquisitionType::ProbabilityOfImprovement => {
            // PI(x) = Φ(Z)
            let improvement = mean - current_best_y - exploration_xi;
            let z = improvement / std_dev;
            standard_normal_cdf(z)
        }
    }
}
