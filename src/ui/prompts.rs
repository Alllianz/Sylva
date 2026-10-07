use std::io::{self, Write};

pub fn prompt_timeframe_for_analysis() -> String {
    println!("\n  Temporalidades disponibles: 1D, 4H, 1H, 15m, 5m, 1m");
    print!("  ⏱️  Selecciona temporalidad [Por defecto '1H']: ");
    let _ = io::stdout().flush();
    let mut input = String::new();
    let _ = io::stdin().read_line(&mut input);
    let trimmed = input.trim();
    if trimmed.is_empty() {
        "1H".to_string()
    } else {
        crate::data::db::normalize_timeframe(trimmed)
    }
}

pub fn prompt_leverage_for_analysis() -> f64 {
    print!("  ⚡ Selecciona el apalancamiento (1-100x) [Por defecto '10x']: ");
    let _ = io::stdout().flush();
    let mut input = String::new();
    let _ = io::stdin().read_line(&mut input);
    let trimmed = input.trim().trim_end_matches(['x', 'X', ' ']);
    if trimmed.is_empty() {
        10.0
    } else {
        trimmed.parse::<f64>().unwrap_or(10.0).clamp(1.0, 100.0)
    }
}

pub fn parse_usize_list(input: &str, default_val: usize) -> Vec<usize> {
    let trimmed = input.trim();
    if trimmed.is_empty() {
        return vec![default_val];
    }
    let mut values = Vec::new();
    for part in trimmed.split(|c: char| c == ',' || c == ';' || c.is_whitespace()) {
        let p = part.trim();
        if !p.is_empty() {
            if let Ok(v) = p.parse::<usize>() {
                if v > 0 && !values.contains(&v) {
                    values.push(v);
                }
            }
        }
    }
    if values.is_empty() {
        vec![default_val]
    } else {
        values.sort_unstable();
        values
    }
}

pub fn pause() {
    print!("\nPresione Enter para continuar...");
    let _ = io::stdout().flush();
    let mut _input = String::new();
    let _ = io::stdin().read_line(&mut _input);
}
