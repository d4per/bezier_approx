//! Least squares vs. robust fitting on data with injected outliers.
//! Run with `cargo run --release --example outliers`.
use bezier_approx::{Fitter, RobustLoss};

struct Rng(u64);
impl Rng {
    fn uniform(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
    fn normal(&mut self) -> f64 {
        let (u, v) = (self.uniform().max(1e-300), self.uniform());
        (-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos()
    }
}

/// Fits the same data three ways and prints error against the true function
/// plus how well the injected outliers were identified.
fn compare(
    fit: &Fitter,
    truth: &[(Vec<f64>, f64)],
    injected: &[usize],
) -> Result<(), Box<dyn std::error::Error>> {
    println!("method        RMS err   max err   σ est.   flagged  (true / false)");
    for (name, loss) in [
        ("least squares", None),
        ("Huber", Some(RobustLoss::HUBER)),
        ("Tukey", Some(RobustLoss::TUKEY)),
    ] {
        let mut f = fit.clone();
        match loss {
            None => {
                f.solve()?;
            }
            Some(loss) => {
                f.solve_robust(loss)?;
            }
        }
        let e = f.evaluate(truth.iter().map(|(x, y)| (x, *y)))?;
        let sigma = f.report().unwrap().sigma.unwrap();
        let flagged = f.outliers(3.0)?;
        let hits = flagged.iter().filter(|i| injected.contains(i)).count();
        println!(
            "{name:<13} {:>8.5}  {:>8.5}  {:>7.4}  {:>7}  ({hits} / {})",
            e.rms,
            e.max_abs,
            sigma,
            flagged.len(),
            flagged.len() - hits
        );
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut rng = Rng(2024);

    // ── 1-D: noise σ = 0.005, 5 % sensor glitches ──
    let f1 = |x: f64| (-x).exp() * (4.0 * x).cos();
    let mut fit = Fitter::unit(1, 8);
    let mut injected = Vec::new();
    for i in 0..400 {
        let x = rng.uniform();
        let mut y = f1(x) + 0.005 * rng.normal();
        if rng.uniform() < 0.05 {
            y = 4.0 * rng.uniform() - 2.0; // glitch: random value in [-2, 2]
            injected.push(i);
        }
        fit.add_sample([x], y)?;
    }
    let truth: Vec<(Vec<f64>, f64)> = (0..=1000)
        .map(|i| i as f64 / 1000.0)
        .map(|x| (vec![x], f1(x)))
        .collect();
    println!(
        "1-D, degree 8, 400 samples, noise σ = 0.005, {} glitches",
        injected.len()
    );
    compare(&fit, &truth, &injected)?;

    // ── 2-D: noise σ = 0.01, 15 % outliers, all biased upwards ──
    let f2 = |x: f64, y: f64| (2.0 * x).sin() * (1.5 * y).cos() + 0.3 * x * y;
    let mut fit = Fitter::unit(2, 4).with_smoothing(1e-5);
    let mut injected = Vec::new();
    for i in 0..3000 {
        let (x, y) = (rng.uniform(), rng.uniform());
        let mut z = f2(x, y) + 0.01 * rng.normal();
        if rng.uniform() < 0.15 {
            z += 0.2 + 0.8 * rng.uniform(); // one-sided: these shift the mean
            injected.push(i);
        }
        fit.add_sample([x, y], z)?;
    }
    let truth: Vec<(Vec<f64>, f64)> = (0..=40)
        .flat_map(|i| (0..=40).map(move |j| (i as f64 / 40.0, j as f64 / 40.0)))
        .map(|(x, y)| (vec![x, y], f2(x, y)))
        .collect();
    println!(
        "\n2-D, degree 4, 3000 samples, noise σ = 0.01, {} one-sided outliers",
        injected.len()
    );
    compare(&fit, &truth, &injected)?;

    // ── The limitation: a lone outlier where there is no other data ──
    let mut fit = Fitter::unit(1, 6);
    for i in 0..200 {
        let x = 0.8 * i as f64 / 199.0; // dense data only on [0, 0.8]
        fit.add_sample([x], f1(x) + 0.005 * rng.normal())?;
    }
    fit.add_sample([0.97], f1(0.97) + 0.5)?; // single bad point in the empty region
    fit.solve_robust(RobustLoss::TUKEY)?;
    println!("\nLone outlier at x = 0.97 (data otherwise only on [0, 0.8]), Tukey fit:");
    println!("   x     |error|   std_error   support");
    for x in [0.4, 0.8, 0.9, 0.97] {
        println!(
            "  {x:.2}   {:>7.4}   {:>9.4}   {:>7.1}",
            (fit.predict([x])? - f1(x)).abs(),
            fit.std_error([x])?,
            fit.support([x])?
        );
    }
    println!(
        "robust weight of the bad point: {:.3}",
        fit.robust_weights().unwrap().last().unwrap()
    );
    Ok(())
}
