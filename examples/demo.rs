//! Run with `cargo run --release --example demo`.
use bezier_approx::Fitter;

/// Tiny deterministic RNG so the example has no dependencies.
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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut rng = Rng(42);

    // ── 1-D: approximate a known function, pick the degree by validation error ──
    let f1 = |x: f64| (-x).exp() * (4.0 * x).cos();
    let train: Vec<([f64; 1], f64)> = (0..100)
        .map(|_| {
            let x = rng.uniform();
            ([x], f1(x))
        })
        .collect();
    let valid: Vec<([f64; 1], f64)> = (0..=1000)
        .map(|i| i as f64 / 1000.0)
        .map(|x| ([x], f1(x)))
        .collect();

    println!("1-D, f(x) = exp(-x)·cos(4x) on [0, 1], 100 random samples");
    println!("degree  coeffs  train RMS   valid RMS   valid max");
    for degree in 1..=12 {
        let mut fit = Fitter::unit(1, degree).with_smoothing(0.0);
        for (x, y) in &train {
            fit.add_sample(x, *y)?;
        }
        let r = fit.solve()?;
        let t = fit.evaluate(train.iter().map(|(x, y)| (x, *y)))?;
        let e = fit.evaluate(valid.iter().map(|(x, y)| (x, *y)))?;
        println!(
            "{degree:>6}  {:>6}  {:>9.2e}   {:>9.2e}   {:>9.2e}",
            r.coefficients, t.rms, e.rms, e.max_abs
        );
    }

    let mut fit1 = Fitter::unit(1, 8).with_smoothing(0.0);
    for (x, y) in &train {
        fit1.add_sample(x, *y)?;
    }
    fit1.solve()?;

    // ── 2-D: noisy "measurements", with no data in the corner x > 0.7, y > 0.7 ──
    let f2 = |x: f64, y: f64| (2.0 * x).sin() * (1.5 * y).cos() + 0.3 * x * y;
    let noise = 0.01;
    let mut fit2 = Fitter::unit(2, 4).with_smoothing(1e-4);
    let mut n = 0;
    while n < 2000 {
        let (x, y) = (rng.uniform(), rng.uniform());
        if x > 0.7 && y > 0.7 {
            continue;
        }
        fit2.add_sample([x, y], f2(x, y) + noise * rng.normal())?;
        n += 1;
    }
    let r2 = fit2.solve()?;
    println!("\n2-D, degree 4, 2000 noisy samples (true noise σ = {noise}), corner left empty");
    println!(
        "{} coefficients, train RMS {:.4}, estimated σ {:.4}",
        r2.coefficients,
        r2.rms_residual,
        r2.sigma.unwrap()
    );
    let grid: Vec<([f64; 2], f64)> = (0..=40)
        .flat_map(|i| (0..=40).map(move |j| (i as f64 / 40.0, j as f64 / 40.0)))
        .map(|(x, y)| ([x, y], f2(x, y)))
        .collect();
    let e2 = fit2.evaluate(grid.iter().map(|(p, y)| (p, *y)))?;
    println!(
        "vs. true function on 41×41 grid: RMS {:.4}, max {:.4} at {:?}",
        e2.rms, e2.max_abs, e2.worst_input
    );
    println!("point          prediction  true       |error|    std_error");
    for p in [[0.3, 0.3], [0.5, 0.5], [0.8, 0.8], [1.0, 1.0]] {
        let pred = fit2.predict(p)?;
        let truth = f2(p[0], p[1]);
        println!(
            "{:<13}  {:>9.5}  {:>9.5}  {:>9.5}  {:>9.5}",
            format!("{p:?}"),
            pred,
            truth,
            (pred - truth).abs(),
            fit2.std_error(p)?
        );
    }

    // ── 1-D spline: a sensor curve with a sharp bend, cubic spline with 8 segments ──
    let f3 = |x: f64| (1.0 + (40.0 * (x - 0.4)).exp()).ln() / 40.0;
    let mut fit3 = Fitter::unit_spline(1, 3, 8);
    for i in 0..200 {
        let x = i as f64 / 199.0;
        fit3.add_sample([x], f3(x))?;
    }
    fit3.solve()?;
    let e3 = fit3.evaluate((0..=1000).map(|i| i as f64 / 1000.0).map(|x| ([x], f3(x))))?;
    // Same storage (11 coefficients) as a single Bézier polynomial, for comparison.
    let mut bez = Fitter::unit(1, 10).with_smoothing(0.0);
    for i in 0..200 {
        let x = i as f64 / 199.0;
        bez.add_sample([x], f3(x))?;
    }
    bez.solve()?;
    let eb = bez.evaluate((0..=1000).map(|i| i as f64 / 1000.0).map(|x| ([x], f3(x))))?;
    println!("\nCurve with a sharp bend, 11 coefficients each:");
    println!("  cubic spline, 8 segments: max error {:.1e}", e3.max_abs);
    println!("  Bézier, degree 10:        max error {:.1e}", eb.max_abs);

    println!("\n// ---- generated code ----");
    print!("{}", fit1.to_rust_code("EXP_COS")?);
    print!("{}", fit2.to_rust_code("SURFACE")?);
    print!("{}", fit3.to_rust_code("BEND")?);
    Ok(())
}
