//! Automatic model selection on five kinds of data.
//! Run with `cargo run --release --example autofit`.
use bezier_approx::{AutoFit, Samples};

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

fn section(title: &str) {
    println!("\n{}\n{title}\n{}", "=".repeat(78), "=".repeat(78));
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut rng = Rng(7);

    section("1. Smooth function, noisy measurements (σ = 0.002)");
    let f = |x: f64| (-x).exp() * (4.0 * x).cos();
    let mut data = Samples::new(1);
    for _ in 0..300 {
        let x = rng.uniform();
        data.push([x], f(x) + 0.002 * rng.normal())?;
    }
    println!("{}", AutoFit::unit(1).run(&data)?);

    section("2. Sensor with a sharp bend (soft kink at x = 0.4), noise σ = 0.001");
    let g = |x: f64| (1.0 + (40.0 * (x - 0.4)).exp()).ln() / 40.0;
    let mut data = Samples::new(1);
    for _ in 0..300 {
        let x = rng.uniform();
        data.push([x], g(x) + 0.001 * rng.normal())?;
    }
    println!("{}", AutoFit::unit(1).run(&data)?);

    section("3. Very little data: five points, all between 0.1 and 0.3");
    let mut data = Samples::new(1);
    for (x, y) in [
        (0.1, 0.2),
        (0.15, 0.25),
        (0.3, 0.44),
        (0.2, 0.31),
        (0.25, 0.37),
    ] {
        data.push([x], y)?;
    }
    println!("{}", AutoFit::unit(1).run(&data)?);

    section("4. 2-D measurements: 4 % glitches and no data in the corner x, y > 0.75");
    let h = |x: f64, y: f64| (2.0 * x).sin() * (1.5 * y).cos() + 0.3 * x * y;
    let mut data = Samples::new(2);
    while data.len() < 1500 {
        let (x, y) = (rng.uniform(), rng.uniform());
        if x > 0.75 && y > 0.75 {
            continue;
        }
        let mut z = h(x, y) + 0.01 * rng.normal();
        if rng.uniform() < 0.04 {
            z += 0.5 + rng.uniform(); // glitch
        }
        data.push([x, y], z)?;
    }
    println!("{}", AutoFit::unit(2).run(&data)?);

    section("5. Exact formula on [0, 2]: smallest model with RMS error ≤ 1e-4");
    let k = |x: f64| x.sqrt() * (1.0 + x).ln();
    let mut data = Samples::new(1);
    for i in 0..400 {
        let x = 2.0 * i as f64 / 399.0;
        data.push([x], k(x))?;
    }
    let report = AutoFit::new(&[0.0], &[2.0])
        .smoothing(0.0)
        .target_rms(1e-4)
        .run(&data)?;
    println!("{report}");
    println!("Exported:\n{}", report.fitter.to_rust_code("SQRT_LOG")?);
    println!("As plain C:\n{}", report.fitter.to_c_fn("sqrt_log")?);
    Ok(())
}
