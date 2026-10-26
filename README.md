# bezier_approx

Fit a smooth function to measurements or to an expensive formula on the host, then
embed it as a Rust `const` that evaluates without allocation. Inputs live in a box
in D dimensions; the output is one number.

Models are tensor-product Bézier polynomials, or chains of them joined smoothly
(clamped uniform B-splines). A spline with one segment *is* a Bézier polynomial, so
both share one fitting engine, one exporter and one evaluator.

```toml
[dependencies]
bezier_approx = "0.1"
```

## Let the crate choose the model

```rust
use bezier_approx::{AutoFit, Samples};

let mut data = Samples::new(2);
data.push([0.1, 0.2], 0.3)?;
data.push([0.4, 0.5], 0.7)?;
// ...

let report = AutoFit::unit(2).run(&data)?;      // or AutoFit::new(&lo, &hi)
println!("{report}");                           // table, recommendation, caveats
println!("{}", report.fitter.to_rust_code("MY_FN")?);
```

`AutoFit` tries Bézier polynomials of degree 1–10 and splines of degree 1, 2 and 3
with 2–32 segments per axis. It scores each by 5-fold cross-validation: every sample
is predicted by a model that never saw it. It recommends the smallest model whose
error is within one standard error of the best, so larger models only win when they
are clearly better. Its summary explains the choice and lists caveats such as too few
samples per coefficient, outliers found (and handled by default), regions with no
data, errors concentrated in one place, or the best model sitting at a search limit.

For exact data from a formula, error keeps falling with model size. Use
`.smoothing(0.0).target_rms(1e-4)` to get the smallest model that is accurate enough.
Other settings: `max_bezier_degree`, `spline_degrees`, `max_segments`,
`max_coefficients` (default 400), `folds`, `robust`, `seed`.

## Or pick the model yourself

```rust
use bezier_approx::Fitter;

let mut fit = Fitter::unit(2, 4);               // Bézier, degree 4 → 25 coefficients
let mut fit = Fitter::unit_spline(2, 3, 8);     // cubic spline, 8 segments per axis → 121
fit.add_sample([0.1, 0.2], 0.3)?;               // stream as many samples as you like
let report = fit.solve()?;                      // or fit.solve_robust(RobustLoss::TUKEY)
let y = fit.predict([0.3, 0.3])?;
let se = fit.std_error([0.3, 0.3])?;            // local uncertainty
let n = fit.support([0.3, 0.3])?;               // ≈ samples backing this prediction
let err = fit.evaluate(validation_iter)?;       // RMS / max error on held-out data
println!("{}", fit.to_rust_code("MY_FN")?);
```

The generated constant is used like this, with no allocation and no training code:

```rust
pub const MY_FN: bezier_approx::BezierFn<'static, 2, 4> = bezier_approx::BezierFn::new(
    [0.0, 0.0], // lo
    [1.0, 1.0], // hi
    8,          // segments
    &[ /* 121 coefficients */ ],
);
let y = MY_FN.eval([0.3, 0.3]);
```

The second type parameter is degree + 1. `BezierFn::new` checks the shape when the
constant is compiled, so a table that does not match it is a compile error.

Each evaluation combines `(degree + 1)^D` coefficients, whatever the number of
segments. Splines are cheap to evaluate, and only the stored table grows.

## Embedded / `no_std`

Training needs the standard library, but evaluating does not. With default features
off, the crate is `no_std` and contains only `BezierFn` and the basis functions, and
it needs no allocator:

```toml
[dependencies]
bezier_approx = { version = "0.1", default-features = false }
```

## Which model when

* **Bézier**: one polynomial over the whole box. Best for smooth functions; few
  coefficients. Struggles with sharp bends, steep regions and jumps.
* **Spline, degree 1**: piecewise linear, i.e. a least-squares-fitted lookup table
  with linear interpolation. Robust and simple; needs more segments for accuracy.
* **Spline, degree 3**: cubic pieces with continuous slope and curvature. Handles
  local features; usually the most accurate per coefficient on uneven functions.

## Outliers

`solve()` is ordinary (weighted) least squares, which is optimal for ordinary noise but
lets a few bad measurements pull the curve. `solve_robust` uses iteratively reweighted
least squares:

* **Huber** limits each outlier's influence; every sample still counts a little.
* **Tukey** ignores gross outliers completely; best when outliers are one-sided or large.
* `outliers(3.0)` lists suspicious samples; `robust_weights()` gives 1 = trusted, 0 = ignored.
* `add_weighted_sample(x, y, w)` handles measurements of known, differing precision.
  Only the ratios of the weights matter; errors and noise estimates are reported
  in units of y.
* `support(x)` near 1 means the fit rests on a single sample there, so an error in
  it cannot be detected by any method.

Samples are stored (`dims + 2` floats each) so the robust fit can revisit them.
`without_sample_storage()` turns that off for pure streaming.

## Examples

`cargo test`, `cargo run --release --example autofit`, `… --example demo`,
`… --example outliers`, and `cargo run --example use_generated`, which contains
constants pasted from the demo output.

## License

MIT, see [LICENSE](LICENSE).
