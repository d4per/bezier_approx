//! Function approximation with tensor-product Bézier polynomials.
//!
//! A Bézier function of degree `n` on `[lo, hi] ⊂ R^D` is
//!
//! ```text
//! f(x) = Σ_i  c_i · B_{i_0}(t_0) · B_{i_1}(t_1) · … · B_{i_{D-1}}(t_{D-1}),   t_d = (x_d - lo_d) / (hi_d - lo_d)
//! ```
//!
//! where `B_k` are the Bernstein polynomials of degree `n` (the same basis a Bézier
//! curve uses) and `c_i` are the control values, `(n+1)^D` of them.
//!
//! * **Training**: [`Fitter`]. Because `f` is *linear* in the control values, fitting is a
//!   linear least-squares problem with a unique optimum. Samples are streamed into the
//!   normal equations; with [`Fitter::without_sample_storage`] memory does not grow
//!   with the number of samples.
//! * **Inference**: [`BezierFn`]. Fixed size, no allocation, usable as a `const`.
//!   [`Fitter::to_rust_code`] emits such a constant.
//! * **Standalone code**: [`Fitter::to_rust_fn`] and [`Fitter::to_c_fn`] write the
//!   fitted function as a plain Rust or C function that needs neither this crate nor
//!   an allocator. [`Model::cost`] gives its operation count and table size.
//! * **Splines**: a [`Model`] with `segments > 1` splits the box into equal pieces per
//!   axis, each a Bézier polynomial, joined smoothly (a clamped uniform B-spline).
//! * **Outliers**: [`Fitter::solve_robust`] with [`RobustLoss::HUBER`] or
//!   [`RobustLoss::TUKEY`] limits or removes the influence of bad samples.
//! * **Model selection**: [`AutoFit`] tries Bézier degrees and spline sizes, scores
//!   them by cross-validation, recommends one, and lists caveats.
//!
//! # Example
//!
//! Train on the host and export the result as Rust source:
//!
//! ```
//! # #[cfg(feature = "std")]
//! # fn main() -> Result<(), bezier_approx::FitError> {
//! use bezier_approx::Fitter;
//!
//! let mut fit = Fitter::unit(1, 3); // degree 3 on [0, 1]
//! for i in 0..=20 {
//!     let x = i as f64 / 20.0;
//!     fit.add_sample([x], x * x)?;
//! }
//! fit.solve()?;
//! assert!((fit.predict([0.5])? - 0.25).abs() < 1e-6);
//! let code = fit.to_rust_code("SQUARE")?; // paste this into the program that needs it
//! # Ok(())
//! # }
//! # #[cfg(not(feature = "std"))]
//! # fn main() {}
//! ```
//!
//! The exported constant evaluates without allocation or training code:
//!
//! ```
//! // x² on [0, 1] as a quadratic Bézier: control values 0, 0, 1.
//! const SQUARE: bezier_approx::BezierFn<'static, 1, 3> =
//!     bezier_approx::BezierFn::new([0.0], [1.0], 1, &[0.0, 0.0, 1.0]);
//! assert!((SQUARE.eval([0.5]) - 0.25).abs() < 1e-15);
//! ```
//!
//! # Features
//!
//! * `std` (default): training ([`Fitter`], [`AutoFit`]). Without it the crate is
//!   `no_std` and contains only the evaluator ([`BezierFn`], [`bspline_basis`],
//!   [`bernstein_into`]), which needs no allocator either, so exported constants can
//!   run on microcontrollers.

#![cfg_attr(not(any(feature = "std", test)), no_std)]
#![warn(missing_docs)]

#[cfg(feature = "std")]
mod auto;
mod eval;
#[cfg(feature = "std")]
mod export;
#[cfg(feature = "std")]
mod fit;

#[cfg(feature = "std")]
pub use auto::{AutoFit, AutoReport, Candidate, Caveat, Limit, RobustMode, Rule, Samples};
pub use eval::{bernstein_into, bspline_basis, BezierFn};
#[cfg(feature = "std")]
pub use fit::{Cost, ErrorReport, FitError, FitReport, Fitter, Model, RobustLoss, RobustReport};
