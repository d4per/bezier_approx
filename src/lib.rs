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
//!   normal equations, so memory does not grow with the number of samples.
//! * **Inference**: [`BezierFn`]. Fixed size, no allocation, usable as a `const`.
//!   [`Fitter::to_rust_code`] emits such a constant.
//! * **Splines**: a [`Model`] with `segments > 1` splits the box into equal pieces per
//!   axis, each a Bézier polynomial, joined smoothly (a clamped uniform B-spline).
//! * **Outliers**: [`Fitter::solve_robust`] with [`RobustLoss::HUBER`] or
//!   [`RobustLoss::TUKEY`] limits or removes the influence of bad samples.
//! * **Model selection**: [`AutoFit`] tries Bézier degrees and spline sizes, scores
//!   them by cross-validation, recommends one, and lists caveats.

mod auto;
mod eval;
mod fit;

pub use auto::{AutoFit, AutoReport, Candidate, Caveat, Limit, RobustMode, Rule, Samples};
pub use eval::{bernstein_into, bspline_basis, BezierFn};
pub use fit::{ErrorReport, FitError, FitReport, Fitter, Model, RobustLoss, RobustReport};
