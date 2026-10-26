//! Training: streaming least-squares fit of the control values.

use crate::eval::bspline_basis;
use std::fmt::{self, Write};

/// Errors from the training API.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum FitError {
    /// Input had the wrong number of coordinates.
    WrongDimension {
        /// Number of input dimensions of the fitter or sample set.
        expected: usize,
        /// Number of coordinates that were passed.
        got: usize,
    },
    /// A training input lies outside the box the fitter was created for.
    OutOfRange {
        /// The axis that is out of range.
        axis: usize,
        /// The offending coordinate.
        value: f64,
    },
    /// An input or output was NaN or infinite.
    NotFinite,
    /// `solve` was called before any sample was added, or `evaluate` or
    /// [`AutoFit::run`](crate::AutoFit::run) got no data.
    NoData,
    /// The system could not be solved (should not happen with smoothing > 0).
    Singular,
    /// `predict`, `std_error` or `to_rust_code` was called before `solve`.
    NotSolved,
    /// `std_error` needs more samples than coefficients to estimate the noise level.
    TooFewSamples,
    /// A sample weight was not finite and positive.
    InvalidWeight,
    /// `solve_robust` needs the samples, but sample storage was disabled.
    NoStoredSamples,
    /// [`AutoFit`](crate::AutoFit) found no model to try: every candidate needs at
    /// least 2 training samples per coefficient and must fit within the configured
    /// limits.
    NoCandidates,
    /// The name passed to `to_rust_code` is not an ASCII Rust identifier, or is a
    /// keyword.
    InvalidName,
}

impl fmt::Display for FitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FitError::WrongDimension { expected, got } => {
                write!(f, "expected {expected} input coordinates, got {got}")
            }
            FitError::OutOfRange { axis, value } => {
                write!(
                    f,
                    "input {value} on axis {axis} is outside the training box"
                )
            }
            FitError::NotFinite => write!(f, "input or output is not finite"),
            FitError::NoData => write!(f, "no samples have been added"),
            FitError::Singular => write!(f, "least-squares system is singular"),
            FitError::NotSolved => write!(f, "call solve() first"),
            FitError::TooFewSamples => {
                write!(f, "need more samples than coefficients to estimate noise")
            }
            FitError::InvalidWeight => write!(f, "sample weight must be finite and positive"),
            FitError::NoStoredSamples => {
                write!(
                    f,
                    "robust fitting needs stored samples (storage was disabled)"
                )
            }
            FitError::NoCandidates => write!(
                f,
                "no model to try: each needs at least 2 training samples per coefficient \
                 and must fit within the limits"
            ),
            FitError::InvalidName => write!(
                f,
                "constant name must be an ASCII Rust identifier that is not a keyword"
            ),
        }
    }
}

impl std::error::Error for FitError {}

/// The shape of a model: polynomial degree and number of pieces per axis.
///
/// `segments == 1` is a single Bézier polynomial over the whole box. With more
/// segments, the box is split into equal pieces along every axis, each piece is a
/// Bézier polynomial, and neighbours join smoothly (a clamped uniform B-spline,
/// with `degree − 1` continuous derivatives across the joins).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Model {
    /// Polynomial degree of each piece.
    pub degree: usize,
    /// Pieces per axis; 1 = a single Bézier polynomial.
    pub segments: usize,
}

impl Model {
    /// A single Bézier polynomial of the given degree.
    pub const fn bezier(degree: usize) -> Self {
        Model {
            degree,
            segments: 1,
        }
    }

    /// A B-spline of the given degree with `segments` pieces per axis.
    pub const fn spline(degree: usize, segments: usize) -> Self {
        Model { degree, segments }
    }

    /// Control values per axis, `degree + segments`.
    pub const fn per_axis(&self) -> usize {
        self.degree + self.segments
    }

    /// Total number of control values in `dims` dimensions, `(degree + segments)^dims`.
    /// Saturates at `usize::MAX` instead of overflowing.
    pub fn coefficients(&self, dims: usize) -> usize {
        saturating_pow(self.per_axis(), dims)
    }

    /// Number of coefficients combined in one evaluation, `(degree + 1)^dims`.
    /// Saturates at `usize::MAX` instead of overflowing.
    pub fn terms(&self, dims: usize) -> usize {
        saturating_pow(self.degree + 1, dims)
    }

    /// Work per call, and storage, of the standalone code that
    /// [`Fitter::to_rust_fn`] and [`Fitter::to_c_fn`] generate for this model in
    /// `dims` dimensions. Saturates at `usize::MAX` instead of overflowing.
    pub fn cost(&self, dims: usize) -> Cost {
        let terms = self.terms(dims);
        let horner = terms - 1;
        // Per axis: map x to the box (sub, mul) and clamp (2 cmp); splines also pick
        // the piece (1 cmp) and, unless constant, centre x within it (2 add, 1 mul).
        let (add, mul, cmp): (usize, usize, usize) = match (self.segments, self.degree) {
            (1, 0) => (0, 0, 0),
            (1, _) => (1, 1, 2),
            (_, 0) => (1, 1, 3),
            _ => (3, 2, 3),
        };
        Cost {
            mul: horner.saturating_add(mul.saturating_mul(dims)),
            add: horner.saturating_add(add.saturating_mul(dims)),
            cmp: cmp.saturating_mul(dims),
            values: saturating_pow(self.segments, dims).saturating_mul(terms),
        }
    }
}

fn saturating_pow(base: usize, exp: usize) -> usize {
    u32::try_from(exp).map_or(usize::MAX, |e| base.saturating_pow(e))
}

/// Work per call, and storage, of the standalone code from [`Fitter::to_rust_fn`]
/// and [`Fitter::to_c_fn`]; see [`Model::cost`].
///
/// The polynomial is evaluated by Horner's scheme, one axis at a time:
/// `(degree + 1)^dims − 1` multiply-add pairs, whatever the number of segments. Each
/// input coordinate adds a few operations to map it into the box (and, for splines,
/// to pick the piece). There are no divisions, square roots or loops; splines add one
/// float-to-integer conversion per axis and a table lookup.
///
/// [`BezierFn::eval`](crate::BezierFn::eval) sums the same number of terms, but also
/// evaluates the basis functions with the Cox–de Boor recurrence, which divides, so
/// it is slower than the standalone code.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub struct Cost {
    /// Floating-point multiplications.
    pub mul: usize,
    /// Floating-point additions and subtractions.
    pub add: usize,
    /// Comparisons, for clamping inputs to the box and picking the spline piece.
    pub cmp: usize,
    /// Coefficients stored in the code, `(segments · (degree + 1))^dims`. For a spline
    /// this is more than its control values, because every piece is stored as its
    /// own polynomial.
    pub values: usize,
}

impl Cost {
    /// Floating-point operations, `mul + add`.
    pub fn flops(&self) -> usize {
        self.mul.saturating_add(self.add)
    }
}

impl fmt::Display for Cost {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let n = |count: usize, what: &str| match count {
            1 => format!("1 {what}"),
            _ => format!("{count} {what}s"),
        };
        write!(
            f,
            "{}, {}, {}",
            n(self.mul, "multiplication"),
            n(self.add, "addition"),
            n(self.cmp, "comparison")
        )
    }
}

impl fmt::Display for Model {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.segments == 1 {
            write!(f, "Bézier, degree {}", self.degree)
        } else {
            write!(
                f,
                "spline, degree {}, {} segments",
                self.degree, self.segments
            )
        }
    }
}

/// Summary returned by [`Fitter::solve`].
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct FitReport {
    /// Number of training samples used.
    pub samples: usize,
    /// Number of control values, `(degree + segments)^dims`.
    pub coefficients: usize,
    /// Root-mean-square residual on the training data, weighted by the sample
    /// weights: `sqrt(Σ wᵢ rᵢ² / Σ wᵢ)`. Exact when samples are stored (the default).
    /// Without stored samples it is computed from the accumulated sums and cannot
    /// resolve residuals below roughly `1e-8 × typical |y|`.
    pub rms_residual: f64,
    /// Estimated noise level of a sample of average weight,
    /// `sqrt(Σ wᵢ rᵢ² / w̄ / (samples - coefficients))` with `w̄` the mean weight.
    /// `None` when there are not more samples than coefficients.
    pub sigma: Option<f64>,
}

/// Loss function for [`Fitter::solve_robust`].
///
/// Residuals are measured in units of a robust noise estimate (1.4826 × the median
/// absolute residual), so the tuning constants mean "this many standard deviations".
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum RobustLoss {
    /// Quadratic for residuals up to `k` standard deviations, linear beyond.
    /// Outliers keep a small, bounded influence. Convex, so the result is unique.
    Huber {
        /// Threshold in standard deviations.
        k: f64,
    },
    /// Tukey's bisquare: residuals beyond `c` standard deviations get weight 0 and are
    /// ignored completely. Not convex, so it is started from a Huber fit.
    Tukey {
        /// Rejection threshold in standard deviations.
        c: f64,
    },
}

impl RobustLoss {
    /// Huber with `k = 1.345`: 95 % as efficient as least squares on clean Gaussian noise.
    pub const HUBER: RobustLoss = RobustLoss::Huber { k: 1.345 };
    /// Tukey with `c = 4.685`: 95 % as efficient as least squares on clean Gaussian noise.
    pub const TUKEY: RobustLoss = RobustLoss::Tukey { c: 4.685 };

    /// IRLS weight for a residual of `u` standard deviations.
    fn weight(self, u: f64) -> f64 {
        let a = u.abs();
        match self {
            RobustLoss::Huber { k } => {
                if a <= k {
                    1.0
                } else {
                    k / a
                }
            }
            RobustLoss::Tukey { c } => {
                if a < c {
                    let z = 1.0 - (a / c) * (a / c);
                    z * z
                } else {
                    0.0
                }
            }
        }
    }

    fn is_valid(self) -> bool {
        let t = match self {
            RobustLoss::Huber { k } => k,
            RobustLoss::Tukey { c } => c,
        };
        t.is_finite() && t > 0.0
    }
}

/// Summary returned by [`Fitter::solve_robust`].
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct RobustReport {
    /// Number of training samples.
    pub samples: usize,
    /// Number of control values, `(degree + segments)^dims`.
    pub coefficients: usize,
    /// Reweighting iterations performed (including the Huber warm-up for Tukey).
    pub iterations: usize,
    /// Whether the coefficients settled before the iteration limit.
    pub converged: bool,
    /// Robust noise estimate for a sample of average weight, 1.4826 × median absolute
    /// residual. Largely unaffected by outliers as long as fewer than about half the
    /// samples are outliers.
    pub scale: f64,
    /// Weighted RMS residual over all samples, outliers included.
    pub rms_residual: f64,
    /// Samples with robust weight 0, i.e. ignored entirely (only Tukey does this).
    pub rejected: usize,
}

/// Error statistics of the fitted function on a data set, from [`Fitter::evaluate`].
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct ErrorReport {
    /// Number of samples evaluated.
    pub count: usize,
    /// Root-mean-square error.
    pub rms: f64,
    /// Largest absolute error.
    pub max_abs: f64,
    /// Input where the largest absolute error occurred.
    pub worst_input: Vec<f64>,
}

/// Fits a tensor-product Bézier function or B-spline (see [`Model`]) to samples
/// `(x, y)` with `x` in a box.
///
/// Samples are accumulated into the normal equations `AᵀA c = Aᵀy`. Adding a sample
/// costs `O(T²)` with `T = (degree+1)^dims` nonzero basis functions per sample, and
/// [`solve`](Fitter::solve) costs `O(N³)` with `N` the number of coefficients.
/// Call it whenever you want the coefficients to reflect the samples added so far.
///
/// The samples themselves are also kept (`dims + 2` floats each), because
/// [`solve_robust`](Fitter::solve_robust) has to revisit them. For pure streaming of
/// very many samples, [`without_sample_storage`](Fitter::without_sample_storage)
/// makes memory independent of the number of samples.
#[derive(Clone)]
pub struct Fitter {
    dims: usize,
    degree: usize,
    segments: usize,
    lo: Vec<f64>,
    hi: Vec<f64>,
    smoothing: f64,
    /// Lower triangle of AᵀA, row-major N×N.
    ata: Vec<f64>,
    aty: Vec<f64>,
    yty: f64,
    count: usize,
    /// Sum of the sample weights.
    wsum: f64,
    // Stored samples: inputs (flat, `dims` per sample), outputs, weights.
    store_samples: bool,
    xs: Vec<f64>,
    ys: Vec<f64>,
    ws: Vec<f64>,
    // State from the last solve.
    coeffs: Option<Vec<f64>>,
    chol: Option<Vec<f64>>,
    report: Option<FitReport>,
    robust_weights: Option<Vec<f64>>,
    /// Mean sample weight at the last solve: the unit of `report.sigma`, residuals
    /// and support until the next solve, even if samples are added in between.
    solved_mean_weight: f64,
}

// The normal equations and stored samples are left out: they can be huge.
impl fmt::Debug for Fitter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Fitter")
            .field("model", &self.model())
            .field("lo", &self.lo)
            .field("hi", &self.hi)
            .field("smoothing", &self.smoothing)
            .field("samples", &self.count)
            .field("store_samples", &self.store_samples)
            .field("report", &self.report)
            .field("robust", &self.robust_weights.is_some())
            .finish_non_exhaustive()
    }
}

impl Fitter {
    /// Creates a fitter for a single Bézier polynomial of the given degree on the box
    /// `[lo, hi]`; the number of dimensions is `lo.len()`.
    ///
    /// # Panics
    /// If `lo` and `hi` differ in length, are empty, `lo[d] >= hi[d]` or the width
    /// `hi[d] - lo[d]` is not finite on some axis, or the model is too large to
    /// represent (`n²` normal-matrix entries overflow `usize`).
    pub fn new(degree: usize, lo: &[f64], hi: &[f64]) -> Self {
        Self::with_model(Model::bezier(degree), lo, hi)
    }

    /// Creates a fitter for a B-spline with `segments` pieces per axis.
    ///
    /// # Panics
    /// As [`with_model`](Fitter::with_model).
    pub fn spline(degree: usize, segments: usize, lo: &[f64], hi: &[f64]) -> Self {
        Self::with_model(Model::spline(degree, segments), lo, hi)
    }

    /// Creates a fitter for any [`Model`] on the box `[lo, hi]`.
    ///
    /// # Panics
    /// As [`new`](Fitter::new), and if `model.segments == 0`.
    pub fn with_model(model: Model, lo: &[f64], hi: &[f64]) -> Self {
        assert!(model.segments >= 1, "need at least one segment");
        assert_valid_box(lo, hi);
        let dims = lo.len();
        let n = model.coefficients(dims);
        let entries = n
            .checked_mul(n)
            .filter(|_| model.terms(dims) < usize::MAX)
            .unwrap_or_else(|| panic!("model too large: {model} in {dims} dimensions"));
        Fitter {
            dims,
            degree: model.degree,
            segments: model.segments,
            lo: lo.to_vec(),
            hi: hi.to_vec(),
            smoothing: 1e-9,
            ata: vec![0.0; entries],
            aty: vec![0.0; n],
            yty: 0.0,
            count: 0,
            wsum: 0.0,
            store_samples: true,
            xs: Vec::new(),
            ys: Vec::new(),
            ws: Vec::new(),
            coeffs: None,
            chol: None,
            report: None,
            robust_weights: None,
            solved_mean_weight: 1.0,
        }
    }

    /// Creates a Bézier fitter on the unit box `[0, 1]^dims`.
    ///
    /// # Panics
    /// If `dims == 0` or the model is too large (see [`new`](Fitter::new)).
    pub fn unit(dims: usize, degree: usize) -> Self {
        Self::new(degree, &vec![0.0; dims], &vec![1.0; dims])
    }

    /// Creates a B-spline fitter on the unit box `[0, 1]^dims`.
    ///
    /// # Panics
    /// If `dims == 0` or `segments == 0`, or the model is too large (see
    /// [`new`](Fitter::new)).
    pub fn unit_spline(dims: usize, degree: usize, segments: usize) -> Self {
        Self::spline(degree, segments, &vec![0.0; dims], &vec![1.0; dims])
    }

    /// Sets the smoothness penalty (default `1e-9`).
    ///
    /// The penalty is on second differences of neighbouring control values along each
    /// axis, i.e. on curvature of the control net. It keeps the fit well defined in
    /// regions without data (they are bridged smoothly) and damps overfitting of noisy
    /// data. It is scaled relative to the amount of data, so the same value gives the
    /// same trade-off for 100 or 100 000 samples. Try `1e-4`–`1e-2` for noisy data.
    ///
    /// It also biases the fit slightly, by roughly this value relative to the size of
    /// the function's curvature. For exact data from a formula, `0.0` gives the most
    /// accurate fit, as long as the data leaves no gaps.
    ///
    /// Models with only two control values per axis (degree 1, one segment) have no
    /// curvature to penalise, so smoothing does not affect them.
    ///
    /// # Panics
    /// If `smoothing` is negative or not finite.
    pub fn with_smoothing(mut self, smoothing: f64) -> Self {
        assert!(
            smoothing >= 0.0 && smoothing.is_finite(),
            "smoothing must be finite and non-negative"
        );
        self.smoothing = smoothing;
        self
    }

    /// Stops storing samples (and drops any already stored), so memory no longer grows
    /// with the number of samples. [`solve_robust`](Fitter::solve_robust) then becomes
    /// unavailable.
    pub fn without_sample_storage(mut self) -> Self {
        self.store_samples = false;
        self.xs = Vec::new();
        self.ys = Vec::new();
        self.ws = Vec::new();
        self
    }

    /// Number of input dimensions.
    pub fn dims(&self) -> usize {
        self.dims
    }

    /// Polynomial degree of each piece.
    pub fn degree(&self) -> usize {
        self.degree
    }

    /// Pieces per axis (1 for a single Bézier polynomial).
    pub fn segments(&self) -> usize {
        self.segments
    }

    /// Degree and segments as a [`Model`].
    pub fn model(&self) -> Model {
        Model::spline(self.degree, self.segments)
    }

    /// Lower corner of the box.
    pub fn lo(&self) -> &[f64] {
        &self.lo
    }

    /// Upper corner of the box.
    pub fn hi(&self) -> &[f64] {
        &self.hi
    }

    /// Number of control values, `(degree + segments)^dims`.
    pub fn num_coefficients(&self) -> usize {
        self.aty.len()
    }

    /// Number of samples added so far.
    pub fn num_samples(&self) -> usize {
        self.count
    }

    /// The control values from the last `solve`, axis 0 varying fastest.
    pub fn coefficients(&self) -> Option<&[f64]> {
        self.coeffs.as_deref()
    }

    /// The report from the last `solve` or `solve_robust`.
    pub fn report(&self) -> Option<&FitReport> {
        self.report.as_ref()
    }

    /// Stored sample `i` (in the order added) as `(x, y, weight)`.
    pub fn sample(&self, i: usize) -> Option<(&[f64], f64, f64)> {
        let y = *self.ys.get(i)?;
        Some((&self.xs[i * self.dims..(i + 1) * self.dims], y, self.ws[i]))
    }

    /// Robust weights from the last `solve_robust`, one per sample that had been added
    /// by then, in the order added: 1 = fully trusted, 0 = ignored as an outlier.
    /// `None` after a plain `solve`.
    pub fn robust_weights(&self) -> Option<&[f64]> {
        self.robust_weights.as_deref()
    }

    /// Indices of stored samples whose residual under the current fit exceeds
    /// `threshold` × the noise estimate (σ from the last solve). `3.0` is a common choice.
    pub fn outliers(&self, threshold: f64) -> Result<Vec<usize>, FitError> {
        if !self.store_samples {
            return Err(FitError::NoStoredSamples);
        }
        let c = self.coeffs.as_ref().ok_or(FitError::NotSolved)?;
        let sigma = self
            .report
            .as_ref()
            .and_then(|r| r.sigma)
            .ok_or(FitError::TooFewSamples)?;
        let r = self.weighted_residuals(c, self.solved_mean_weight);
        Ok((0..r.len())
            .filter(|&i| r[i].abs() > threshold * sigma.max(self.noise_floor()))
            .collect())
    }

    /// Adds one training sample with weight 1. `x` must lie inside the box.
    pub fn add_sample(&mut self, x: impl AsRef<[f64]>, y: f64) -> Result<(), FitError> {
        self.add_weighted_sample(x, y, 1.0)
    }

    /// Adds one training sample with a relative weight.
    ///
    /// Use weight `1/variance` (relative) when measurements differ in precision: a
    /// sample with half the standard deviation of the others gets weight 4. Weight 2
    /// is equivalent to adding the sample twice.
    pub fn add_weighted_sample(
        &mut self,
        x: impl AsRef<[f64]>,
        y: f64,
        weight: f64,
    ) -> Result<(), FitError> {
        if !y.is_finite() {
            return Err(FitError::NotFinite);
        }
        if !(weight.is_finite() && weight > 0.0) {
            return Err(FitError::InvalidWeight);
        }
        let x = x.as_ref();
        let (idx, val) = self.basis(x, true)?;
        accumulate(&mut self.ata, &mut self.aty, &idx, &val, y, weight);
        self.yty += weight * y * y;
        self.wsum += weight;
        self.count += 1;
        if self.store_samples {
            self.xs.extend_from_slice(x);
            self.ys.push(y);
            self.ws.push(weight);
        }
        Ok(())
    }

    /// Solves for the control values by (weighted) least squares, using all samples
    /// added so far.
    pub fn solve(&mut self) -> Result<FitReport, FitError> {
        if self.count == 0 {
            return Err(FitError::NoData);
        }
        let n = self.num_coefficients();
        let (c, m) = self.solve_system(&self.ata, &self.aty)?;
        let rss = if self.store_samples {
            self.weighted_residuals(&c, self.mean_weight())
                .iter()
                .map(|r| r * r)
                .sum()
        } else {
            // RSS = yᵀy − 2cᵀAᵀy + cᵀAᵀAc, from the accumulated statistics.
            let mut quad = 0.0;
            for i in 0..n {
                let row = &self.ata[i * n..i * n + i];
                let off: f64 = row.iter().zip(&c).map(|(a, cj)| a * cj).sum();
                quad += c[i] * (self.ata[i * n + i] * c[i] + 2.0 * off);
            }
            let cty: f64 = c.iter().zip(&self.aty).map(|(a, b)| a * b).sum();
            (self.yty - 2.0 * cty + quad).max(0.0) / self.mean_weight()
        };
        let report = FitReport {
            samples: self.count,
            coefficients: n,
            rms_residual: (rss / self.count as f64).sqrt(),
            sigma: (self.count > n).then(|| (rss / (self.count - n) as f64).sqrt()),
        };
        self.coeffs = Some(c);
        self.chol = Some(m);
        self.report = Some(report.clone());
        self.robust_weights = None;
        self.solved_mean_weight = self.mean_weight();
        Ok(report)
    }

    /// Solves with a robust loss that limits (Huber) or removes (Tukey) the influence
    /// of outliers, by iteratively reweighted least squares:
    ///
    /// 1. start from the ordinary least-squares fit;
    /// 2. estimate the noise level robustly from the median absolute residual;
    /// 3. give each sample a weight from its residual in units of that noise level;
    /// 4. re-solve the weighted least-squares problem; repeat until the fit settles.
    ///
    /// Works as long as outliers are less than roughly half the samples. An outlier
    /// in a region with almost no other data still pulls the fit, because nothing
    /// contradicts it; [`support`](Fitter::support) near 1 flags such regions.
    ///
    /// Afterwards, [`robust_weights`](Fitter::robust_weights) tells which samples were
    /// distrusted, and `predict`, `std_error` and `to_rust_code` use the robust fit.
    /// Each iteration costs one pass over the stored samples plus one `O(N³)` solve.
    ///
    /// # Panics
    /// If the tuning constant of `loss` is not finite and positive.
    pub fn solve_robust(&mut self, loss: RobustLoss) -> Result<RobustReport, FitError> {
        const MAX_ITER: usize = 100;
        if self.count == 0 {
            return Err(FitError::NoData);
        }
        if !self.store_samples {
            return Err(FitError::NoStoredSamples);
        }
        assert!(
            loss.is_valid(),
            "robust tuning constant must be finite and positive"
        );
        let n = self.num_coefficients();
        let m = self.count;
        let mean_w = self.mean_weight();

        let (mut c, mut chol) = self.solve_system(&self.ata, &self.aty)?;
        let mut weights = vec![1.0; m];
        let mut scale = 0.0;
        let mut iterations = 0;
        let mut converged = false;
        // Tukey's loss has local minima, so it starts from a converged Huber fit.
        let phases: &[RobustLoss] = match loss {
            RobustLoss::Huber { .. } => &[loss],
            RobustLoss::Tukey { .. } => &[RobustLoss::HUBER, loss],
        };
        for &phase in phases {
            converged = false;
            for _ in 0..MAX_ITER {
                iterations += 1;
                let r = self.weighted_residuals(&c, mean_w);
                scale = robust_scale(&r);
                let s = scale.max(self.noise_floor());
                for (w, ri) in weights.iter_mut().zip(&r) {
                    *w = phase.weight(ri / s);
                }
                let (ata, aty) = self.reweighted_normal_equations(&weights);
                let (c_new, chol_new) = self.solve_system(&ata, &aty)?;
                let max_c = c_new.iter().fold(0.0f64, |a, v| a.max(v.abs()));
                let max_dc = c_new
                    .iter()
                    .zip(&c)
                    .fold(0.0f64, |a, (x, y)| a.max((x - y).abs()));
                c = c_new;
                chol = chol_new;
                if max_dc <= 1e-10 * max_c {
                    converged = true;
                    break;
                }
            }
        }

        let r = self.weighted_residuals(&c, mean_w);
        let rms_residual = (r.iter().map(|v| v * v).sum::<f64>() / m as f64).sqrt();
        let report = RobustReport {
            samples: m,
            coefficients: n,
            iterations,
            converged,
            scale,
            rms_residual,
            rejected: weights.iter().filter(|&&w| w == 0.0).count(),
        };
        self.coeffs = Some(c);
        self.chol = Some(chol);
        self.report = Some(FitReport {
            samples: m,
            coefficients: n,
            rms_residual,
            sigma: (m > n).then_some(scale),
        });
        self.robust_weights = Some(weights);
        self.solved_mean_weight = mean_w;
        Ok(report)
    }

    /// Evaluates the fitted function. Inputs outside the box are clamped,
    /// exactly like [`BezierFn::eval`](crate::BezierFn::eval).
    pub fn predict(&self, x: impl AsRef<[f64]>) -> Result<f64, FitError> {
        let c = self.coeffs.as_ref().ok_or(FitError::NotSolved)?;
        let (idx, val) = self.basis(x.as_ref(), false)?;
        Ok(idx.iter().zip(&val).map(|(&i, v)| c[i] * v).sum())
    }

    /// Approximate 1-σ standard error of the prediction at `x`.
    ///
    /// Computed as `σ·sqrt(b(x)ᵀ M⁻¹ b(x))` from the coefficient covariance (after
    /// `solve_robust`, σ is the robust noise estimate). It is large where data is
    /// sparse and small where it is dense. It only accounts for random noise: it
    /// cannot detect that the degree is too low for the function's shape (check with
    /// [`evaluate`](Fitter::evaluate) on held-out data), nor that the few samples a
    /// region rests on are wrong (check with [`support`](Fitter::support)).
    pub fn std_error(&self, x: impl AsRef<[f64]>) -> Result<f64, FitError> {
        if self.coeffs.is_none() {
            return Err(FitError::NotSolved);
        }
        let sigma = self
            .report
            .as_ref()
            .and_then(|r| r.sigma)
            .ok_or(FitError::TooFewSamples)?;
        Ok(sigma / self.support(x)?.sqrt())
    }

    /// Roughly how many samples' worth of data back the prediction at `x`,
    /// `1 / (b(x)ᵀ M⁻¹ b(x))`, counting samples of average weight (rejected outliers
    /// count 0).
    ///
    /// Tens or more: the prediction is an average over many samples, so a bad one is
    /// outvoted and shows up as an outlier. Around 1: the fit at `x` is pinned by a
    /// single sample and simply passes through it, so an error in that sample cannot
    /// be detected by any method, robust or not. Below 1: there is effectively no data
    /// and the smoothing penalty is filling in. Unlike `std_error`, this needs no
    /// noise estimate, so it also works with few samples. Like `predict`, it reflects
    /// the last solve, not samples added since.
    pub fn support(&self, x: impl AsRef<[f64]>) -> Result<f64, FitError> {
        let l = self.chol.as_ref().ok_or(FitError::NotSolved)?;
        let (idx, val) = self.basis(x.as_ref(), false)?;
        let n = self.num_coefficients();
        let mut z = vec![0.0; n];
        for (&i, &v) in idx.iter().zip(&val) {
            z[i] = v;
        }
        // Entries before the first nonzero stay zero in L z = b; skip them.
        solve_lower_from(l, n, &mut z, idx[0]);
        Ok(1.0 / z[idx[0]..].iter().map(|v| v * v).sum::<f64>() / self.solved_mean_weight)
    }

    /// Error statistics on any data set, typically held-out validation samples.
    /// Fails with [`FitError::NoData`] if `data` is empty.
    pub fn evaluate<X, I>(&self, data: I) -> Result<ErrorReport, FitError>
    where
        X: AsRef<[f64]>,
        I: IntoIterator<Item = (X, f64)>,
    {
        let mut count = 0;
        let mut sq = 0.0;
        let mut max_abs = 0.0;
        let mut worst_input = Vec::new();
        for (x, y) in data {
            let e = (self.predict(x.as_ref())? - y).abs();
            count += 1;
            sq += e * e;
            if e >= max_abs {
                max_abs = e;
                worst_input = x.as_ref().to_vec();
            }
        }
        if count == 0 {
            return Err(FitError::NoData);
        }
        Ok(ErrorReport {
            count,
            rms: (sq / count as f64).sqrt(),
            max_abs,
            worst_input,
        })
    }

    /// Emits Rust source for a `const` [`BezierFn`](crate::BezierFn) holding the
    /// fitted function, e.g.
    /// `pub const NAME: bezier_approx::BezierFn<'static, 2, 4> = bezier_approx::BezierFn::new(…);`
    /// (the second parameter is degree + 1). Floats are printed in shortest round-trip
    /// form, so the constant reproduces the fitted coefficients bit for bit, and the
    /// shape is checked at compile time by [`BezierFn::new`](crate::BezierFn::new).
    ///
    /// Fails with [`FitError::InvalidName`] unless `name` is an ASCII identifier that
    /// is not a keyword, so the output is always a single valid item.
    pub fn to_rust_code(&self, name: &str) -> Result<String, FitError> {
        if !is_identifier(name) {
            return Err(FitError::InvalidName);
        }
        let c = self.coeffs.as_ref().ok_or(FitError::NotSolved)?;
        if c.iter().any(|v| !v.is_finite()) {
            return Err(FitError::NotFinite);
        }
        let n1 = self.degree + self.segments; // control values per axis
        let p1 = self.degree + 1;
        let arr = |v: &[f64]| {
            let items: Vec<String> = v.iter().map(|x| format!("{x:?}")).collect();
            format!("[{}]", items.join(", "))
        };
        let mut s = String::new();
        let _ = writeln!(
            s,
            "// Generated by bezier_approx: {}-D {}, {} coefficients.",
            self.dims,
            self.model(),
            c.len()
        );
        if let Some(r) = &self.report {
            let _ = writeln!(
                s,
                "// Trained on {} samples, training RMS residual {:.3e}.",
                r.samples, r.rms_residual
            );
        }
        let _ = writeln!(s, "#[rustfmt::skip]");
        let _ = writeln!(
            s,
            "#[allow(clippy::unreadable_literal, clippy::approx_constant)]"
        );
        let _ = writeln!(
            s,
            "pub const {name}: bezier_approx::BezierFn<'static, {}, {p1}> = bezier_approx::BezierFn::new(",
            self.dims
        );
        let _ = writeln!(s, "    {}, // lo", arr(&self.lo));
        let _ = writeln!(s, "    {}, // hi", arr(&self.hi));
        let _ = writeln!(s, "    {}, // segments", self.segments);
        let _ = writeln!(s, "    &[");
        for (row, chunk) in c.chunks(n1).enumerate() {
            // Indices of the higher axes for this run along axis 0.
            let label = (self.dims > 1).then(|| {
                let mut r = row;
                let idx: Vec<String> = (1..self.dims)
                    .map(|d| {
                        let i = r % n1;
                        r /= n1;
                        format!("i{d}={i}")
                    })
                    .collect();
                idx.join(", ")
            });
            // At most 5 values per line; the label goes on the run's first line.
            for (k, line) in chunk.chunks(5).enumerate() {
                let items: Vec<String> = line.iter().map(|x| format!("{x:?}")).collect();
                let _ = write!(s, "        {},", items.join(", "));
                if let (0, Some(label)) = (k, &label) {
                    let _ = write!(s, " // {label}");
                }
                s.push('\n');
            }
        }
        let _ = writeln!(s, "    ],");
        let _ = writeln!(s, ");");
        Ok(s)
    }

    /// Emits a standalone Rust function `pub fn name(x: [f64; D]) -> f64` that
    /// computes the fitted function with plain arithmetic: no crate, no allocation,
    /// `no_std`-compatible. Like [`BezierFn::eval`](crate::BezierFn::eval), it clamps
    /// inputs to the box.
    ///
    /// The function is written as polynomials in power form, evaluated by Horner's
    /// scheme. Each input is first mapped to `u ∈ [-1, 1]` across the box (or, for
    /// a spline, across its piece), because powers of a centred variable stay well
    /// conditioned, whereas a degree-15 polynomial in `x ∈ [0, 1]` itself can lose half
    /// its digits to cancellation. A single Bézier polynomial is written out with its
    /// coefficients inline; a spline stores each piece's coefficients in a table and
    /// picks the row for the piece `x` falls in. [`Model::cost`] gives the operation
    /// count and table size, which also appear in a comment above the function, with
    /// the largest difference from [`predict`](Fitter::predict) at 1000 test inputs.
    ///
    /// Fails with [`FitError::InvalidName`] unless `name` is an ASCII identifier that
    /// is not a keyword.
    pub fn to_rust_fn(&self, name: &str) -> Result<String, FitError> {
        crate::export::standalone(self, name, crate::export::Lang::Rust)
    }

    /// Emits a standalone C99 function `double name(const double x[D])`, the C
    /// counterpart of [`to_rust_fn`](Fitter::to_rust_fn). It needs no headers or
    /// libraries, and does not convert NaN to an integer (which would be undefined
    /// behaviour): NaN inputs give NaN, as in Rust.
    ///
    /// Fails with [`FitError::InvalidName`] unless `name` is an ASCII C identifier that
    /// is not a keyword.
    pub fn to_c_fn(&self, name: &str) -> Result<String, FitError> {
        crate::export::standalone(self, name, crate::export::Lang::C)
    }

    /// Mean sample weight. Residuals, noise estimates and support are expressed for a
    /// sample of this weight, so scaling all weights by a constant changes nothing.
    fn mean_weight(&self) -> f64 {
        if self.count == 0 {
            1.0
        } else {
            self.wsum / self.count as f64
        }
    }

    /// Residuals below this are floating-point rounding, not noise: 1e-9 × the
    /// (weighted) RMS of the outputs. Without it, exact data (residuals ~1e-13) would
    /// make rounding differences look like outliers.
    fn noise_floor(&self) -> f64 {
        let y_rms = (self.yty / self.wsum.max(f64::MIN_POSITIVE)).sqrt();
        (1e-9 * y_rms).max(f64::MIN_POSITIVE)
    }

    /// Adds the smoothness penalty and a tiny ridge term to a copy of the normal
    /// matrix, factorises it and solves. Returns `(coefficients, cholesky_factor)`.
    fn solve_system(&self, ata: &[f64], aty: &[f64]) -> Result<(Vec<f64>, Vec<f64>), FitError> {
        let n = aty.len();
        let mean_diag = (0..n).map(|i| ata[i * n + i]).sum::<f64>() / n as f64;
        let mut m = ata.to_vec();
        add_smoothness_penalty(
            &mut m,
            n,
            self.dims,
            self.degree + self.segments,
            self.smoothing * mean_diag,
        );
        // Tiny ridge term so the factorisation never fails on degenerate data.
        let jitter = 1e-12 * mean_diag.max(f64::MIN_POSITIVE);
        for i in 0..n {
            m[i * n + i] += jitter;
        }
        if !cholesky(&mut m, n) {
            return Err(FitError::Singular);
        }
        let mut c = aty.to_vec();
        solve_lower(&m, n, &mut c);
        solve_upper(&m, n, &mut c);
        Ok((c, m))
    }

    /// Residuals `sqrt(wᵢ / w̄)·(yᵢ − f(xᵢ))` of the stored samples, with `w̄ = mean_w`
    /// the mean weight, so that all of them share the noise level of an average-weight
    /// sample.
    fn weighted_residuals(&self, c: &[f64], mean_w: f64) -> Vec<f64> {
        (0..self.ys.len())
            .map(|i| {
                let x = &self.xs[i * self.dims..(i + 1) * self.dims];
                let (idx, val) = self.basis(x, false).expect("stored samples are valid");
                let fx: f64 = idx.iter().zip(&val).map(|(&i, v)| c[i] * v).sum();
                (self.ws[i] / mean_w).sqrt() * (self.ys[i] - fx)
            })
            .collect()
    }

    /// Normal equations of the stored samples with weights `wᵢ · robustᵢ`.
    fn reweighted_normal_equations(&self, robust: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let n = self.num_coefficients();
        let mut ata = vec![0.0; n * n];
        let mut aty = vec![0.0; n];
        for (i, &rw) in robust.iter().enumerate() {
            let w = self.ws[i] * rw;
            if w > 0.0 {
                let x = &self.xs[i * self.dims..(i + 1) * self.dims];
                let (idx, val) = self.basis(x, false).expect("stored samples are valid");
                accumulate(&mut ata, &mut aty, &idx, &val, self.ys[i], w);
            }
        }
        (ata, aty)
    }

    /// The `(degree+1)^dims` nonzero tensor-product basis functions at `x`, as
    /// (coefficient index, value) pairs with indices in ascending order.
    #[allow(clippy::needless_range_loop)]
    fn basis(&self, x: &[f64], strict: bool) -> Result<(Vec<usize>, Vec<f64>), FitError> {
        if x.len() != self.dims {
            return Err(FitError::WrongDimension {
                expected: self.dims,
                got: x.len(),
            });
        }
        let p1 = self.degree + 1;
        let per_axis = self.degree + self.segments;
        let terms = p1.pow(self.dims as u32);
        let mut val = vec![0.0; terms];
        let mut idx = vec![0usize; terms];
        let mut b1 = vec![0.0; p1];
        val[0] = 1.0;
        let mut len = 1;
        let mut stride = 1;
        for d in 0..self.dims {
            if !x[d].is_finite() {
                return Err(FitError::NotFinite);
            }
            let mut t = (x[d] - self.lo[d]) / (self.hi[d] - self.lo[d]);
            if strict && !(-1e-12..=1.0 + 1e-12).contains(&t) {
                return Err(FitError::OutOfRange {
                    axis: d,
                    value: x[d],
                });
            }
            t = t.clamp(0.0, 1.0);
            let first = bspline_basis(t, self.segments, &mut b1);
            // Expand in place: new[i + len·k] = old[i]·b1[k]; k = 0 last so old survives.
            for k in (0..p1).rev() {
                for i in 0..len {
                    val[i + len * k] = val[i] * b1[k];
                    idx[i + len * k] = idx[i] + (first + k) * stride;
                }
            }
            len *= p1;
            stride *= per_axis;
        }
        Ok((idx, val))
    }
}

/// Panics unless `lo` and `hi` describe a box with at least one axis, each with
/// `lo < hi` and a finite width (what [`BezierFn::new`](crate::BezierFn::new) accepts).
pub(crate) fn assert_valid_box(lo: &[f64], hi: &[f64]) {
    assert!(!lo.is_empty(), "need at least one dimension");
    assert_eq!(lo.len(), hi.len(), "lo and hi must have the same length");
    for (d, (&a, &b)) in lo.iter().zip(hi).enumerate() {
        // Also rejects infinite or NaN bounds: their widths are not finite.
        assert!(
            a < b && (b - a).is_finite(),
            "axis {d}: need lo < hi and a finite width"
        );
    }
}

/// Whether `name` can name the exported constant: an ASCII identifier other than
/// `_` or a keyword (strict or reserved, in any edition).
pub(crate) fn is_identifier(name: &str) -> bool {
    const KEYWORDS: &[&str] = &[
        "Self", "abstract", "as", "async", "await", "become", "box", "break", "const", "continue",
        "crate", "do", "dyn", "else", "enum", "extern", "false", "final", "fn", "for", "gen", "if",
        "impl", "in", "let", "loop", "macro", "match", "mod", "move", "mut", "override", "priv",
        "pub", "ref", "return", "self", "static", "struct", "super", "trait", "true", "try",
        "type", "typeof", "unsafe", "unsized", "use", "virtual", "where", "while", "yield",
    ];
    let mut chars = name.chars();
    matches!(chars.next(), Some(c) if c.is_ascii_alphabetic() || c == '_')
        && chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
        && name != "_"
        && !KEYWORDS.contains(&name)
}

/// Adds `w·b·bᵀ` to the lower triangle of `ata` and `w·y·b` to `aty`, for a sparse
/// `b` given as ascending indices `idx` with values `val`.
fn accumulate(ata: &mut [f64], aty: &mut [f64], idx: &[usize], val: &[f64], y: f64, w: f64) {
    let n = aty.len();
    for (a, (&ia, &va)) in idx.iter().zip(val).enumerate() {
        let wva = w * va;
        if wva != 0.0 {
            let row = &mut ata[ia * n..];
            for (&ib, &vb) in idx[..=a].iter().zip(&val[..=a]) {
                row[ib] += wva * vb;
            }
            aty[ia] += wva * y;
        }
    }
}

/// Robust standard deviation of residuals: 1.4826 × median(|r|).
/// (1.4826 makes it match the standard deviation for Gaussian noise.)
fn robust_scale(r: &[f64]) -> f64 {
    let mut a: Vec<f64> = r.iter().map(|v| v.abs()).collect();
    let mid = a.len() / 2;
    let (_, median, _) = a.select_nth_unstable_by(mid, f64::total_cmp);
    1.4826 * *median
}

/// Adds `w · Σ_axes DᵀD` to the lower triangle of `m`, where `D` takes second
/// differences of neighbouring control values along one axis; `per_axis` is the
/// number of values per axis. With fewer than three values per axis there is no
/// curvature to penalise and nothing is added (a first-difference penalty would pull
/// linear fits towards constants). For B-splines this is the P-spline penalty of
/// Eilers & Marx.
fn add_smoothness_penalty(m: &mut [f64], n: usize, dims: usize, per_axis: usize, w: f64) {
    if per_axis < 3 || w == 0.0 {
        return;
    }
    let stencil = [1.0, -2.0, 1.0];
    let mut stride = 1;
    for _ in 0..dims {
        for base in 0..n {
            if (base / stride) % per_axis + 2 >= per_axis {
                continue;
            }
            for (ka, &sa) in stencil.iter().enumerate() {
                let a = base + ka * stride;
                for (kb, &sb) in stencil.iter().enumerate() {
                    let b = base + kb * stride;
                    if b <= a {
                        m[a * n + b] += w * sa * sb;
                    }
                }
            }
        }
        stride *= per_axis;
    }
}

/// In-place Cholesky factorisation of the lower triangle of `m` (row-major n×n).
fn cholesky(m: &mut [f64], n: usize) -> bool {
    for j in 0..n {
        let (head, tail) = m.split_at_mut(j * n + j);
        let row_j = &head[j * n..];
        let d = tail[0] - row_j.iter().map(|v| v * v).sum::<f64>();
        if d.is_nan() || d <= 0.0 {
            return false;
        }
        let d = d.sqrt();
        tail[0] = d;
        for i in j + 1..n {
            let s: f64 = (0..j).map(|k| m[i * n + k] * m[j * n + k]).sum();
            m[i * n + j] = (m[i * n + j] - s) / d;
        }
    }
    true
}

/// Solves `L z = v` in place.
fn solve_lower(l: &[f64], n: usize, v: &mut [f64]) {
    solve_lower_from(l, n, v, 0);
}

/// Solves `L z = v` in place when `v[..start]` is zero (so `z[..start]` is too).
fn solve_lower_from(l: &[f64], n: usize, v: &mut [f64], start: usize) {
    for i in start..n {
        let s: f64 = (start..i).map(|k| l[i * n + k] * v[k]).sum();
        v[i] = (v[i] - s) / l[i * n + i];
    }
}

/// Solves `Lᵀ x = v` in place.
fn solve_upper(l: &[f64], n: usize, v: &mut [f64]) {
    for i in (0..n).rev() {
        let s: f64 = (i + 1..n).map(|k| l[k * n + i] * v[k]).sum();
        v[i] = (v[i] - s) / l[i * n + i];
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::BezierFn;

    fn grid2(steps: usize) -> impl Iterator<Item = [f64; 2]> {
        (0..=steps).flat_map(move |i| {
            (0..=steps).map(move |j| [i as f64 / steps as f64, j as f64 / steps as f64])
        })
    }

    #[test]
    fn reproduces_cubic_exactly() {
        let f = |x: f64| 2.0 * x * x * x - x + 0.25;
        let mut fit = Fitter::new(3, &[-1.0], &[2.0]).with_smoothing(0.0);
        for i in 0..=50 {
            let x = -1.0 + 3.0 * i as f64 / 50.0;
            fit.add_sample([x], f(x)).unwrap();
        }
        let r = fit.solve().unwrap();
        assert!(r.rms_residual < 1e-10, "{r:?}");
        let pts = (0..=33).map(|i| -1.0 + 3.0 * i as f64 / 33.0);
        let e = fit.evaluate(pts.map(|x| ([x], f(x)))).unwrap();
        assert!(e.max_abs < 1e-9, "{e:?}");
    }

    #[test]
    fn reproduces_2d_polynomial() {
        let f = |p: [f64; 2]| p[0] * p[1] + 0.5 * p[0] * p[0] - p[1];
        let mut fit = Fitter::unit(2, 2).with_smoothing(0.0);
        for p in grid2(10) {
            fit.add_sample(p, f(p)).unwrap();
        }
        fit.solve().unwrap();
        let e = fit.evaluate(grid2(7).map(|p| (p, f(p)))).unwrap();
        assert!(e.max_abs < 1e-9, "{e:?}");
    }

    #[test]
    fn const_fn_matches_fitter() {
        let f = |p: [f64; 2]| (3.0 * p[0]).sin() * (1.0 + p[1] * p[1]);
        let mut fit = Fitter::new(4, &[0.0, -1.0], &[1.0, 1.0]);
        for p in grid2(12) {
            let q = [p[0], 2.0 * p[1] - 1.0];
            fit.add_sample(q, f(q)).unwrap();
        }
        fit.solve().unwrap();
        let bf = BezierFn::<2, 5>::new([0.0, -1.0], [1.0, 1.0], 1, fit.coefficients().unwrap());
        for p in [[0.1, -0.9], [0.5, 0.0], [0.93, 0.77], [1.5, 3.0]] {
            assert!((bf.eval(p) - fit.predict(p).unwrap()).abs() < 1e-12);
        }
    }

    #[test]
    fn exported_floats_round_trip() {
        let mut fit = Fitter::unit(1, 6);
        for i in 0..=40 {
            let x = i as f64 / 40.0;
            fit.add_sample([x], (5.0 * x).cos()).unwrap();
        }
        fit.solve().unwrap();
        let code = fit.to_rust_code("COS5").unwrap();
        let body = code.split("&[").nth(1).unwrap().split("],").next().unwrap();
        let parsed: Vec<f64> = body
            .split([',', '\n'])
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(|s| s.parse().unwrap())
            .collect();
        assert_eq!(parsed, fit.coefficients().unwrap());
    }

    #[test]
    fn rejects_out_of_range_and_wrong_dims() {
        let mut fit = Fitter::unit(2, 2);
        assert!(matches!(
            fit.add_sample([1.5, 0.5], 0.0),
            Err(FitError::OutOfRange { axis: 0, .. })
        ));
        assert!(matches!(
            fit.add_sample([0.5], 0.0),
            Err(FitError::WrongDimension { .. })
        ));
        assert_eq!(fit.predict([0.5, 0.5]), Err(FitError::NotSolved));
    }

    #[test]
    fn underdetermined_still_solves() {
        // 3 samples, 25 coefficients: smoothing fills in the rest.
        let mut fit = Fitter::unit(2, 4);
        fit.add_sample([0.1, 0.1], 1.0).unwrap();
        fit.add_sample([0.9, 0.2], 2.0).unwrap();
        fit.add_sample([0.5, 0.8], 3.0).unwrap();
        let r = fit.solve().unwrap();
        assert!(r.sigma.is_none());
        assert!((fit.predict([0.5, 0.8]).unwrap() - 3.0).abs() < 1e-3);
        assert!(fit.predict([0.5, 0.5]).unwrap().is_finite());
    }

    #[test]
    fn std_error_is_larger_where_data_is_missing() {
        let mut fit = Fitter::unit(1, 5);
        let mut seed = 1u64;
        for i in 0..200 {
            let x = 0.6 * i as f64 / 199.0; // data only in [0, 0.6]
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let noise = ((seed >> 11) as f64 / (1u64 << 53) as f64 - 0.5) * 0.02;
            fit.add_sample([x], x.sin() + noise).unwrap();
        }
        fit.solve().unwrap();
        let inside = fit.std_error([0.3]).unwrap();
        let outside = fit.std_error([0.95]).unwrap();
        assert!(outside > 5.0 * inside, "inside {inside}, outside {outside}");
    }

    /// Deterministic noise in [-0.5, 0.5).
    fn lcg(seed: &mut u64) -> f64 {
        *seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (*seed >> 11) as f64 / (1u64 << 53) as f64 - 0.5
    }

    /// 200 samples of a cubic with small noise; every 10th sample is a gross outlier.
    fn cubic_with_outliers(fit: &mut Fitter) -> (impl Fn(f64) -> f64, Vec<usize>) {
        let f = |x: f64| 2.0 * x * x * x - x + 0.25;
        let mut seed = 7;
        let mut bad = Vec::new();
        for i in 0..200 {
            let x = (i as f64 + 0.5) / 200.0;
            let mut y = f(x) + 0.002 * lcg(&mut seed);
            if i % 10 == 3 {
                y += 2.0 + lcg(&mut seed);
                bad.push(i);
            }
            fit.add_sample([x], y).unwrap();
        }
        (f, bad)
    }

    fn max_err(fit: &Fitter, f: impl Fn(f64) -> f64) -> f64 {
        let pts = (0..=100).map(|i| i as f64 / 100.0);
        fit.evaluate(pts.map(|x| ([x], f(x)))).unwrap().max_abs
    }

    #[test]
    fn weight_two_equals_duplicate_sample() {
        let mut a = Fitter::unit(1, 3);
        let mut b = Fitter::unit(1, 3);
        for (x, y) in [(0.1, 1.0), (0.4, 0.2), (0.7, 0.9), (0.9, 0.3)] {
            a.add_weighted_sample([x], y, 2.0).unwrap();
            b.add_sample([x], y).unwrap();
            b.add_sample([x], y).unwrap();
        }
        a.add_sample([0.5], 5.0).unwrap();
        b.add_sample([0.5], 5.0).unwrap();
        a.solve().unwrap();
        b.solve().unwrap();
        for (p, q) in a
            .coefficients()
            .unwrap()
            .iter()
            .zip(b.coefficients().unwrap())
        {
            assert!((p - q).abs() < 1e-9);
        }
        assert_eq!(
            a.add_weighted_sample([0.5], 1.0, 0.0),
            Err(FitError::InvalidWeight)
        );
    }

    #[test]
    fn tukey_ignores_gross_outliers() {
        let mut fit = Fitter::unit(1, 3);
        let (f, bad) = cubic_with_outliers(&mut fit);
        fit.solve().unwrap();
        let ls_err = max_err(&fit, &f);
        let r = fit.solve_robust(RobustLoss::TUKEY).unwrap();
        let tukey_err = max_err(&fit, &f);
        assert!(r.converged, "{r:?}");
        assert!(
            ls_err > 0.1,
            "least squares should be visibly pulled: {ls_err}"
        );
        assert!(tukey_err < 2e-3, "tukey error {tukey_err}");
        // Exactly the injected outliers are rejected.
        let w = fit.robust_weights().unwrap();
        let rejected: Vec<usize> = (0..w.len()).filter(|&i| w[i] == 0.0).collect();
        assert_eq!(rejected, bad);
        assert_eq!(fit.outliers(3.0).unwrap(), bad);
        // Noise estimate reflects the inlier noise (uniform ±0.001 → σ ≈ 0.00058).
        assert!(r.scale < 0.001, "{r:?}");
    }

    #[test]
    fn huber_limits_outlier_influence() {
        let mut fit = Fitter::unit(1, 3);
        let (f, _) = cubic_with_outliers(&mut fit);
        fit.solve().unwrap();
        let ls_err = max_err(&fit, &f);
        let r = fit.solve_robust(RobustLoss::HUBER).unwrap();
        let huber_err = max_err(&fit, &f);
        assert!(r.converged, "{r:?}");
        assert_eq!(r.rejected, 0);
        assert!(huber_err < 0.1 * ls_err, "ls {ls_err}, huber {huber_err}");
    }

    #[test]
    fn robust_agrees_with_least_squares_on_clean_data() {
        let f = |x: f64| (3.0 * x).sin();
        let mut fit = Fitter::unit(1, 5);
        let mut seed = 11;
        for i in 0..300 {
            let x = i as f64 / 299.0;
            fit.add_sample([x], f(x) + 0.01 * lcg(&mut seed)).unwrap();
        }
        fit.solve().unwrap();
        let ls = fit.predict([0.37]).unwrap();
        fit.solve_robust(RobustLoss::TUKEY).unwrap();
        let robust = fit.predict([0.37]).unwrap();
        assert!((ls - robust).abs() < 0.002, "ls {ls}, robust {robust}");
        // A plain solve afterwards returns to least squares.
        fit.solve().unwrap();
        assert!(fit.robust_weights().is_none());
        assert_eq!(fit.predict([0.37]).unwrap(), ls);
    }

    #[test]
    fn robust_needs_stored_samples() {
        let mut fit = Fitter::unit(1, 2).without_sample_storage();
        fit.add_sample([0.5], 1.0).unwrap();
        assert!(fit.solve().is_ok());
        assert_eq!(
            fit.solve_robust(RobustLoss::HUBER),
            Err(FitError::NoStoredSamples)
        );
    }

    #[test]
    fn support_reveals_lone_sample() {
        let mut fit = Fitter::unit(1, 6);
        for i in 0..200 {
            fit.add_sample([0.8 * i as f64 / 199.0], 0.0).unwrap();
        }
        fit.add_sample([0.97], 1.0).unwrap();
        fit.solve_robust(RobustLoss::TUKEY).unwrap();
        let dense = fit.support([0.4]).unwrap();
        let lone = fit.support([0.97]).unwrap();
        assert!(dense > 10.0, "dense {dense}");
        assert!(lone < 1.5, "lone {lone}");
    }

    #[test]
    fn one_segment_spline_is_bezier() {
        let mut a = Fitter::unit(2, 3);
        let mut b = Fitter::unit_spline(2, 3, 1);
        for p in grid2(9) {
            let y = (2.0 * p[0]).sin() + p[1] * p[1];
            a.add_sample(p, y).unwrap();
            b.add_sample(p, y).unwrap();
        }
        a.solve().unwrap();
        b.solve().unwrap();
        assert_eq!(a.coefficients(), b.coefficients());
    }

    #[test]
    fn cubic_spline_reproduces_cubic_and_linear_spline_reproduces_kinks() {
        let mut fit = Fitter::unit_spline(1, 3, 5).with_smoothing(0.0);
        let f = |x: f64| 2.0 * x * x * x - x + 0.25;
        for i in 0..=60 {
            let x = i as f64 / 60.0;
            fit.add_sample([x], f(x)).unwrap();
        }
        fit.solve().unwrap();
        let e = fit
            .evaluate((0..=97).map(|i| i as f64 / 97.0).map(|x| ([x], f(x))))
            .unwrap();
        assert!(e.max_abs < 1e-9, "{e:?}");

        // Piecewise linear with kinks exactly at the knots 0.25, 0.5, 0.75.
        let g = |x: f64| (x - 0.5).abs() + (x - 0.25).max(0.0);
        let mut fit = Fitter::unit_spline(1, 1, 4).with_smoothing(0.0);
        for i in 0..=40 {
            let x = i as f64 / 40.0;
            fit.add_sample([x], g(x)).unwrap();
        }
        fit.solve().unwrap();
        let e = fit
            .evaluate((0..=97).map(|i| i as f64 / 97.0).map(|x| ([x], g(x))))
            .unwrap();
        assert!(e.max_abs < 1e-9, "{e:?}");
    }

    #[test]
    fn spline_const_fn_matches_fitter_and_round_trips() {
        let f = |p: [f64; 2]| (5.0 * p[0]).sin() * (1.0 + (3.0 * p[1]).cos());
        let mut fit = Fitter::spline(2, 4, &[0.0, -1.0], &[1.0, 1.0]);
        for p in grid2(20) {
            let q = [p[0], 2.0 * p[1] - 1.0];
            fit.add_sample(q, f(q)).unwrap();
        }
        fit.solve().unwrap();
        assert_eq!(fit.num_coefficients(), 36);
        let bf = BezierFn::<2, 3>::new([0.0, -1.0], [1.0, 1.0], 4, fit.coefficients().unwrap());
        for p in [
            [0.1, -0.9],
            [0.5, 0.0],
            [0.93, 0.77],
            [0.25, 0.5],
            [1.5, 3.0],
            [-1.0, -2.0],
        ] {
            assert!(
                (bf.eval(p) - fit.predict(p).unwrap()).abs() < 1e-12,
                "{p:?}"
            );
        }
        let code = fit.to_rust_code("S").unwrap();
        assert!(code.contains("    4, // segments"), "{code}");
        assert!(code.contains("BezierFn<'static, 2, 3>"), "{code}");
    }

    fn assert_close(a: f64, b: f64, what: &str) {
        assert!(
            (a - b).abs() <= 1e-12 * a.abs().max(b.abs()),
            "{what}: {a} vs {b}"
        );
    }

    #[test]
    fn scaling_all_weights_changes_nothing() {
        let fit_with = |w: f64| {
            let mut fit = Fitter::unit(1, 3);
            let _ = cubic_with_outliers(&mut fit);
            // Re-add with weights: same samples, every weight multiplied by `w`.
            let mut scaled = Fitter::unit(1, 3);
            for i in 0..fit.num_samples() {
                let (x, y, _) = fit.sample(i).unwrap();
                scaled.add_weighted_sample(x, y, w).unwrap();
            }
            scaled
        };
        let (mut a, mut b) = (fit_with(1.0), fit_with(100.0));
        let (ra, rb) = (a.solve().unwrap(), b.solve().unwrap());
        assert_close(ra.rms_residual, rb.rms_residual, "rms_residual");
        assert_close(ra.sigma.unwrap(), rb.sigma.unwrap(), "sigma");
        assert_close(
            a.support([0.5]).unwrap(),
            b.support([0.5]).unwrap(),
            "support",
        );
        assert_close(
            a.std_error([0.5]).unwrap(),
            b.std_error([0.5]).unwrap(),
            "std_error",
        );
        assert_eq!(a.outliers(3.0).unwrap(), b.outliers(3.0).unwrap());
        let (ra, rb) = (
            a.solve_robust(RobustLoss::TUKEY).unwrap(),
            b.solve_robust(RobustLoss::TUKEY).unwrap(),
        );
        assert_close(ra.scale, rb.scale, "robust scale");
        assert_close(ra.rms_residual, rb.rms_residual, "robust rms_residual");
        assert_eq!(a.outliers(3.0).unwrap(), b.outliers(3.0).unwrap());
    }

    #[test]
    fn residuals_are_in_units_of_y_with_mixed_weights() {
        // A constant fit to ±1, both with weight 3: the residuals are ±1, so the
        // weighted RMS sqrt(Σ w r² / Σ w) is 1 (dividing by the count would give √3).
        let mut fit = Fitter::unit(1, 0).without_sample_storage();
        fit.add_weighted_sample([0.2], 1.0, 3.0).unwrap();
        fit.add_weighted_sample([0.8], -1.0, 3.0).unwrap();
        let r = fit.solve().unwrap();
        assert!((r.rms_residual - 1.0).abs() < 1e-6, "{r:?}");
    }

    #[test]
    fn errors_before_solve_and_on_empty_data() {
        let mut fit = Fitter::unit(1, 2);
        assert_eq!(fit.std_error([0.5]), Err(FitError::NotSolved));
        fit.add_sample([0.5], 1.0).unwrap();
        fit.solve().unwrap();
        assert_eq!(
            fit.evaluate(std::iter::empty::<([f64; 1], f64)>()),
            Err(FitError::NoData)
        );
    }

    #[test]
    fn huge_models_saturate_or_panic_instead_of_overflowing() {
        assert_eq!(Model::bezier(1).coefficients(64), usize::MAX);
        assert_eq!(Model::bezier(1).terms(64), usize::MAX);
        let r = std::panic::catch_unwind(|| Fitter::unit(64, 1));
        let msg = r.unwrap_err();
        let msg = msg.downcast_ref::<String>().unwrap();
        assert!(msg.contains("model too large"), "{msg}");
    }

    #[test]
    fn smoothing_does_not_bias_linear_models() {
        let mut fit = Fitter::unit(1, 1).with_smoothing(1.0);
        for i in 0..=10 {
            let x = i as f64 / 10.0;
            fit.add_sample([x], 2.0 * x).unwrap();
        }
        fit.solve().unwrap();
        assert!(fit.predict([0.0]).unwrap().abs() < 1e-9);
        assert!((fit.predict([1.0]).unwrap() - 2.0).abs() < 1e-9);
    }

    #[test]
    fn spline_robust_and_support_work() {
        let mut fit = Fitter::unit_spline(1, 3, 6);
        let (f, bad) = cubic_with_outliers(&mut fit);
        let r = fit.solve_robust(RobustLoss::TUKEY).unwrap();
        assert!(r.converged);
        assert!(max_err(&fit, &f) < 3e-3);
        assert_eq!(fit.outliers(3.0).unwrap(), bad);
        assert!(fit.support([0.5]).unwrap() > 10.0);
    }

    #[test]
    #[should_panic(expected = "finite width")]
    fn rejects_box_with_overflowing_width() {
        // Both bounds are finite, but hi - lo is not: every x would map to t = 0.
        let _ = Fitter::new(1, &[-1e308], &[1e308]);
    }

    #[test]
    fn sample_index_out_of_range_is_none() {
        let mut fit = Fitter::unit(2, 1);
        fit.add_weighted_sample([0.25, 0.5], 1.0, 2.0).unwrap();
        assert_eq!(fit.sample(0), Some((&[0.25, 0.5][..], 1.0, 2.0)));
        assert_eq!(fit.sample(1), None);
        assert_eq!(fit.sample(usize::MAX), None);
    }

    #[test]
    fn queries_reflect_the_last_solve_after_adding_samples() {
        let mut fit = Fitter::unit(1, 3);
        let (_, bad) = cubic_with_outliers(&mut fit);
        fit.solve_robust(RobustLoss::TUKEY).unwrap();
        let (support, se) = (fit.support([0.5]).unwrap(), fit.std_error([0.5]).unwrap());
        // Heavy samples on the fit change the mean weight, the unit of all of these.
        let y = fit.predict([0.5]).unwrap();
        for _ in 0..50 {
            fit.add_weighted_sample([0.5], y, 10.0).unwrap();
        }
        assert_eq!(fit.support([0.5]).unwrap(), support);
        assert_eq!(fit.std_error([0.5]).unwrap(), se);
        assert_eq!(fit.robust_weights().unwrap().len(), 200);
        assert_eq!(fit.outliers(3.0).unwrap(), bad);
    }

    #[test]
    fn debug_output_leaves_out_the_normal_equations() {
        let s = format!("{:?}", Fitter::unit(2, 9));
        assert!(s.starts_with("Fitter {") && s.len() < 300, "{s}");
    }

    #[test]
    fn to_rust_code_rejects_names_that_are_not_identifiers() {
        let mut fit = Fitter::unit(1, 1);
        fit.add_sample([0.0], 0.0).unwrap();
        fit.add_sample([1.0], 1.0).unwrap();
        fit.solve().unwrap();
        let bad = [
            "",
            "_",
            "1X",
            "MY FN",
            "X: u8 = 0; //",
            "fn",
            "Self",
            "gen",
            "Ä",
        ];
        for name in bad {
            assert_eq!(
                fit.to_rust_code(name),
                Err(FitError::InvalidName),
                "{name:?}"
            );
        }
        for good in ["F", "_F", "MY_FN_2", "lower"] {
            assert!(fit.to_rust_code(good).is_ok(), "{good:?}");
        }
    }
}
