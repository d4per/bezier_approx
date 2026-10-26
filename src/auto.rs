//! Automatic model selection: try Bézier polynomials of increasing degree and
//! B-splines of increasing size, score each by k-fold cross-validation, recommend
//! one, and explain the choice along with anything the user should watch out for.

use crate::fit::{assert_valid_box, FitError, Fitter, Model, RobustLoss};
use std::fmt::{self, Write};
use std::ops::Range;

/// Segment counts tried for each spline degree, smallest first.
const SEGMENT_LADDER: &[usize] = &[2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32, 48, 64];

/// Training samples for [`AutoFit`].
#[derive(Debug, Clone)]
pub struct Samples {
    dims: usize,
    xs: Vec<f64>,
    ys: Vec<f64>,
    ws: Vec<f64>,
}

impl Samples {
    /// An empty sample set with `dims` input coordinates per sample.
    ///
    /// # Panics
    /// If `dims == 0`.
    pub fn new(dims: usize) -> Self {
        assert!(dims > 0, "need at least one dimension");
        Samples {
            dims,
            xs: Vec::new(),
            ys: Vec::new(),
            ws: Vec::new(),
        }
    }

    /// Adds a sample with weight 1.
    pub fn push(&mut self, x: impl AsRef<[f64]>, y: f64) -> Result<(), FitError> {
        self.push_weighted(x, y, 1.0)
    }

    /// Adds a sample with a relative weight (see [`Fitter::add_weighted_sample`]).
    pub fn push_weighted(
        &mut self,
        x: impl AsRef<[f64]>,
        y: f64,
        weight: f64,
    ) -> Result<(), FitError> {
        let x = x.as_ref();
        if x.len() != self.dims {
            return Err(FitError::WrongDimension {
                expected: self.dims,
                got: x.len(),
            });
        }
        if !y.is_finite() || x.iter().any(|v| !v.is_finite()) {
            return Err(FitError::NotFinite);
        }
        if !(weight.is_finite() && weight > 0.0) {
            return Err(FitError::InvalidWeight);
        }
        self.xs.extend_from_slice(x);
        self.ys.push(y);
        self.ws.push(weight);
        Ok(())
    }

    /// Number of input coordinates per sample.
    pub fn dims(&self) -> usize {
        self.dims
    }

    /// Number of samples.
    pub fn len(&self) -> usize {
        self.ys.len()
    }

    /// Whether there are no samples.
    pub fn is_empty(&self) -> bool {
        self.ys.is_empty()
    }

    /// Sample `i` as `(x, y, weight)`.
    pub fn get(&self, i: usize) -> Option<(&[f64], f64, f64)> {
        let y = *self.ys.get(i)?;
        Some((&self.xs[i * self.dims..(i + 1) * self.dims], y, self.ws[i]))
    }
}

/// How [`AutoFit`] deals with outliers.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum RobustMode {
    /// Plain least squares. Outliers are detected and reported, not removed.
    Off,
    /// Check the data for outliers; if there are some (but fewer than a quarter of
    /// the samples), exclude them from model selection and fit with Tukey's loss.
    Auto,
    /// Always select and fit with this robust loss.
    Always(RobustLoss),
}

/// What stopped a family of candidates from growing further.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum Limit {
    /// [`AutoFit::max_bezier_degree`] was reached.
    MaxDegree,
    /// [`AutoFit::max_segments`] was reached.
    MaxSegments,
    /// [`AutoFit::max_coefficients`] was reached.
    MaxCoefficients,
    /// Larger models would have fewer than 2 training samples per coefficient.
    Data,
}

/// One model that was tried, with its cross-validated error.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct Candidate {
    /// Degree and segments.
    pub model: Model,
    /// Values to store: `(degree + segments)^dims`.
    pub coefficients: usize,
    /// Products per evaluation: `(degree + 1)^dims`.
    pub terms: usize,
    /// Cross-validated RMS error, weighted by the sample weights: each sample is
    /// predicted by a model trained without it. Outliers are excluded when they are
    /// being handled, unless too few samples would be left to score any model.
    pub cv_rms: f64,
    /// Standard error of `cv_rms`, from its spread across folds.
    pub cv_se: f64,
    /// Largest cross-validated error (outliers excluded like for `cv_rms`).
    pub cv_max: f64,
}

/// How the recommended model was picked.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum Rule {
    /// The smallest model whose CV error is within one standard error of the lowest
    /// CV error. Larger models only win if they are clearly better.
    OneStandardError {
        /// Lowest CV error plus its standard error.
        threshold: f64,
    },
    /// The smallest model whose CV error meets [`AutoFit::target_rms`] (the value).
    Target(f64),
}

/// Something about the data or the recommendation the user should know.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum Caveat {
    /// Few samples per coefficient: the fit can react strongly to single points.
    FewSamplesPerCoefficient {
        /// Samples used (outliers excluded).
        samples: usize,
        /// Coefficients of the recommended model.
        coefficients: usize,
    },
    /// The recommended model is the largest of its kind that was tried.
    AtLimit {
        /// The recommended model.
        model: Model,
        /// What prevented trying a larger one.
        limit: Limit,
    },
    /// Samples that disagree strongly with the model.
    Outliers {
        /// Indices into the samples, in the order they were added.
        indices: Vec<usize>,
        /// Whether the robust fit ignored them.
        handled: bool,
    },
    /// Far more large errors than random noise would produce, so they were not
    /// treated as outliers. Usually the error is concentrated where the function has
    /// a feature the models cannot follow.
    UnevenErrors {
        /// Fraction of samples with large errors.
        fraction: f64,
        /// The input with the largest error.
        worst_at: Vec<f64>,
    },
    /// A region where the prediction rests on less than one sample's worth of data.
    SparseRegion {
        /// [`Fitter::support`] at `at`.
        support: f64,
        /// The input with the least support.
        at: Vec<f64>,
    },
    /// The cross-validated error varies a lot between folds.
    UnstableEstimate {
        /// Standard error of the CV error relative to the CV error.
        relative: f64,
    },
    /// No candidate met the requested target.
    TargetNotMet {
        /// The requested [`AutoFit::target_rms`].
        target: f64,
        /// The lowest CV error of any candidate.
        best: f64,
    },
    /// Tensor-product models get large quickly in many dimensions.
    HighDimensional {
        /// Number of input dimensions.
        dims: usize,
    },
}

impl fmt::Display for Caveat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Caveat::FewSamplesPerCoefficient {
                samples,
                coefficients,
            } => write!(
                f,
                "Only {:.1} samples per coefficient ({samples} samples, {coefficients} \
                 coefficients). The fit can react strongly to individual points and the \
                 error estimates are rough; more data would make it more trustworthy.",
                *samples as f64 / *coefficients as f64
            ),
            Caveat::AtLimit { model, limit } => match limit {
                Limit::Data => write!(
                    f,
                    "{model} is the largest model of its kind the amount of data allows \
                     (at least 2 training samples per coefficient). A more flexible model \
                     might fit better, but more samples are needed to try one."
                ),
                _ => {
                    let what = match limit {
                        Limit::MaxDegree => "the highest degree allowed (max_bezier_degree)",
                        Limit::MaxSegments => "the most segments allowed (max_segments)",
                        _ => "the coefficient budget (max_coefficients)",
                    };
                    write!(
                        f,
                        "{model} is the most flexible model of its kind that was tried: it \
                         hit {what}, and the error was still falling. Raising that limit \
                         might give a better fit. If your data is exact (computed from a \
                         formula), the error usually keeps falling with model size; then set \
                         target_rms to get the smallest model that is accurate enough."
                    )
                }
            },
            Caveat::Outliers { indices, handled } => {
                let shown: Vec<String> = indices.iter().take(5).map(|i| format!("#{i}")).collect();
                let more = if indices.len() > 5 { ", …" } else { "" };
                if *handled {
                    write!(
                        f,
                        "{} samples look like outliers ({}{more}) and were ignored by the robust \
                         fit. If your data should not contain glitches, check these points: \
                         they may be real features the model cannot follow.",
                        indices.len(),
                        shown.join(", ")
                    )
                } else {
                    write!(
                        f,
                        "{} samples look like outliers ({}{more}) but were kept, because robust \
                         fitting is off. Consider RobustMode::Auto.",
                        indices.len(),
                        shown.join(", ")
                    )
                }
            }
            Caveat::UnevenErrors { fraction, worst_at } => write!(
                f,
                "{:.0} % of samples have errors far above the typical error, which random \
                 noise would not produce. Most likely the error is concentrated where the \
                 function has a feature the model cannot follow well (a sharp bend, a very \
                 steep slope, a jump); the largest error is near {worst_at:?}. More samples \
                 or segments there, or splitting the domain, may help. (Heavily contaminated \
                 data can look the same; these samples were not treated as outliers.)",
                100.0 * fraction
            ),
            Caveat::SparseRegion { support, at } => write!(
                f,
                "Little or no data near {at:?}: the prediction there rests on about \
                 {support:.2} samples' worth of data, so it is mostly a guess carried over \
                 from neighbouring regions. Add samples there or don't rely on that region."
            ),
            Caveat::UnstableEstimate { relative } => write!(
                f,
                "The cross-validated error varies a lot between folds (standard error \
                 {:.0} % of the estimate). More data would make the choice and the error \
                 estimate more reliable.",
                100.0 * relative
            ),
            Caveat::TargetNotMet { target, best } => write!(
                f,
                "No model reached the target error {target:.2e}; the best was {best:.2e}. \
                 The recommendation falls back to the one-standard-error rule. More data, \
                 less noise, or higher limits may be needed."
            ),
            Caveat::HighDimensional { dims } => write!(
                f,
                "In {dims} dimensions the number of coefficients grows as (values per \
                 axis)^{dims}, so only coarse models fit within the limits."
            ),
        }
    }
}

/// The outcome of [`AutoFit::run`].
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct AutoReport {
    /// Every model that was tried, in the order tried.
    pub candidates: Vec<Candidate>,
    /// Index of the recommended candidate.
    pub chosen: usize,
    /// Index of the candidate with the lowest cross-validated error.
    pub lowest: usize,
    /// How the recommendation was made.
    pub rule: Rule,
    /// The robust loss used, if outliers were handled. They are then also excluded
    /// from the cross-validation scores, unless too few samples would be left to
    /// score any model; [`summary`](AutoReport::summary) says which.
    pub robust: Option<RobustLoss>,
    /// Things to watch out for, most important first.
    pub caveats: Vec<Caveat>,
    /// The recommended model, fitted to all samples. Export it with
    /// [`Fitter::to_rust_code`].
    pub fitter: Fitter,
    samples: usize,
    folds: usize,
    /// Whether outliers were excluded from the cross-validation scores.
    robust_scoring: bool,
}

impl AutoReport {
    /// The recommended candidate, `&candidates[chosen]`.
    pub fn chosen(&self) -> &Candidate {
        &self.candidates[self.chosen]
    }

    /// A readable report: all candidates, the recommendation and why, and caveats.
    pub fn summary(&self) -> String {
        let mut s = String::new();
        let how = match self.robust {
            None => "least squares".to_string(),
            Some(loss) => {
                let name = match loss {
                    RobustLoss::Huber { .. } => "Huber",
                    RobustLoss::Tukey { .. } => "Tukey",
                };
                if self.robust_scoring {
                    format!("robust {name} fit, outliers excluded from scoring")
                } else {
                    format!("robust {name} fit; too few samples to exclude outliers from scoring")
                }
            }
        };
        let _ = writeln!(
            s,
            "Tried {} models on {} samples ({}-D) with {}-fold cross-validation ({how}).",
            self.candidates.len(),
            self.samples,
            self.fitter.dims(),
            self.folds
        );
        let _ = writeln!(
            s,
            "coeffs = values to store, terms = products per evaluation.\n"
        );
        let _ = writeln!(
            s,
            "  {:<30} {:>6} {:>6}   {:>20}   {:>9}",
            "model", "coeffs", "terms", "CV error (± SE)", "CV max"
        );
        for (i, c) in self.candidates.iter().enumerate() {
            let mark = if i == self.chosen {
                "  ← recommended"
            } else if i == self.lowest {
                "  (lowest CV error)"
            } else {
                ""
            };
            let _ = writeln!(
                s,
                "  {:<30} {:>6} {:>6}   {:>9.2e} ± {:<8.1e}   {:>9.2e}{mark}",
                c.model.to_string(),
                c.coefficients,
                c.terms,
                c.cv_rms,
                c.cv_se,
                c.cv_max
            );
        }
        let c = self.chosen();
        let _ = writeln!(
            s,
            "\nRecommendation: {} — {} coefficients, {} terms per evaluation.\n\
             Expected error on new inputs ≈ {:.2e} RMS; the largest cross-validated error was {:.2e}.",
            c.model, c.coefficients, c.terms, c.cv_rms, c.cv_max
        );
        let low = &self.candidates[self.lowest];
        match &self.rule {
            Rule::Target(t) => {
                let _ = writeln!(
                    s,
                    "It is the smallest model whose cross-validated error meets the target {t:.2e}."
                );
            }
            Rule::OneStandardError { threshold } => {
                if self.chosen == self.lowest {
                    let _ = writeln!(
                        s,
                        "It has the lowest cross-validated error, and no smaller model comes within \
                         one standard error of it."
                    );
                } else {
                    let _ = writeln!(
                        s,
                        "The lowest error, {:.2e}, came from {} with {} coefficients. The \
                         recommended model is within one standard error of that ({:.2e} ≤ {:.2e}) \
                         and smaller, so the extra size is not clearly paying off.",
                        low.cv_rms, low.model, low.coefficients, c.cv_rms, threshold
                    );
                }
            }
        }
        if let Some(sigma) = self.fitter.report().and_then(|r| r.sigma) {
            let _ = writeln!(
                s,
                "Typical residual of the final fit (measurement noise plus anything the \
                 model cannot follow): σ ≈ {sigma:.2e}."
            );
        }
        if self.caveats.is_empty() {
            let _ = writeln!(s, "\nNo caveats.");
        } else {
            let _ = writeln!(s, "\nCaveats:");
            for c in &self.caveats {
                let _ = writeln!(s, "  - {c}");
            }
        }
        s
    }
}

impl fmt::Display for AutoReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.summary())
    }
}

/// Tries Bézier polynomials and B-splines on a data set and recommends one.
///
/// ```no_run
/// # use bezier_approx::{AutoFit, Samples};
/// let mut data = Samples::new(1);
/// data.push([0.1], 0.2)?;
/// // ...
/// let report = AutoFit::unit(1).run(&data)?;
/// println!("{report}");
/// println!("{}", report.fitter.to_rust_code("MY_FN")?);
/// # Ok::<(), bezier_approx::FitError>(())
/// ```
///
/// Each candidate is scored by k-fold cross-validation: the samples are split into
/// `folds` groups, and each group is predicted by a model trained on the others. That
/// measures how well a model does on inputs it has not seen, which is what matters;
/// the error on the training data itself always improves with model size.
///
/// Within each family (Bézier; splines of each degree) models are tried from small to
/// large, and a family stops growing once three steps in a row fail to improve its
/// error by at least 1 %, or when a limit is hit.
#[derive(Debug, Clone)]
pub struct AutoFit {
    lo: Vec<f64>,
    hi: Vec<f64>,
    max_bezier_degree: usize,
    spline_degrees: Vec<usize>,
    max_segments: usize,
    max_coefficients: usize,
    folds: usize,
    smoothing: f64,
    robust: RobustMode,
    target_rms: Option<f64>,
    seed: u64,
}

impl AutoFit {
    /// Model selection on the box `[lo, hi]`.
    ///
    /// # Panics
    /// If `lo` and `hi` differ in length, are empty, or `lo[d] >= hi[d]` or the width
    /// `hi[d] - lo[d]` is not finite on some axis.
    pub fn new(lo: &[f64], hi: &[f64]) -> Self {
        assert_valid_box(lo, hi);
        AutoFit {
            lo: lo.to_vec(),
            hi: hi.to_vec(),
            max_bezier_degree: 10,
            spline_degrees: vec![1, 2, 3],
            max_segments: 32,
            max_coefficients: 400,
            folds: 5,
            smoothing: 1e-9,
            robust: RobustMode::Auto,
            target_rms: None,
            seed: 0x5eed,
        }
    }

    /// Model selection on the unit box `[0, 1]^dims`.
    ///
    /// # Panics
    /// If `dims == 0`.
    pub fn unit(dims: usize) -> Self {
        Self::new(&vec![0.0; dims], &vec![1.0; dims])
    }

    /// Highest Bézier degree to try (default 10; 0 skips Bézier models).
    pub fn max_bezier_degree(mut self, degree: usize) -> Self {
        self.max_bezier_degree = degree;
        self
    }

    /// Spline degrees to try (default 1, 2, 3; empty skips splines). Degree 1 is
    /// piecewise linear (a fitted lookup table), 3 is a smooth cubic spline.
    pub fn spline_degrees(mut self, degrees: &[usize]) -> Self {
        self.spline_degrees = degrees.to_vec();
        self
    }

    /// Most segments per axis to try for splines (default 32). Segment counts are
    /// tried along a ladder (2, 3, 4, 5, 6, 8, 10, 12, 16, …) that ends at this value.
    pub fn max_segments(mut self, segments: usize) -> Self {
        self.max_segments = segments;
        self
    }

    /// Largest number of coefficients to try (default 400). Training cost grows with
    /// the cube of this; the exported constant grows linearly.
    pub fn max_coefficients(mut self, n: usize) -> Self {
        self.max_coefficients = n;
        self
    }

    /// Number of cross-validation folds (default 5, at most the number of samples).
    ///
    /// # Panics
    /// If `folds < 2`.
    pub fn folds(mut self, folds: usize) -> Self {
        assert!(folds >= 2, "need at least 2 folds");
        self.folds = folds;
        self
    }

    /// Smoothness penalty passed to every fit (see [`Fitter::with_smoothing`]).
    ///
    /// # Panics
    /// If `smoothing` is negative or not finite.
    pub fn smoothing(mut self, smoothing: f64) -> Self {
        assert!(
            smoothing >= 0.0 && smoothing.is_finite(),
            "smoothing must be finite and non-negative"
        );
        self.smoothing = smoothing;
        self
    }

    /// Outlier handling (default [`RobustMode::Auto`]).
    pub fn robust(mut self, mode: RobustMode) -> Self {
        self.robust = mode;
        self
    }

    /// Recommend the smallest model whose cross-validated RMS error is at most
    /// `target`, instead of using the one-standard-error rule. Useful for exact data
    /// (computed from a formula) and for memory-constrained targets.
    ///
    /// # Panics
    /// If `target` is negative or NaN.
    pub fn target_rms(mut self, target: f64) -> Self {
        assert!(target >= 0.0, "target must be non-negative");
        self.target_rms = Some(target);
        self
    }

    /// Seed for the random assignment of samples to folds (default fixed, so runs
    /// are reproducible).
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Runs the selection, fits the recommended model to all samples and checks it.
    ///
    /// Fails with [`FitError::NoData`] if `data` is empty, and with
    /// [`FitError::NoCandidates`] if no model fits the limits and the data (every model
    /// needs at least 2 training samples per coefficient).
    #[allow(clippy::needless_range_loop)]
    pub fn run(&self, data: &Samples) -> Result<AutoReport, FitError> {
        let dims = self.lo.len();
        if data.dims() != dims {
            return Err(FitError::WrongDimension {
                expected: dims,
                got: data.dims(),
            });
        }
        let m = data.len();
        if m == 0 {
            return Err(FitError::NoData);
        }
        if m < 2 {
            return Err(FitError::NoCandidates);
        }
        for i in 0..m {
            let (x, _, _) = data.get(i).expect("index in range");
            for d in 0..dims {
                let t = (x[d] - self.lo[d]) / (self.hi[d] - self.lo[d]);
                if !(-1e-12..=1.0 + 1e-12).contains(&t) {
                    return Err(FitError::OutOfRange {
                        axis: d,
                        value: x[d],
                    });
                }
            }
        }
        let folds = self.folds.min(m);
        let fold_of = fold_assignment(m, folds, self.seed);
        let y_rms = (data.ys.iter().map(|y| y * y).sum::<f64>() / m as f64).sqrt();
        // CV errors below this are rounding noise and count as equal.
        let tol = 1e-9 * y_rms.max(f64::MIN_POSITIVE);
        let mut caveats = Vec::new();

        // 1. Select with plain least squares.
        let mut weights = data.ws.clone();
        let mut sel = self.select(data, &weights, &fold_of, folds, tol)?;

        // 2. Look for outliers with a robust fit of the chosen model.
        let loss = match self.robust {
            RobustMode::Always(loss) => loss,
            _ => RobustLoss::TUKEY,
        };
        let mut probe = self.new_fitter(sel.candidates[sel.chosen].model);
        feed(&mut probe, data, &weights, |_| true)?;
        probe.solve_robust(loss)?;
        let rw = probe.robust_weights().expect("robust fit done").to_vec();
        let rejected: Vec<usize> = (0..m).filter(|&i| rw[i] == 0.0).collect();
        let mut uneven = None;
        let use_robust = if matches!(self.robust, RobustMode::Always(_)) {
            true
        } else if rejected.is_empty() {
            false
        } else if rejected.len() * 4 > m {
            uneven = Some(rejected.len() as f64 / m as f64);
            false
        } else if self.robust == RobustMode::Auto {
            true
        } else {
            caveats.push(Caveat::Outliers {
                indices: rejected,
                handled: false,
            });
            false
        };
        let mut robust = None;
        let mut robust_scoring = false;
        if use_robust {
            // 3. Select again with the robust weights: outliers drop out of both
            //    training and scoring, and doubtful samples count for less.
            for (w, r) in weights.iter_mut().zip(&rw) {
                *w *= r;
            }
            match self.select(data, &weights, &fold_of, folds, tol) {
                Ok(s) => {
                    sel = s;
                    robust_scoring = true;
                }
                // Without the outliers no model has enough samples left to be scored:
                // keep the selection above, but still fit the chosen model robustly.
                Err(FitError::NoCandidates) => {}
                Err(e) => return Err(e),
            }
            robust = Some(loss);
        }

        // 4. Fit the recommended model to all samples.
        let chosen = sel.candidates[sel.chosen].clone();
        let mut fitter = self.new_fitter(chosen.model);
        feed(&mut fitter, data, &data.ws, |_| true)?;
        match robust {
            Some(loss) => {
                fitter.solve_robust(loss)?;
                let rw = fitter.robust_weights().expect("robust fit done");
                let rejected: Vec<usize> = (0..m).filter(|&i| rw[i] == 0.0).collect();
                if !rejected.is_empty() {
                    caveats.insert(
                        0,
                        Caveat::Outliers {
                            indices: rejected,
                            handled: true,
                        },
                    );
                }
            }
            None => {
                fitter.solve()?;
            }
        }

        // 5. Check the result.
        if let Some(fraction) = uneven {
            let mut worst = (-1.0, Vec::new());
            for i in 0..m {
                let (x, y, w) = data.get(i).expect("index in range");
                let e = w.sqrt() * (fitter.predict(x)? - y).abs();
                if e > worst.0 {
                    worst = (e, x.to_vec());
                }
            }
            caveats.push(Caveat::UnevenErrors {
                fraction,
                worst_at: worst.1,
            });
        }
        if let Some(target) = sel.target_missed {
            caveats.push(Caveat::TargetNotMet {
                target,
                best: sel.candidates[sel.lowest].cv_rms,
            });
        }
        let family = sel
            .families
            .iter()
            .find(|f| f.range.contains(&sel.chosen))
            .expect("chosen candidate belongs to a family");
        let met_target = matches!(sel.rule, Rule::Target(_));
        if let Stop::Limit(limit) = family.stop {
            if sel.chosen + 1 == family.range.end && !met_target {
                caveats.push(Caveat::AtLimit {
                    model: chosen.model,
                    limit,
                });
            }
        }
        let used = weights.iter().filter(|&&w| w > 0.0).count();
        if used < 5 * chosen.coefficients {
            caveats.push(Caveat::FewSamplesPerCoefficient {
                samples: used,
                coefficients: chosen.coefficients,
            });
        }
        if chosen.cv_rms > tol && chosen.cv_se > 0.3 * chosen.cv_rms {
            caveats.push(Caveat::UnstableEstimate {
                relative: chosen.cv_se / chosen.cv_rms,
            });
        }
        let (support, at) = self.weakest_support(&fitter)?;
        if support < 1.0 {
            caveats.push(Caveat::SparseRegion { support, at });
        }
        if dims >= 4 {
            caveats.push(Caveat::HighDimensional { dims });
        }

        Ok(AutoReport {
            candidates: sel.candidates,
            chosen: sel.chosen,
            lowest: sel.lowest,
            rule: sel.rule,
            robust,
            caveats,
            fitter,
            samples: m,
            folds,
            robust_scoring,
        })
    }

    fn new_fitter(&self, model: Model) -> Fitter {
        Fitter::with_model(model, &self.lo, &self.hi).with_smoothing(self.smoothing)
    }

    /// Cross-validates the candidate families and applies the selection rule.
    fn select(
        &self,
        data: &Samples,
        weights: &[f64],
        fold_of: &[usize],
        folds: usize,
        tol: f64,
    ) -> Result<Selection, FitError> {
        let dims = self.lo.len();
        let used: Vec<usize> = (0..data.len()).filter(|&i| weights[i] > 0.0).collect();
        let largest_fold = (0..folds)
            .map(|f| used.iter().filter(|&&i| fold_of[i] == f).count())
            .max()
            .unwrap_or(0);
        let max_by_data = (used.len() - largest_fold) / 2;

        let mut families_models = vec![(
            (1..=self.max_bezier_degree)
                .map(Model::bezier)
                .collect::<Vec<_>>(),
            Limit::MaxDegree,
        )];
        for &p in &self.spline_degrees {
            let ladder = segment_ladder(self.max_segments)
                .into_iter()
                .map(|k| Model::spline(p, k))
                .collect();
            families_models.push((ladder, Limit::MaxSegments));
        }

        let mut candidates = Vec::new();
        let mut families = Vec::new();
        for (models, end_limit) in families_models {
            let start = candidates.len();
            let mut stop = Stop::Limit(end_limit);
            let mut best = f64::INFINITY;
            let mut stale = 0;
            for model in models {
                let n = model.coefficients(dims);
                if n > self.max_coefficients {
                    stop = Stop::Limit(Limit::MaxCoefficients);
                    break;
                }
                if n > max_by_data {
                    stop = Stop::Limit(Limit::Data);
                    break;
                }
                let c = self.cross_validate(model, data, weights, fold_of, folds)?;
                if c.cv_rms < 0.99 * best && best > tol {
                    best = c.cv_rms;
                    stale = 0;
                } else {
                    stale += 1;
                }
                candidates.push(c);
                if stale >= 3 {
                    stop = Stop::Plateau;
                    break;
                }
            }
            families.push(Family {
                range: start..candidates.len(),
                stop,
            });
        }
        if candidates.is_empty() {
            return Err(FitError::NoCandidates);
        }

        let lowest = (0..candidates.len())
            .min_by(|&a, &b| {
                let (ca, cb) = (&candidates[a], &candidates[b]);
                ca.cv_rms
                    .total_cmp(&cb.cv_rms)
                    .then(ca.coefficients.cmp(&cb.coefficients))
            })
            .expect("nonempty");
        // Smallest model satisfying `ok`; among equal sizes, the more accurate one.
        let smallest = |ok: &dyn Fn(&Candidate) -> bool| {
            (0..candidates.len())
                .filter(|&i| ok(&candidates[i]))
                .min_by(|&a, &b| {
                    let (ca, cb) = (&candidates[a], &candidates[b]);
                    ca.coefficients
                        .cmp(&cb.coefficients)
                        .then(ca.cv_rms.total_cmp(&cb.cv_rms))
                })
        };
        let mut target_missed = None;
        if let Some(target) = self.target_rms {
            if let Some(chosen) = smallest(&|c| c.cv_rms <= target) {
                return Ok(Selection {
                    candidates,
                    families,
                    chosen,
                    lowest,
                    rule: Rule::Target(target),
                    target_missed,
                });
            }
            target_missed = Some(target);
        }
        let threshold = candidates[lowest].cv_rms + candidates[lowest].cv_se.max(tol);
        let chosen = smallest(&|c| c.cv_rms <= threshold).expect("lowest qualifies");
        Ok(Selection {
            candidates,
            families,
            chosen,
            lowest,
            rule: Rule::OneStandardError { threshold },
            target_missed,
        })
    }

    /// k-fold cross-validated error of one model.
    fn cross_validate(
        &self,
        model: Model,
        data: &Samples,
        weights: &[f64],
        fold_of: &[usize],
        folds: usize,
    ) -> Result<Candidate, FitError> {
        // Weighted RMS, sqrt(Σ w e² / Σ w), so it is in units of y whatever the
        // scale of the weights.
        let mut sq = 0.0;
        let mut wsum = 0.0;
        let mut max = 0.0f64;
        let mut fold_rms = Vec::with_capacity(folds);
        for f in 0..folds {
            let mut fit = self.new_fitter(model).without_sample_storage();
            feed(&mut fit, data, weights, |i| fold_of[i] != f)?;
            fit.solve()?;
            let (mut fsq, mut fw) = (0.0, 0.0);
            for i in (0..data.len()).filter(|&i| fold_of[i] == f && weights[i] > 0.0) {
                let (x, y, _) = data.get(i).expect("index in range");
                let e = fit.predict(x)? - y;
                fsq += weights[i] * e * e;
                fw += weights[i];
                max = max.max(e.abs());
            }
            if fw > 0.0 {
                fold_rms.push((fsq / fw).sqrt());
                sq += fsq;
                wsum += fw;
            }
        }
        let cv_rms = if wsum > 0.0 { (sq / wsum).sqrt() } else { 0.0 };
        let k = fold_rms.len() as f64;
        let mean = fold_rms.iter().sum::<f64>() / k;
        let var = fold_rms
            .iter()
            .map(|r| (r - mean) * (r - mean))
            .sum::<f64>()
            / (k - 1.0).max(1.0);
        let dims = self.lo.len();
        Ok(Candidate {
            model,
            coefficients: model.coefficients(dims),
            terms: model.terms(dims),
            cv_rms,
            cv_se: (var / k).sqrt(),
            cv_max: max,
        })
    }

    /// The lowest [`Fitter::support`] over a grid covering the box, and where.
    fn weakest_support(&self, fitter: &Fitter) -> Result<(f64, Vec<f64>), FitError> {
        let dims = self.lo.len();
        let mut points: Vec<Vec<f64>> = Vec::new();
        if dims <= 3 {
            let per_axis: usize = [101, 21, 9][dims - 1];
            let total = per_axis.pow(dims as u32);
            for mut k in 0..total {
                let mut t = Vec::with_capacity(dims);
                for _ in 0..dims {
                    t.push((k % per_axis) as f64 / (per_axis - 1) as f64);
                    k /= per_axis;
                }
                points.push(t);
            }
        } else {
            let mut state = self.seed ^ 0x9e37_79b9_7f4a_7c15;
            for _ in 0..400 {
                points.push((0..dims).map(|_| unit_random(&mut state)).collect());
            }
        }
        let mut worst = (f64::INFINITY, Vec::new());
        for t in points {
            let x: Vec<f64> = (0..dims)
                .map(|d| self.lo[d] + t[d] * (self.hi[d] - self.lo[d]))
                .collect();
            let s = fitter.support(&x)?;
            if s < worst.0 {
                worst = (s, x);
            }
        }
        Ok(worst)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Stop {
    Plateau,
    Limit(Limit),
}

#[derive(Debug)]
struct Family {
    range: Range<usize>,
    stop: Stop,
}

struct Selection {
    candidates: Vec<Candidate>,
    families: Vec<Family>,
    chosen: usize,
    lowest: usize,
    rule: Rule,
    target_missed: Option<f64>,
}

/// Adds the samples with positive weight for which `include(i)` holds.
#[allow(clippy::needless_range_loop)]
fn feed(
    fit: &mut Fitter,
    data: &Samples,
    weights: &[f64],
    include: impl Fn(usize) -> bool,
) -> Result<(), FitError> {
    for i in 0..data.len() {
        if weights[i] > 0.0 && include(i) {
            let (x, y, _) = data.get(i).expect("index in range");
            fit.add_weighted_sample(x, y, weights[i])?;
        }
    }
    Ok(())
}

/// Segment counts to try, smallest first: [`SEGMENT_LADDER`] up to `max`, ending at
/// `max` itself so that hitting the end really means `max_segments` was reached.
fn segment_ladder(max: usize) -> Vec<usize> {
    let mut ladder: Vec<usize> = SEGMENT_LADDER
        .iter()
        .copied()
        .filter(|&k| k <= max)
        .collect();
    if max >= 2 && ladder.last() != Some(&max) {
        ladder.push(max);
    }
    ladder
}

fn splitmix(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

fn unit_random(state: &mut u64) -> f64 {
    (splitmix(state) >> 11) as f64 / (1u64 << 53) as f64
}

/// Random but reproducible assignment of samples to folds of (almost) equal size.
fn fold_assignment(m: usize, folds: usize, seed: u64) -> Vec<usize> {
    let mut order: Vec<usize> = (0..m).collect();
    let mut state = seed;
    for i in (1..m).rev() {
        let j = (splitmix(&mut state) % (i as u64 + 1)) as usize;
        order.swap(i, j);
    }
    let mut fold = vec![0; m];
    for (pos, &i) in order.iter().enumerate() {
        fold[i] = pos % folds;
    }
    fold
}

#[cfg(test)]
mod tests {
    use super::*;

    fn noise(state: &mut u64) -> f64 {
        unit_random(state) - 0.5
    }

    #[test]
    fn exact_cubic_picks_bezier_degree_3() {
        let mut data = Samples::new(1);
        for i in 0..80 {
            let x = i as f64 / 79.0;
            data.push([x], 2.0 * x * x * x - x + 0.25).unwrap();
        }
        // Exact data: no smoothing bias, so the true model is found exactly.
        let r = AutoFit::unit(1).smoothing(0.0).run(&data).unwrap();
        assert_eq!(r.chosen().model, Model::bezier(3), "{}", r.summary());
        assert!(r.chosen().cv_rms < 1e-9);
        assert!(r.robust.is_none(), "{}", r.summary());
    }

    #[test]
    fn kinked_function_picks_a_spline() {
        // A soft kink: flat, then rising. Hard for one polynomial, easy for splines.
        let f = |x: f64| (1.0 + (40.0 * (x - 0.4)).exp()).ln() / 40.0;
        let mut data = Samples::new(1);
        let mut st = 3;
        for _ in 0..300 {
            let x = unit_random(&mut st);
            data.push([x], f(x) + 0.001 * noise(&mut st)).unwrap();
        }
        let r = AutoFit::unit(1).run(&data).unwrap();
        assert!(r.chosen().model.segments > 1, "{}", r.summary());
        assert!(r.chosen().cv_rms < 0.002, "{}", r.summary());
    }

    #[test]
    fn tiny_data_gets_linear_model_and_caveats() {
        let mut data = Samples::new(1);
        for (x, y) in [
            (0.1, 0.2),
            (0.15, 0.25),
            (0.3, 0.44),
            (0.2, 0.31),
            (0.25, 0.37),
        ] {
            data.push([x], y).unwrap();
        }
        let r = AutoFit::unit(1).run(&data).unwrap();
        assert_eq!(r.chosen().model, Model::bezier(1));
        let has = |pred: fn(&Caveat) -> bool| r.caveats.iter().any(pred);
        assert!(
            has(|c| matches!(c, Caveat::FewSamplesPerCoefficient { .. })),
            "{}",
            r.summary()
        );
        assert!(
            has(|c| matches!(c, Caveat::SparseRegion { .. })),
            "{}",
            r.summary()
        );
        assert!(
            has(|c| matches!(
                c,
                Caveat::AtLimit {
                    limit: Limit::Data,
                    ..
                }
            )),
            "{}",
            r.summary()
        );
    }

    #[test]
    fn outliers_are_detected_and_handled() {
        let f = |x: f64, y: f64| (2.0 * x).sin() + y * y;
        let mut data = Samples::new(2);
        let mut st = 9;
        let mut bad = Vec::new();
        for i in 0..600 {
            let (x, y) = (unit_random(&mut st), unit_random(&mut st));
            let mut z = f(x, y) + 0.01 * noise(&mut st);
            if i % 25 == 7 {
                z += 1.0 + unit_random(&mut st);
                bad.push(i);
            }
            data.push([x, y], z).unwrap();
        }
        let r = AutoFit::unit(2).run(&data).unwrap();
        assert!(r.robust.is_some(), "{}", r.summary());
        match &r.caveats[0] {
            Caveat::Outliers { indices, handled } => {
                assert!(*handled);
                assert_eq!(indices, &bad);
            }
            other => panic!("expected outlier caveat first, got {other:?}"),
        }
        let err = (r.fitter.predict([0.5, 0.5]).unwrap() - f(0.5, 0.5)).abs();
        assert!(err < 0.01, "{err}\n{}", r.summary());
    }

    #[test]
    fn target_picks_smallest_sufficient_model() {
        let f = |x: f64| (-x).exp() * (4.0 * x).cos();
        let mut data = Samples::new(1);
        for i in 0..200 {
            let x = i as f64 / 199.0;
            data.push([x], f(x)).unwrap();
        }
        let r = AutoFit::unit(1).target_rms(1e-3).run(&data).unwrap();
        assert!(matches!(r.rule, Rule::Target(_)));
        assert!(r.chosen().cv_rms <= 1e-3);
        let smaller_ok = r
            .candidates
            .iter()
            .any(|c| c.cv_rms <= 1e-3 && c.coefficients < r.chosen().coefficients);
        assert!(!smaller_ok, "{}", r.summary());
    }

    #[test]
    fn scaling_all_weights_changes_nothing() {
        let f = |x: f64| (3.0 * x).sin();
        let (mut a, mut b) = (Samples::new(1), Samples::new(1));
        let mut st = 5;
        for _ in 0..200 {
            let x = unit_random(&mut st);
            let y = f(x) + 0.01 * noise(&mut st);
            a.push([x], y).unwrap();
            b.push_weighted([x], y, 100.0).unwrap();
        }
        let ra = AutoFit::unit(1).target_rms(5e-3).run(&a).unwrap();
        let rb = AutoFit::unit(1).target_rms(5e-3).run(&b).unwrap();
        assert_eq!(ra.chosen, rb.chosen);
        assert_eq!(ra.rule, rb.rule);
        assert_eq!(ra.candidates.len(), rb.candidates.len());
        for (ca, cb) in ra.candidates.iter().zip(&rb.candidates) {
            let rel = (ca.cv_rms - cb.cv_rms).abs() / ca.cv_rms;
            assert!(rel < 1e-9, "{}: {} vs {}", ca.model, ca.cv_rms, cb.cv_rms);
        }
    }

    #[test]
    fn outlier_in_tiny_data_is_still_handled() {
        // Without the outlier, too few samples remain to score any model, so the
        // first selection is kept and only the final fit is robust.
        let mut data = Samples::new(1);
        for (i, x) in [0.1, 0.3, 0.5, 0.7, 0.9].into_iter().enumerate() {
            data.push([x], 2.0 * x + if i == 2 { 5.0 } else { 0.0 })
                .unwrap();
        }
        let r = AutoFit::unit(1).run(&data).unwrap();
        let s = r.summary();
        assert_eq!(r.chosen().model, Model::bezier(1), "{s}");
        assert!(r.robust.is_some(), "{s}");
        assert!(s.contains("too few samples to exclude outliers"), "{s}");
        let outliers = Caveat::Outliers {
            indices: vec![2],
            handled: true,
        };
        assert!(r.caveats.contains(&outliers), "{s}");
        assert!((r.fitter.predict([0.5]).unwrap() - 1.0).abs() < 1e-6, "{s}");
    }

    #[test]
    fn no_model_to_try_is_reported_as_such() {
        let mut data = Samples::new(1);
        assert_eq!(AutoFit::unit(1).run(&data).err(), Some(FitError::NoData));
        data.push([0.5], 1.0).unwrap();
        assert_eq!(
            AutoFit::unit(1).run(&data).err(),
            Some(FitError::NoCandidates)
        );
        for i in 0..50 {
            let x = i as f64 / 49.0;
            data.push([x], x.sin()).unwrap();
        }
        let nothing = AutoFit::unit(1).max_bezier_degree(0).spline_degrees(&[]);
        assert_eq!(nothing.run(&data).err(), Some(FitError::NoCandidates));
        let too_small = AutoFit::unit(1).max_coefficients(1);
        assert_eq!(too_small.run(&data).err(), Some(FitError::NoCandidates));
    }

    #[test]
    fn sample_index_out_of_range_is_none() {
        let mut data = Samples::new(2);
        data.push([0.25, 0.5], 1.0).unwrap();
        assert_eq!(data.get(0), Some((&[0.25, 0.5][..], 1.0, 1.0)));
        assert_eq!(data.get(1), None);
        assert_eq!(data.get(usize::MAX), None);
    }

    #[test]
    fn segment_ladder_ends_at_max_segments() {
        assert_eq!(segment_ladder(1), Vec::<usize>::new());
        assert_eq!(segment_ladder(4), [2, 3, 4]);
        assert_eq!(*segment_ladder(32).last().unwrap(), 32);
        assert_eq!(*segment_ladder(40).last().unwrap(), 40);
        assert_eq!(segment_ladder(100)[12..], [48, 64, 100]);
    }
}
