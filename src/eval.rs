//! Inference: allocation-free evaluation of a trained function.

/// Writes the Bernstein basis of degree `b.len() - 1` at `t` into `b`.
///
/// Uses the de Casteljau recurrence `B_j^k = (1-t)·B_j^(k-1) + t·B_(j-1)^(k-1)`,
/// which is numerically stable and needs no binomial coefficients.
#[inline]
pub fn bernstein_into(t: f64, b: &mut [f64]) {
    if b.is_empty() {
        return;
    }
    let s = 1.0 - t;
    b.fill(0.0);
    b[0] = 1.0;
    for k in 1..b.len() {
        for j in (1..=k).rev() {
            b[j] = s * b[j] + t * b[j - 1];
        }
        b[0] *= s;
    }
}

/// Evaluates the B-spline basis functions of degree `b.len() - 1` that are nonzero
/// at `t ∈ [0, 1]`, for `segments` equal pieces with a clamped knot vector, and writes
/// them into `b`. Returns the index of the first of them; the others follow
/// consecutively. With `segments == 1` this is exactly the Bernstein (Bézier) basis.
///
/// This is the Cox–de Boor recurrence (Piegl & Tiller, *The NURBS Book*, A2.2),
/// with the knot values computed on the fly so no scratch memory is needed.
///
/// # Panics
/// If `b` is empty.
#[inline]
#[allow(clippy::needless_range_loop)]
pub fn bspline_basis(t: f64, segments: usize, b: &mut [f64]) -> usize {
    assert!(!b.is_empty(), "basis slice must hold degree + 1 values");
    let p = b.len() - 1;
    let k = segments.max(1);
    let t = t.clamp(0.0, 1.0);
    let span = ((t * k as f64) as usize).min(k - 1);
    // Clamped uniform knots: p+1 zeros, interior knots j/k, p+1 ones.
    let knot = |j: usize| ((j as f64 - p as f64) / k as f64).clamp(0.0, 1.0);
    let i = span + p; // knot span index in the full knot vector
    b[0] = 1.0;
    for j in 1..=p {
        let mut saved = 0.0;
        for r in 0..j {
            let right = knot(i + r + 1) - t;
            let left = t - knot(i + r + 1 - j);
            let temp = b[r] / (right + left);
            b[r] = saved + right * temp;
            saved = left * temp;
        }
        b[j] = saved;
    }
    span
}

/// A trained function `f: [lo, hi] ⊂ R^D → R`: a tensor-product Bézier patch, or a
/// smooth chain of them (a clamped uniform B-spline) when `segments > 1`.
///
/// * `D`  – number of input dimensions.
/// * `P1` – degree + 1. Each evaluation combines `P1^D` coefficients.
/// * `segments` – pieces per axis; 1 = a single Bézier polynomial. Neighbouring
///   pieces join with `degree − 1` continuous derivatives.
/// * `coeffs` – `(segments + P1 − 1)^D` control values, axis 0 varying fastest.
///
/// Inputs outside the box are clamped to its boundary. The output always lies
/// between the smallest and largest control value (convex hull property).
///
/// Build one with [`new`](BezierFn::new), usually in a `const` emitted by
/// `Fitter::to_rust_code`.
#[derive(Clone, Copy, Debug)]
pub struct BezierFn<'a, const D: usize, const P1: usize> {
    lo: [f64; D],
    hi: [f64; D],
    segments: usize,
    coeffs: &'a [f64],
}

impl<'a, const D: usize, const P1: usize> BezierFn<'a, D, P1> {
    /// Creates the function on the box `[lo, hi]` with `segments` pieces per axis and
    /// `(segments + P1 − 1)^D` control values, axis 0 varying fastest.
    ///
    /// In a `const` item the checks run at compile time, so a coefficient table that
    /// does not match the declared shape is a compile error:
    ///
    /// ```compile_fail
    /// // Degree 1 (P1 = 2) with one segment needs 2 coefficients, not 3.
    /// const F: bezier_approx::BezierFn<'static, 1, 2> =
    ///     bezier_approx::BezierFn::new([0.0], [1.0], 1, &[1.0, 2.0, 3.0]);
    /// let _ = F.eval([0.5]);
    /// ```
    ///
    /// # Panics
    /// If `P1 == 0`, `segments == 0`, `lo[d] >= hi[d]` or the width `hi[d] - lo[d]` is
    /// not finite on some axis, or `coeffs.len() != (segments + P1 − 1)^D`.
    pub const fn new(lo: [f64; D], hi: [f64; D], segments: usize, coeffs: &'a [f64]) -> Self {
        assert!(P1 >= 1, "P1 (degree + 1) must be at least 1");
        assert!(segments >= 1, "need at least one segment");
        let per_axis = segments + P1 - 1;
        let mut expected: usize = 1;
        let mut d = 0;
        while d < D {
            assert!(
                lo[d] < hi[d] && hi[d] - lo[d] < f64::INFINITY,
                "need lo < hi and a finite width on every axis"
            );
            expected = match expected.checked_mul(per_axis) {
                Some(n) => n,
                None => panic!("coefficient count overflows usize"),
            };
            d += 1;
        }
        assert!(
            coeffs.len() == expected,
            "coefficient count must be (segments + P1 - 1)^D"
        );
        BezierFn {
            lo,
            hi,
            segments,
            coeffs,
        }
    }

    /// Lower corner of the box.
    pub const fn lo(&self) -> &[f64; D] {
        &self.lo
    }

    /// Upper corner of the box.
    pub const fn hi(&self) -> &[f64; D] {
        &self.hi
    }

    /// Pieces per axis (1 for a single Bézier polynomial).
    pub const fn segments(&self) -> usize {
        self.segments
    }

    /// The control values, axis 0 varying fastest.
    pub const fn coeffs(&self) -> &'a [f64] {
        self.coeffs
    }

    /// Evaluates the function at `x`.
    pub fn eval(&self, x: [f64; D]) -> f64 {
        let per_axis = self.segments + P1 - 1;
        let mut basis = [[0.0f64; P1]; D];
        let mut first = [0usize; D];
        for (d, b) in basis.iter_mut().enumerate() {
            let t = (x[d] - self.lo[d]) / (self.hi[d] - self.lo[d]);
            first[d] = bspline_basis(t, self.segments, b);
        }
        // Σ over the P1^D nonzero terms, walked like an odometer (axis 0 fastest).
        let mut idx = [0usize; D];
        let mut sum = 0.0;
        loop {
            let mut w = 1.0;
            let mut flat = 0;
            let mut stride = 1;
            for d in 0..D {
                w *= basis[d][idx[d]];
                flat += (first[d] + idx[d]) * stride;
                stride *= per_axis;
            }
            sum += w * self.coeffs[flat];
            let mut d = 0;
            loop {
                if d == D {
                    return sum;
                }
                idx[d] += 1;
                if idx[d] < P1 {
                    break;
                }
                idx[d] = 0;
                d += 1;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bernstein_matches_closed_form() {
        let mut b = [0.0; 5];
        let t: f64 = 0.37;
        bernstein_into(t, &mut b);
        let binom = [1.0, 4.0, 6.0, 4.0, 1.0];
        for k in 0..5 {
            let exact = binom[k] * t.powi(k as i32) * (1.0 - t).powi(4 - k as i32);
            assert!((b[k] - exact).abs() < 1e-15);
        }
        assert!((b.iter().sum::<f64>() - 1.0).abs() < 1e-15);
    }

    /// Reference Cox–de Boor recursion on the explicit clamped knot vector.
    fn naive_bspline(i: usize, p: usize, knots: &[f64], t: f64) -> f64 {
        if p == 0 {
            let last = knots.len() - 1;
            let in_span = knots[i] <= t && t < knots[i + 1];
            // Close the last nonempty span at t = 1.
            let at_end = t == knots[last] && knots[i] < knots[i + 1] && knots[i + 1] == knots[last];
            return if in_span || at_end { 1.0 } else { 0.0 };
        }
        let mut v = 0.0;
        let d1 = knots[i + p] - knots[i];
        if d1 > 0.0 {
            v += (t - knots[i]) / d1 * naive_bspline(i, p - 1, knots, t);
        }
        let d2 = knots[i + p + 1] - knots[i + 1];
        if d2 > 0.0 {
            v += (knots[i + p + 1] - t) / d2 * naive_bspline(i + 1, p - 1, knots, t);
        }
        v
    }

    #[test]
    fn bspline_basis_matches_cox_de_boor() {
        for (p, k) in [(1, 1), (2, 1), (3, 1), (1, 4), (2, 3), (3, 5), (4, 2)] {
            let mut knots = vec![0.0; p + 1];
            knots.extend((1..k).map(|j| j as f64 / k as f64));
            knots.extend(vec![1.0; p + 1]);
            for &t in &[0.0, 0.07, 0.3, 0.5, 0.61, 0.999, 1.0] {
                let mut b = vec![0.0; p + 1];
                let first = bspline_basis(t, k, &mut b);
                for i in 0..p + k {
                    let expected = naive_bspline(i, p, &knots, t);
                    let got = if (first..=first + p).contains(&i) {
                        b[i - first]
                    } else {
                        0.0
                    };
                    assert!((got - expected).abs() < 1e-12, "p={p} k={k} t={t} i={i}");
                }
            }
        }
    }

    #[test]
    fn const_fn_evaluates() {
        // x·y on [0, 1]² as a bilinear patch: only the corner (1, 1) is nonzero.
        const XY: BezierFn<'static, 2, 2> =
            BezierFn::new([0.0; 2], [1.0; 2], 1, &[0.0, 0.0, 0.0, 1.0]);
        assert!((XY.eval([0.3, 0.6]) - 0.18).abs() < 1e-15);
        assert_eq!(XY.eval([2.0, 2.0]), 1.0); // clamped to the box
        assert_eq!(XY.coeffs().len(), 4);
    }

    #[test]
    #[should_panic(expected = "coefficient count")]
    fn new_rejects_wrong_coefficient_count() {
        let c = [0.0; 9];
        let _ = BezierFn::<2, 3>::new([0.0; 2], [1.0; 2], 2, &c);
    }

    #[test]
    #[should_panic(expected = "lo < hi")]
    fn new_rejects_empty_box() {
        let _ = BezierFn::<1, 2>::new([1.0], [1.0], 1, &[0.0, 1.0]);
    }
}
