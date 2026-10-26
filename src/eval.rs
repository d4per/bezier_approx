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
#[inline]
#[allow(clippy::needless_range_loop)]
pub fn bspline_basis(t: f64, segments: usize, b: &mut [f64]) -> usize {
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
#[derive(Clone, Copy, Debug)]
pub struct BezierFn<'a, const D: usize, const P1: usize> {
    pub lo: [f64; D],
    pub hi: [f64; D],
    pub segments: usize,
    pub coeffs: &'a [f64],
}

impl<const D: usize, const P1: usize> BezierFn<'_, D, P1> {
    /// Evaluates the function at `x`.
    pub fn eval(&self, x: [f64; D]) -> f64 {
        let per_axis = self.segments + P1 - 1;
        debug_assert_eq!(
            self.coeffs.len(),
            per_axis.pow(D as u32),
            "coefficient count mismatch"
        );
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
