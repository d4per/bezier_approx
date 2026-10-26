//! Standalone code: the fitted function rewritten as plain polynomials and printed as
//! Rust or C that needs neither this crate nor an allocator.

use crate::auto::unit_random;
use crate::fit::{is_identifier, FitError, Fitter};
use std::fmt::Write;

/// Language of the code from [`standalone`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Lang {
    Rust,
    C,
}

/// Number of test inputs at which the generated code is compared with the model.
const CHECK_POINTS: usize = 1000;

/// Source of a standalone function `name` that computes the fitted function, as
/// documented on [`Fitter::to_rust_fn`] and [`Fitter::to_c_fn`].
pub(crate) fn standalone(fit: &Fitter, name: &str, lang: Lang) -> Result<String, FitError> {
    let valid = match lang {
        Lang::Rust => is_identifier(name),
        Lang::C => is_c_identifier(name),
    };
    if !valid {
        return Err(FitError::InvalidName);
    }
    let pf = PowerForm::new(fit)?;
    let (dims, p1, k) = (pf.dims, pf.degree + 1, pf.segments);
    // Per axis: (offset, scale) of the map from x to u (Bézier) or to t (spline).
    let maps: Vec<(f64, f64)> = (0..dims)
        .map(|d| match k {
            1 => centre_scale(pf.lo[d], pf.hi[d]),
            _ => (pf.lo[d], k as f64 / (pf.hi[d] - pf.lo[d])),
        })
        .collect();
    let finite = pf.pieces.iter().flatten().all(|v| v.is_finite())
        && maps.iter().all(|(a, b)| a.is_finite() && b.is_finite());
    if !finite {
        return Err(FitError::NotFinite);
    }
    let head = header(fit, &pf, lang, pf.max_deviation(fit)?);
    let rust = lang == Lang::Rust;

    let mut s = String::new();
    if rust {
        for line in &head {
            let _ = writeln!(s, "/// {line}");
        }
        let _ = writeln!(s, "#[rustfmt::skip]");
        let _ = writeln!(
            s,
            "#[allow(clippy::unreadable_literal, clippy::approx_constant)]"
        );
        let _ = writeln!(s, "pub fn {name}(x: [f64; {dims}]) -> f64 {{");
    } else {
        let _ = writeln!(s, "/* {}", head[0]);
        for line in &head[1..] {
            let _ = writeln!(s, " * {line}");
        }
        let _ = writeln!(s, " */");
        let _ = writeln!(s, "double {name}(const double x[{dims}])\n{{");
    }
    let comment = |s: &mut String, text: &str| {
        let _ = if rust {
            writeln!(s, "    // {text}")
        } else {
            writeln!(s, "    /* {text} */")
        };
    };
    let float = if rust { "let" } else { "double" };

    if k == 1 {
        if p1 == 1 {
            let _ = writeln!(s, "    {}", if rust { "let _ = x;" } else { "(void)x;" });
        } else {
            comment(&mut s, "Map each input from the box to [-1, 1].");
            for (d, &(centre, scale)) in maps.iter().enumerate() {
                let (centre, scale) = (lit(centre), lit(scale));
                if rust {
                    let _ = writeln!(
                        s,
                        "    let u{d} = ((x[{d}] - {centre}) * {scale}).clamp(-1.0, 1.0);"
                    );
                } else {
                    let _ = writeln!(s, "    double u{d} = (x[{d}] - {centre}) * {scale};");
                    let _ = writeln!(
                        s,
                        "    u{d} = u{d} < -1.0 ? -1.0 : u{d} > 1.0 ? 1.0 : u{d};"
                    );
                }
            }
        }
    } else {
        write_table(&mut s, &pf, lang);
        let what = if p1 > 1 {
            "Find the piece along each axis, and the position in it, u in [-1, 1]."
        } else {
            "Find the piece along each axis."
        };
        comment(&mut s, what);
        let (top, last) = (lit(k as f64), k - 1);
        for (d, &(lo, scale)) in maps.iter().enumerate() {
            let (lo, scale) = (lit(lo), lit(scale));
            if rust {
                let _ = writeln!(
                    s,
                    "    let t{d} = ((x[{d}] - {lo}) * {scale}).clamp(0.0, {top});"
                );
                let _ = writeln!(s, "    let s{d} = (t{d} as usize).min({last});");
                if p1 > 1 {
                    let _ = writeln!(s, "    let u{d} = 2.0 * (t{d} - s{d} as f64) - 1.0;");
                }
            } else {
                let _ = writeln!(s, "    double t{d} = (x[{d}] - {lo}) * {scale};");
                let _ = writeln!(
                    s,
                    "    t{d} = t{d} < 0.0 ? 0.0 : t{d} > {top} ? {top} : t{d};"
                );
                let _ = writeln!(
                    s,
                    "    int s{d} = t{d} < {} ? (int)t{d} : {last};",
                    lit(last as f64)
                );
                if p1 > 1 {
                    let _ = writeln!(s, "    double u{d} = 2.0 * (t{d} - s{d}) - 1.0;");
                }
            }
        }
        let index: Vec<String> = (0..dims)
            .map(|d| match k.pow(d as u32) {
                1 => format!("s{d}"),
                stride => format!("{stride} * s{d}"),
            })
            .collect();
        if rust {
            let _ = writeln!(s, "    let c = &C[{}];", index.join(" + "));
        } else {
            let _ = writeln!(s, "    const double *c = C[{}];", index.join(" + "));
        }
    }

    let coef = |i: usize| {
        if k == 1 {
            lit(pf.pieces[0][i])
        } else {
            format!("c[{i}]")
        }
    };
    let (levels, value) = horner(dims, p1, coef);
    for (i, lets) in levels.iter().enumerate() {
        let level = i + 1;
        let powers: Vec<String> = (level..dims)
            .map(|d| format!("u{d}^{}", index_name(d)))
            .collect();
        let vars: Vec<String> = (0..level).map(|d| format!("u{d}")).collect();
        let indices: Vec<String> = (level..dims).map(index_name).collect();
        let suffix = format!("_{}", indices.join("_"));
        comment(
            &mut s,
            &format!(
                "p{level}{suffix}: the factor of {}, a polynomial in {}.",
                powers.join(if rust { "·" } else { "*" }),
                vars.join(", ")
            ),
        );
        for (name, expr) in lets {
            let _ = writeln!(s, "    {float} {name} = {expr};");
        }
    }
    if rust {
        let _ = writeln!(s, "    {value}\n}}");
    } else {
        let _ = writeln!(s, "    return {value};\n}}");
    }
    Ok(s)
}

/// The comment lines above the function: what was fitted, what the code needs, its
/// cost, and how closely it matches the model.
fn header(fit: &Fitter, pf: &PowerForm, lang: Lang, deviation: f64) -> Vec<String> {
    let (dims, k) = (pf.dims, pf.segments);
    let model = fit.model();
    let cost = model.cost(dims);
    let rust = lang == Lang::Rust;
    let mut head = Vec::new();
    let boxes: Vec<String> = (0..dims)
        .map(|d| format!("[{}, {}]", pf.lo[d], pf.hi[d]))
        .collect();
    let (desc, times) = if rust {
        (model.to_string(), " × ")
    } else {
        (model.to_string().replace('é', "e"), " x ")
    };
    head.push(format!(
        "Generated by bezier_approx: {dims}-D {desc}, on {}.",
        boxes.join(times)
    ));
    if let Some(r) = fit.report() {
        head.push(format!(
            "Trained on {} samples, training RMS residual {:.3e}.",
            r.samples, r.rms_residual
        ));
    }
    head.push(if rust {
        "Plain Rust: no crate, no allocation, works in `no_std`. Inputs outside the box are \
         clamped to it."
            .into()
    } else {
        "Plain C99: no headers or libraries. Inputs outside the box are clamped to it.".into()
    });
    let lookup = if k > 1 {
        let s = if dims == 1 { "" } else { "s" };
        format!(", {dims} float-to-integer conversion{s} and a table lookup")
    } else {
        String::new()
    };
    head.push(format!("Per call: {cost}{lookup}; no divisions or loops."));
    head.push(format!(
        "Stores {} coefficients ({} bytes).",
        cost.values,
        cost.values.saturating_mul(8)
    ));
    let deviation = match deviation {
        0.0 => "0".to_string(),
        d => format!("{d:.1e}"),
    };
    head.push(format!(
        "Largest difference from the fitted model at {CHECK_POINTS} test inputs: {deviation}."
    ));
    head
}

/// The spline's coefficient table `C`, one row per piece.
fn write_table(s: &mut String, pf: &PowerForm, lang: Lang) {
    let (dims, p1, k) = (pf.dims, pf.degree + 1, pf.segments);
    let rust = lang == Lang::Rust;
    let vars: Vec<String> = (0..dims).map(|d| format!("u{d}")).collect();
    let about = match (p1, dims) {
        (1, _) => "One value per piece.".to_string(),
        (_, 1) => "One row per piece: its polynomial in u0, constant term first.".into(),
        _ => format!(
            "One row per piece: its polynomial in {}, the power of u0 varying fastest.",
            vars.join(", ")
        ),
    };
    let (rows, terms) = (pf.pieces.len(), pf.pieces[0].len());
    let (open, close) = if rust { ("[", "]") } else { ("{", "}") };
    if rust {
        let _ = writeln!(s, "    // {about}");
        let _ = writeln!(s, "    static C: [[f64; {terms}]; {rows}] = [");
    } else {
        let _ = writeln!(s, "    /* {about} */");
        let _ = writeln!(s, "    static const double C[{rows}][{terms}] = {{");
    }
    for (piece, a) in pf.pieces.iter().enumerate() {
        let label: Vec<String> = (0..dims)
            .map(|d| {
                let seg = piece / k.pow(d as u32) % k;
                let w = pf.hi[d] - pf.lo[d];
                let at = |j: usize| pf.lo[d] + w * j as f64 / k as f64;
                format!("x{d} in [{}, {}]", at(seg), at(seg + 1))
            })
            .collect();
        let label = if rust {
            format!("// {}", label.join(", "))
        } else {
            format!("/* {} */", label.join(", "))
        };
        let lines: Vec<String> = a
            .chunks(p1)
            .map(|run| run.iter().map(|&v| lit(v)).collect::<Vec<_>>().join(", "))
            .collect();
        if dims == 1 {
            let _ = writeln!(s, "        {open}{}{close}, {label}", lines[0]);
        } else {
            let _ = writeln!(s, "        {open} {label}");
            for line in lines {
                let _ = writeln!(s, "            {line},");
            }
            let _ = writeln!(s, "        {close},");
        }
    }
    let _ = writeln!(s, "    {close};");
}

/// Shortest representation that reads back as the same `f64`; valid in Rust and C.
fn lit(v: f64) -> String {
    format!("{v:?}")
}

/// Name for the power of `u_axis` in comments: j, k, l, … for axes 1, 2, 3, …
fn index_name(axis: usize) -> String {
    match u8::try_from(axis).ok().filter(|&a| a < 17) {
        Some(a) => char::from(b'i' + a).to_string(),
        None => format!("e{axis}"),
    }
}

/// Nested Horner evaluation of a tensor-product polynomial with `p1` coefficients per
/// axis, where `coef(i)` prints coefficient `i` (power of `u0` varying fastest).
///
/// Axis 0 is evaluated for every combination of the higher powers, then axis 1 over
/// those results, and so on. Returns the intermediate assignments `(name, expression)`
/// of each level (`p1_…` after axis 0, `p2_…` after axis 1, …) and the final
/// expression. Uses `p1^dims − 1` multiplications and additions.
fn horner(
    dims: usize,
    p1: usize,
    coef: impl Fn(usize) -> String,
) -> (Vec<Vec<(String, String)>>, String) {
    let mut levels = Vec::new();
    if p1 == 1 {
        return (levels, coef(0));
    }
    let mut operands: Vec<String> = (0..p1.pow(dims as u32)).map(coef).collect();
    for d in 0..dims {
        let exprs: Vec<String> = operands
            .chunks(p1)
            .map(|run| horner_1d(&format!("u{d}"), run))
            .collect();
        if d + 1 == dims {
            return (levels, exprs.into_iter().next().expect("one value left"));
        }
        let lets: Vec<(String, String)> = exprs
            .into_iter()
            .enumerate()
            .map(|(mut r, expr)| {
                let mut name = format!("p{}", d + 1);
                for _ in d + 1..dims {
                    let _ = write!(name, "_{}", r % p1);
                    r /= p1;
                }
                (name, expr)
            })
            .collect();
        operands = lets.iter().map(|(name, _)| name.clone()).collect();
        levels.push(lets);
    }
    unreachable!("dims >= 1")
}

/// `a0 + u * (a1 + u * (… + u * an))`.
fn horner_1d(u: &str, terms: &[String]) -> String {
    let n = terms.len() - 1;
    let mut s = terms[n].clone();
    for i in (0..n).rev() {
        s = if i + 1 == n {
            format!("{} + {u} * {s}", terms[i])
        } else {
            format!("{} + {u} * ({s})", terms[i])
        };
    }
    s
}

/// Whether `name` can name the generated C function: an ASCII identifier that is not
/// a keyword of any C standard up to C23.
fn is_c_identifier(name: &str) -> bool {
    const KEYWORDS: &[&str] = &[
        "_Alignas",
        "_Alignof",
        "_Atomic",
        "_BitInt",
        "_Bool",
        "_Complex",
        "_Decimal128",
        "_Decimal32",
        "_Decimal64",
        "_Generic",
        "_Imaginary",
        "_Noreturn",
        "_Static_assert",
        "_Thread_local",
        "alignas",
        "alignof",
        "auto",
        "bool",
        "break",
        "case",
        "char",
        "const",
        "constexpr",
        "continue",
        "default",
        "do",
        "double",
        "else",
        "enum",
        "extern",
        "false",
        "float",
        "for",
        "goto",
        "if",
        "inline",
        "int",
        "long",
        "nullptr",
        "register",
        "restrict",
        "return",
        "short",
        "signed",
        "sizeof",
        "static",
        "static_assert",
        "struct",
        "switch",
        "thread_local",
        "true",
        "typedef",
        "typeof",
        "typeof_unqual",
        "union",
        "unsigned",
        "void",
        "volatile",
        "while",
    ];
    let mut chars = name.chars();
    matches!(chars.next(), Some(c) if c.is_ascii_alphabetic() || c == '_')
        && chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
        && !KEYWORDS.contains(&name)
}

/// The fitted function as piecewise polynomials in power form.
///
/// Piece `s` (one segment index per axis, axis 0 fastest) is the part of the box
/// where every `x_d` lies in segment `s_d`. On it the function is
/// `Σ_m a_m · u_0^m_0 ··· u_(D-1)^m_(D-1)`, where `u_d ∈ [-1, 1]` is the position of
/// `x_d` within its segment. Centring the variable keeps the monomials well
/// conditioned: their coefficients stay moderate where powers of `[0, 1]` would cancel.
struct PowerForm {
    dims: usize,
    degree: usize,
    segments: usize,
    lo: Vec<f64>,
    hi: Vec<f64>,
    /// `(degree + 1)^dims` coefficients per piece, the power of `u_0` varying fastest.
    pieces: Vec<Vec<f64>>,
}

impl PowerForm {
    fn new(fit: &Fitter) -> Result<Self, FitError> {
        let c = fit.coefficients().ok_or(FitError::NotSolved)?;
        let (dims, p, k) = (fit.dims(), fit.degree(), fit.segments());
        let p1 = p + 1;
        let per_axis = p + k;
        let terms = p1.pow(dims as u32);
        let basis: Vec<Vec<f64>> = (0..k).map(|s| segment_basis(p, k, s)).collect();
        let mut pieces = Vec::with_capacity(k.pow(dims as u32));
        for piece in 0..k.pow(dims as u32) {
            let seg: Vec<usize> = (0..dims).map(|d| piece / k.pow(d as u32) % k).collect();
            // The control values acting on this piece: those of basis functions s_d + j_d.
            let mut a: Vec<f64> = (0..terms)
                .map(|local| {
                    let (mut r, mut flat, mut stride) = (local, 0, 1);
                    for &s in &seg {
                        flat += (s + r % p1) * stride;
                        r /= p1;
                        stride *= per_axis;
                    }
                    c[flat]
                })
                .collect();
            // Change basis one axis at a time: B-spline basis functions → powers of u_d.
            let mut stride = 1;
            for &s in &seg {
                let m = &basis[s];
                let mut out = vec![0.0; terms];
                for base in (0..terms).filter(|b| b / stride % p1 == 0) {
                    for e in 0..p1 {
                        out[base + e * stride] =
                            (0..p1).map(|j| a[base + j * stride] * m[j * p1 + e]).sum();
                    }
                }
                a = out;
                stride *= p1;
            }
            pieces.push(a);
        }
        Ok(PowerForm {
            dims,
            degree: p,
            segments: k,
            lo: fit.lo().to_vec(),
            hi: fit.hi().to_vec(),
            pieces,
        })
    }

    /// Evaluates with the same operations, in the same order, as the generated code.
    fn eval(&self, x: &[f64]) -> f64 {
        let (p1, k) = (self.degree + 1, self.segments);
        let mut u = vec![0.0; self.dims];
        let mut piece = 0;
        let mut stride = 1;
        for d in 0..self.dims {
            if k == 1 {
                let (centre, scale) = centre_scale(self.lo[d], self.hi[d]);
                u[d] = ((x[d] - centre) * scale).clamp(-1.0, 1.0);
            } else {
                let scale = k as f64 / (self.hi[d] - self.lo[d]);
                let t = ((x[d] - self.lo[d]) * scale).clamp(0.0, k as f64);
                let s = (t as usize).min(k - 1);
                u[d] = 2.0 * (t - s as f64) - 1.0;
                piece += s * stride;
                stride *= k;
            }
        }
        let mut vals = self.pieces[piece].clone();
        for &ud in &u {
            vals = vals
                .chunks(p1)
                .map(|run| {
                    let (last, rest) = run.split_last().expect("p1 >= 1");
                    rest.iter().rev().fold(*last, |h, &a| a + ud * h)
                })
                .collect();
        }
        vals[0]
    }

    /// Largest difference from `fit.predict` at pseudo-random inputs in the box.
    fn max_deviation(&self, fit: &Fitter) -> Result<f64, FitError> {
        let mut state = 0x5eed_c0de;
        let mut x = vec![0.0; self.dims];
        let mut worst = 0.0f64;
        for _ in 0..CHECK_POINTS {
            for (d, xd) in x.iter_mut().enumerate() {
                *xd = self.lo[d] + unit_random(&mut state) * (self.hi[d] - self.lo[d]);
            }
            worst = worst.max((self.eval(&x) - fit.predict(&x)?).abs());
        }
        Ok(worst)
    }
}

/// Centre and scale mapping `[lo, hi]` onto `[-1, 1]`: `u = (x − centre) · scale`.
fn centre_scale(lo: f64, hi: f64) -> (f64, f64) {
    let w = hi - lo;
    (lo + 0.5 * w, 2.0 / w)
}

/// The `p + 1` B-spline basis functions that are nonzero on segment `s` of `k`
/// (clamped uniform knots, as in [`bspline_basis`](crate::bspline_basis)), as
/// polynomials in the position `u ∈ [-1, 1]` within the segment. Entry `j·(p+1) + e`
/// is the coefficient of `u^e` in basis function `s + j`.
///
/// This is the Cox–de Boor recurrence of `bspline_basis` carried out on polynomials.
/// Lengths are measured in half-segments, so `t = 2s + 1 + u` and the knots are even
/// integers: every factor is exact, and only the divisions round.
#[allow(clippy::needless_range_loop)]
fn segment_basis(p: usize, k: usize, s: usize) -> Vec<f64> {
    let p1 = p + 1;
    let knot = |j: usize| 2.0 * (j as f64 - p as f64).clamp(0.0, k as f64);
    let t0 = (2 * s + 1) as f64;
    let i = s + p;
    let mut b = vec![vec![0.0; p1]; p1];
    b[0][0] = 1.0;
    for j in 1..=p {
        let mut saved = vec![0.0; p1];
        for r in 0..j {
            let (kr, kl) = (knot(i + r + 1), knot(i + r + 1 - j));
            let temp: Vec<f64> = b[r].iter().map(|v| v / (kr - kl)).collect();
            // right = kr − t = (kr − t0) − u,  left = t − kl = (t0 − kl) + u
            let mut next = saved;
            add_linear_times(&mut next, kr - t0, -1.0, &temp);
            b[r] = next;
            saved = vec![0.0; p1];
            add_linear_times(&mut saved, t0 - kl, 1.0, &temp);
        }
        b[j] = saved;
    }
    b.concat()
}

/// `out += (a + b·u) · q` for polynomials stored as coefficient vectors; the top
/// coefficient of `q` must be 0.
fn add_linear_times(out: &mut [f64], a: f64, b: f64, q: &[f64]) {
    for e in 0..out.len() {
        out[e] += a * q[e] + if e > 0 { b * q[e - 1] } else { 0.0 };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{bspline_basis, Model};

    /// A fitter for `model` on `[lo, hi]`, trained on a smooth function at
    /// pseudo-random inputs.
    fn trained(model: Model, lo: &[f64], hi: &[f64]) -> Fitter {
        let dims = lo.len();
        let mut fit = Fitter::with_model(model, lo, hi);
        let mut state = 17;
        let n = 3 * fit.num_coefficients() + 20;
        for _ in 0..n {
            let x: Vec<f64> = (0..dims)
                .map(|d| lo[d] + unit_random(&mut state) * (hi[d] - lo[d]))
                .collect();
            let y = x
                .iter()
                .enumerate()
                .map(|(d, v)| (v * (d + 2) as f64).sin())
                .sum::<f64>();
            fit.add_sample(&x, y).unwrap();
        }
        fit.solve().unwrap();
        fit
    }

    const SHAPES: &[(usize, usize, usize)] = &[
        // (dims, degree, segments)
        (1, 0, 1),
        (1, 1, 1),
        (1, 5, 1),
        (1, 12, 1),
        (1, 0, 4),
        (1, 1, 6),
        (1, 3, 8),
        (2, 0, 1),
        (2, 1, 1),
        (2, 4, 1),
        (2, 1, 3),
        (2, 2, 4),
        (3, 2, 1),
        (3, 2, 2),
        (4, 1, 2),
    ];

    fn box_for(dims: usize) -> (Vec<f64>, Vec<f64>) {
        let lo: Vec<f64> = (0..dims).map(|d| -1.0 + 0.5 * d as f64).collect();
        let hi: Vec<f64> = (0..dims).map(|d| 2.0 + d as f64).collect();
        (lo, hi)
    }

    #[test]
    fn segment_basis_matches_bspline_basis() {
        for (p, k) in [
            (0, 1),
            (1, 1),
            (3, 1),
            (8, 1),
            (1, 5),
            (2, 3),
            (3, 7),
            (4, 2),
        ] {
            for s in 0..k {
                let m = segment_basis(p, k, s);
                for u in [-1.0, -0.6, 0.0, 0.3, 0.999] {
                    let t = (s as f64 + (u + 1.0) / 2.0) / k as f64;
                    let mut b = vec![0.0; p + 1];
                    assert_eq!(bspline_basis(t, k, &mut b), s);
                    for (j, bj) in b.iter().enumerate() {
                        let row = &m[j * (p + 1)..(j + 1) * (p + 1)];
                        let poly = row.iter().rev().fold(0.0, |h, &a| a + u * h);
                        assert!((poly - bj).abs() < 1e-13, "p={p} k={k} s={s} u={u} j={j}");
                    }
                }
            }
        }
    }

    #[test]
    fn power_form_matches_predict_inside_and_outside_the_box() {
        for &(dims, degree, segments) in SHAPES {
            let (lo, hi) = box_for(dims);
            let fit = trained(Model::spline(degree, segments), &lo, &hi);
            let pf = PowerForm::new(&fit).unwrap();
            let mut state = 99;
            for _ in 0..300 {
                // A margin of 20 % on each side tests the clamping as well.
                let x: Vec<f64> = (0..dims)
                    .map(|d| lo[d] + (1.4 * unit_random(&mut state) - 0.2) * (hi[d] - lo[d]))
                    .collect();
                let (got, want) = (pf.eval(&x), fit.predict(&x).unwrap());
                assert!(
                    (got - want).abs() < 1e-12 * (1.0 + want.abs()),
                    "{dims}-D degree {degree}, {segments} segments at {x:?}: {got} vs {want}"
                );
            }
        }
    }

    /// Operations in the function body: binary operators are always written with a
    /// space on each side, unlike the sign of a negative literal. Integer index
    /// arithmetic (`C[s0 + 4 * s1]`) and comments are skipped.
    fn count_ops(code: &str) -> (usize, usize, usize) {
        let body = code.split_once("\n{").map_or(code, |(_, b)| b);
        let body = body.split_once(") -> f64 {").map_or(body, |(_, b)| b);
        let (mut mul, mut add, mut cmp) = (0, 0, 0);
        for line in body.lines().filter(|l| !l.contains("C[")) {
            let line = line.split("//").next().unwrap().split("/*").next().unwrap();
            mul += line.matches(" * ").count();
            add += line.matches(" + ").count() + line.matches(" - ").count();
            cmp += line.matches(" < ").count() + line.matches(" > ").count();
            cmp += 2 * line.matches(".clamp(").count() + line.matches(".min(").count();
        }
        (mul, add, cmp)
    }

    #[test]
    fn operation_counts_match_model_cost() {
        for &(dims, degree, segments) in SHAPES {
            let (lo, hi) = box_for(dims);
            let model = Model::spline(degree, segments);
            let fit = trained(model, &lo, &hi);
            let cost = model.cost(dims);
            for code in [fit.to_rust_fn("f").unwrap(), fit.to_c_fn("f").unwrap()] {
                assert_eq!(
                    count_ops(&code),
                    (cost.mul, cost.add, cost.cmp),
                    "{model} in {dims}-D:\n{code}"
                );
                assert!(code.contains(&format!("Per call: {cost}")), "{code}");
                assert!(
                    code.contains(&format!("Stores {} coefficients", cost.values)),
                    "{code}"
                );
            }
        }
    }

    #[test]
    fn cost_formulas() {
        let c = Model::bezier(4).cost(2);
        assert_eq!(
            (c.mul, c.add, c.cmp, c.values, c.flops()),
            (26, 26, 4, 25, 52)
        );
        let c = Model::spline(3, 8).cost(1);
        assert_eq!((c.mul, c.add, c.cmp, c.values), (5, 6, 3, 32));
        assert_eq!(Model::bezier(0).cost(3).flops(), 0);
        assert_eq!(Model::bezier(1).cost(64).values, usize::MAX);
        assert_eq!(Model::spline(3, 8).cost(usize::MAX).cmp, usize::MAX);
    }

    /// The code with every float literal (and its sign) replaced by `#`, to compare
    /// its shape.
    fn shape(code: &str) -> String {
        let mut out = String::new();
        let mut chars = code.chars().peekable();
        let mut prev = ' ';
        while let Some(c) = chars.next() {
            let next_digit = chars.peek().is_some_and(char::is_ascii_digit);
            let sign = c == '-' && next_digit && " ([{".contains(prev);
            if sign || c.is_ascii_digit() && !(prev.is_ascii_alphanumeric() || prev == '_') {
                let mut tok = c.to_string();
                while let Some(&n) = chars.peek() {
                    let exp_sign = (n == '-' || n == '+') && tok.ends_with('e');
                    if n.is_ascii_digit() || n == '.' || n == 'e' || exp_sign {
                        tok.push(n);
                        chars.next();
                    } else {
                        break;
                    }
                }
                out.push_str(if tok.contains('.') { "#" } else { &tok });
                prev = '0';
            } else {
                out.push(c);
                prev = c;
            }
        }
        out
    }

    #[test]
    fn bezier_code_shape() {
        let fit = trained(Model::bezier(2), &[0.0, -1.0], &[1.0, 1.0]);
        let body = |code: &str| shape(code.split_once("#[allow").map_or(code, |(_, b)| b));
        assert_eq!(
            body(&fit.to_rust_fn("surface").unwrap()),
            "(clippy::unreadable_literal, clippy::approx_constant)]
pub fn surface(x: [f64; 2]) -> f64 {
    // Map each input from the box to [-1, 1].
    let u0 = ((x[0] - #) * #).clamp(#, #);
    let u1 = ((x[1] - #) * #).clamp(#, #);
    // p1_j: the factor of u1^j, a polynomial in u0.
    let p1_0 = # + u0 * (# + u0 * #);
    let p1_1 = # + u0 * (# + u0 * #);
    let p1_2 = # + u0 * (# + u0 * #);
    p1_0 + u1 * (p1_1 + u1 * p1_2)
}
"
        );
        let c = fit.to_c_fn("surface").unwrap();
        assert!(
            c.starts_with(
                "/* Generated by bezier_approx: 2-D Bezier, degree 2, on [0, 1] x [-1, 1].\n"
            ),
            "{c}"
        );
        assert!(c.is_ascii(), "{c}");
        assert_eq!(
            shape(c.split_once(" */\n").unwrap().1),
            "double surface(const double x[2])
{
    /* Map each input from the box to [-1, 1]. */
    double u0 = (x[0] - #) * #;
    u0 = u0 < # ? # : u0 > # ? # : u0;
    double u1 = (x[1] - #) * #;
    u1 = u1 < # ? # : u1 > # ? # : u1;
    /* p1_j: the factor of u1^j, a polynomial in u0. */
    double p1_0 = # + u0 * (# + u0 * #);
    double p1_1 = # + u0 * (# + u0 * #);
    double p1_2 = # + u0 * (# + u0 * #);
    return p1_0 + u1 * (p1_1 + u1 * p1_2);
}
"
        );
    }

    #[test]
    fn spline_code_shape() {
        let fit = trained(Model::spline(1, 2), &[0.0, 0.0], &[1.0, 2.0]);
        let rust = fit.to_rust_fn("f").unwrap();
        assert_eq!(
            shape(rust.split_once("-> f64 {\n").unwrap().1),
            "    // One row per piece: its polynomial in u0, u1, the power of u0 varying fastest.
    static C: [[f64; 4]; 4] = [
        [ // x0 in [0, #], x1 in [0, 1]
            #, #,
            #, #,
        ],
        [ // x0 in [#, 1], x1 in [0, 1]
            #, #,
            #, #,
        ],
        [ // x0 in [0, #], x1 in [1, 2]
            #, #,
            #, #,
        ],
        [ // x0 in [#, 1], x1 in [1, 2]
            #, #,
            #, #,
        ],
    ];
    // Find the piece along each axis, and the position in it, u in [-1, 1].
    let t0 = ((x[0] - #) * #).clamp(#, #);
    let s0 = (t0 as usize).min(1);
    let u0 = # * (t0 - s0 as f64) - #;
    let t1 = ((x[1] - #) * #).clamp(#, #);
    let s1 = (t1 as usize).min(1);
    let u1 = # * (t1 - s1 as f64) - #;
    let c = &C[s0 + 2 * s1];
    // p1_j: the factor of u1^j, a polynomial in u0.
    let p1_0 = c[0] + u0 * c[1];
    let p1_1 = c[2] + u0 * c[3];
    p1_0 + u1 * p1_1
}
"
        );
        let c = fit.to_c_fn("f").unwrap();
        let tail = c.split_once("    };\n").unwrap().1;
        assert_eq!(
            shape(tail),
            "    /* Find the piece along each axis, and the position in it, u in [-1, 1]. */
    double t0 = (x[0] - #) * #;
    t0 = t0 < # ? # : t0 > # ? # : t0;
    int s0 = t0 < # ? (int)t0 : 1;
    double u0 = # * (t0 - s0) - #;
    double t1 = (x[1] - #) * #;
    t1 = t1 < # ? # : t1 > # ? # : t1;
    int s1 = t1 < # ? (int)t1 : 1;
    double u1 = # * (t1 - s1) - #;
    const double *c = C[s0 + 2 * s1];
    /* p1_j: the factor of u1^j, a polynomial in u0. */
    double p1_0 = c[0] + u0 * c[1];
    double p1_1 = c[2] + u0 * c[3];
    return p1_0 + u1 * p1_1;
}
"
        );
        assert!(c.contains("    static const double C[4][4] = {\n        { /* x0 in [0, 0.5], x1 in [0, 1] */\n"), "{c}");
    }

    #[test]
    fn three_dimensions_name_two_levels() {
        let fit = trained(Model::bezier(1), &[0.0; 3], &[1.0; 3]);
        let code = fit.to_rust_fn("f").unwrap();
        assert!(
            code.contains("// p1_j_k: the factor of u1^j·u2^k, a polynomial in u0.\n"),
            "{code}"
        );
        assert!(code.contains("    let p1_1_1 = "), "{code}");
        assert!(
            code.contains("// p2_k: the factor of u2^k, a polynomial in u0, u1.\n"),
            "{code}"
        );
        assert!(
            code.contains("    let p2_1 = p1_0_1 + u1 * p1_1_1;\n"),
            "{code}"
        );
        assert!(code.contains("    p2_0 + u2 * p2_1\n}"), "{code}");
        let c = fit.to_c_fn("f").unwrap();
        assert!(
            c.contains("/* p1_j_k: the factor of u1^j*u2^k, a polynomial in u0. */"),
            "{c}"
        );
        assert!(c.is_ascii(), "{c}");
    }

    #[test]
    fn constant_models_use_no_inputs() {
        let fit = trained(Model::bezier(0), &[0.0, 0.0], &[1.0, 1.0]);
        let code = fit.to_rust_fn("f").unwrap();
        assert!(code.contains("    let _ = x;\n"), "{code}");
        assert!(!code.contains("u0"), "{code}");
        let fit = trained(Model::spline(0, 3), &[0.0], &[1.0]);
        let code = fit.to_c_fn("f").unwrap();
        assert!(code.contains("    return c[0];\n}"), "{code}");
        assert!(!code.contains("u0"), "{code}");
    }

    #[test]
    fn names_are_checked_per_language() {
        let fit = trained(Model::bezier(1), &[0.0], &[1.0]);
        for bad in ["", "1x", "my fn", "x)", "Ä"] {
            assert_eq!(fit.to_rust_fn(bad), Err(FitError::InvalidName), "{bad:?}");
            assert_eq!(fit.to_c_fn(bad), Err(FitError::InvalidName), "{bad:?}");
        }
        assert_eq!(fit.to_rust_fn("fn"), Err(FitError::InvalidName));
        assert!(fit.to_c_fn("fn").is_ok());
        for keyword in ["double", "int", "static", "_Bool", "restrict", "nullptr"] {
            assert_eq!(
                fit.to_c_fn(keyword),
                Err(FitError::InvalidName),
                "{keyword}"
            );
        }
        assert!(fit.to_rust_fn("double").is_ok());
        assert_eq!(Fitter::unit(1, 2).to_c_fn("f"), Err(FitError::NotSolved));
    }

    #[test]
    fn header_reports_the_check_against_the_model() {
        let fit = trained(Model::spline(3, 5), &[0.0], &[1.0]);
        let code = fit.to_rust_fn("f").unwrap();
        let line = code
            .lines()
            .find(|l| l.contains("Largest difference"))
            .unwrap();
        let value: f64 = line
            .rsplit(": ")
            .next()
            .unwrap()
            .trim_end_matches('.')
            .parse()
            .unwrap();
        assert!(value < 1e-14, "{line}");
        assert!(
            code.contains("1 float-to-integer conversion and a table lookup"),
            "{code}"
        );
    }
}
