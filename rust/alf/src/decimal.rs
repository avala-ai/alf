//! Exact comparison of decimal literals, without arbitrary-precision arithmetic.
//!
//! The header rules ask one question repeatedly: *do these two decimal texts
//! denote the same number?* — `9007199254740993.0` against the `...992` float64
//! actually read, or `18446744073709552000` against the shortest repr
//! `1.8446744073709552e+19`.
//!
//! That question needs exactness but not arithmetic. Normalising each text to
//! `(sign, significant digits, exponent)` answers it with string comparison, and
//! sidesteps pulling a bignum crate into a decoder that must also build for
//! `wasm32-unknown-unknown`.

/// A decimal literal, reduced to a form where equality is structural.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Decimal {
    pub negative: bool,
    /// Significant digits, no leading or trailing zeros. Empty means zero.
    pub digits: String,
    /// Power of ten applied to `digits` read as an integer.
    pub exponent: i64,
}

impl Decimal {
    pub fn is_zero(&self) -> bool {
        self.digits.is_empty()
    }

    /// True when this literal denotes a mathematical integer.
    pub fn is_integral(&self) -> bool {
        self.is_zero() || self.exponent >= 0
    }
}

/// Parse a JSON number literal exactly. `None` when the text is not a number.
pub fn parse(text: &str) -> Option<Decimal> {
    let bytes = text.as_bytes();
    let mut index = 0usize;
    let mut negative = false;
    if index < bytes.len() && (bytes[index] == b'-' || bytes[index] == b'+') {
        negative = bytes[index] == b'-';
        index += 1;
    }

    let mut digits = String::new();
    let mut fraction_len = 0i64;
    let mut saw_digit = false;

    while index < bytes.len() && bytes[index].is_ascii_digit() {
        digits.push(bytes[index] as char);
        saw_digit = true;
        index += 1;
    }
    if index < bytes.len() && bytes[index] == b'.' {
        index += 1;
        while index < bytes.len() && bytes[index].is_ascii_digit() {
            digits.push(bytes[index] as char);
            fraction_len += 1;
            saw_digit = true;
            index += 1;
        }
    }
    if !saw_digit {
        return None;
    }

    let mut exponent = -fraction_len;
    if index < bytes.len() && (bytes[index] == b'e' || bytes[index] == b'E') {
        index += 1;
        let mut exp_negative = false;
        if index < bytes.len() && (bytes[index] == b'-' || bytes[index] == b'+') {
            exp_negative = bytes[index] == b'-';
            index += 1;
        }
        let mut value: i64 = 0;
        let mut saw_exp_digit = false;
        while index < bytes.len() && bytes[index].is_ascii_digit() {
            // Saturating: an exponent past i64 is astronomically out of range
            // either way, and the caller only compares the result.
            value = value
                .saturating_mul(10)
                .saturating_add((bytes[index] - b'0') as i64);
            saw_exp_digit = true;
            index += 1;
        }
        if !saw_exp_digit {
            return None;
        }
        exponent = exponent.saturating_add(if exp_negative { -value } else { value });
    }
    if index != bytes.len() {
        return None;
    }

    // Strip leading zeros; they carry no value.
    let leading = digits.len() - digits.trim_start_matches('0').len();
    digits.drain(..leading);
    // Strip trailing zeros, raising the exponent to compensate — this is what
    // makes `1.50` and `15e-1` compare equal.
    while digits.ends_with('0') {
        digits.pop();
        exponent = exponent.saturating_add(1);
    }
    if digits.is_empty() {
        // Zero. Normalise the exponent so `0`, `0.0` and `0e10` all agree.
        return Some(Decimal {
            negative: false,
            digits,
            exponent: 0,
        });
    }
    Some(Decimal {
        negative,
        digits,
        exponent,
    })
}

/// The exact decimal a finite f64 denotes, via its shortest round-trip spelling.
///
/// Shortest repr is precisely what float64 preserves, which is the comparison
/// the header rule wants — not the full binary expansion.
pub fn from_f64_shortest(value: f64) -> Option<Decimal> {
    if !value.is_finite() {
        return None;
    }
    parse(&format!("{value}"))
}

/// The exact integer a finite integral f64 denotes.
pub fn from_f64_integer(value: f64) -> Option<Decimal> {
    if !value.is_finite() {
        return None;
    }
    // Explicit precision prints the exact decimal expansion rather than the
    // shortest one, which is what "which integer is this really" needs.
    parse(&format!("{value:.0}"))
}
