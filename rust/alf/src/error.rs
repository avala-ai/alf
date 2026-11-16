//! One error type, carrying the refusal identifier the conformance corpus names.
//!
//! Every other implementation maps its human-readable message onto an identifier
//! in its conformance runner, with a table of substring needles. That table is a
//! second source of truth and it drifts: a reworded message silently stops
//! matching and the runner reports an unnamed refusal.
//!
//! Rust carries the identifier IN the error instead. The message is for humans,
//! the identifier is for the corpus, and neither can drift from the other because
//! they are constructed together at the single place the rule is enforced.

use std::fmt;

/// A refusal, named by the identifier `tests/conformance/generator/invalid.py`
/// declares for the rule that was broken.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AlfError {
    /// The corpus identifier, e.g. `header-not-json`. Empty only for errors that
    /// no invalid case covers, which the runner surfaces as a gap rather than a
    /// pass.
    pub id: &'static str,
    pub message: String,
}

impl AlfError {
    pub fn new(id: &'static str, message: impl Into<String>) -> Self {
        AlfError {
            id,
            message: message.into(),
        }
    }
}

impl fmt::Display for AlfError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.id, self.message)
    }
}

impl std::error::Error for AlfError {}

pub type Result<T> = std::result::Result<T, AlfError>;

/// Shorthand for constructing a refusal.
macro_rules! refuse {
    ($id:expr, $($arg:tt)*) => {
        return Err($crate::error::AlfError::new($id, format!($($arg)*)))
    };
}

pub(crate) use refuse;
