//! The JSON header: parsed by hand, because the rules are about the TEXT.
//!
//! `serde_json` cannot express three things the corpus requires, and each of
//! them is a rule about the bytes rather than the decoded value:
//!
//! - **Duplicate keys.** A map has already lost them by the time serde hands one
//!   over; last-write-wins is indistinguishable from a well-formed header.
//! - **Nesting depth, enforced BEFORE recursion.** The spec is explicit that a
//!   reader refuses deep input rather than discovering it by recursing into it
//!   (`spec/format.md` §4). A parser that recurses first and counts afterwards
//!   has already risked the stack on hostile input.
//! - **Number exactness.** `_total_points: 9007199254740993` parses to a float
//!   that is a DIFFERENT integer. Catching that needs the original token, not the
//!   value it became.
//!
//! So this is a scanner over the raw bytes that produces a value tree while
//! enforcing the rules in the same pass.

use std::collections::BTreeMap;

use crate::error::{refuse, AlfError, Result};

/// `spec/format.md` §4. A larger DECLARED header is refused even when the bytes
/// are all present — the point is to bound work before doing it.
pub const MAX_HEADER_BYTES: usize = 10_000_000;

/// `spec/format.md` §4. The root object counts as depth 1.
pub const MAX_NESTING_DEPTH: usize = 256;

/// A header value. `Number` keeps its original token so exactness is checkable.
#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    Null,
    Bool(bool),
    /// (parsed f64, original token). The token is what makes an inexact integer
    /// detectable after the fact.
    Number(f64, String),
    String(String),
    Array(Vec<Value>),
    /// BTreeMap for deterministic emission; duplicates are refused during the
    /// parse, so nothing is lost by collapsing to a map afterwards.
    Object(BTreeMap<String, Value>),
}

impl Value {
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Value::Number(v, _) => Some(*v),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::String(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_array(&self) -> Option<&Vec<Value>> {
        match self {
            Value::Array(a) => Some(a),
            _ => None,
        }
    }

    pub fn as_object(&self) -> Option<&BTreeMap<String, Value>> {
        match self {
            Value::Object(o) => Some(o),
            _ => None,
        }
    }

    /// The original numeric token, for exactness checks.
    pub fn number_token(&self) -> Option<&str> {
        match self {
            Value::Number(_, token) => Some(token),
            _ => None,
        }
    }
}

/// Parse the header bytes into an object, enforcing every textual rule.
pub fn parse(bytes: &[u8]) -> Result<BTreeMap<String, Value>> {
    let text = std::str::from_utf8(bytes).map_err(|_| {
        AlfError::new(
            "header-not-utf8",
            "Header is not valid UTF-8; the header is a UTF-8 JSON object",
        )
    })?;

    // A zero-length header is a header: it states no metadata, which is a thing
    // a small `.ali` legitimately has nothing to say beyond its payload.
    // Refusing it as "not an object" would reject a conforming file for being
    // terse (`ali-zero-length-header`).
    if text.is_empty() {
        return Ok(BTreeMap::new());
    }

    // Depth FIRST, over the raw text, before anything recurses. A hostile header
    // is refused on a linear scan rather than on the way back up a blown stack.
    check_depth(text)?;

    let mut parser = Parser {
        bytes: text.as_bytes(),
        pos: 0,
    };
    parser.skip_whitespace();
    if parser.peek() != Some(b'{') {
        refuse!(
            "header-not-an-object",
            "Header must be a JSON object, got a {} at the top level",
            parser.describe_next()
        );
    }
    let value = parser.parse_value(0)?;
    parser.skip_whitespace();
    if parser.pos != parser.bytes.len() {
        refuse!(
            "header-not-json",
            "Header is not valid JSON: trailing content after the top-level object"
        );
    }
    match value {
        Value::Object(map) => Ok(map),
        _ => refuse!("header-not-an-object", "Header must be a JSON object"),
    }
}

/// Linear scan for collection nesting, ignoring braces inside strings.
fn check_depth(text: &str) -> Result<()> {
    let bytes = text.as_bytes();
    let mut depth = 0usize;
    let mut in_string = false;
    let mut escaped = false;
    for &byte in bytes {
        if in_string {
            if escaped {
                escaped = false;
            } else if byte == b'\\' {
                escaped = true;
            } else if byte == b'"' {
                in_string = false;
            }
            continue;
        }
        match byte {
            b'"' => in_string = true,
            b'{' | b'[' => {
                depth += 1;
                if depth > MAX_NESTING_DEPTH {
                    refuse!(
                        "header-nesting-too-deep",
                        "header-nesting-too-deep: more than {MAX_NESTING_DEPTH} nested \
                         objects/arrays"
                    );
                }
            }
            b'}' | b']' => depth = depth.saturating_sub(1),
            _ => {}
        }
    }
    Ok(())
}

struct Parser<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Parser<'a> {
    fn peek(&self) -> Option<u8> {
        self.bytes.get(self.pos).copied()
    }

    fn describe_next(&self) -> &'static str {
        match self.peek() {
            Some(b'[') => "array",
            Some(b'"') => "string",
            Some(b't') | Some(b'f') => "boolean",
            Some(b'n') => "null",
            Some(c) if c == b'-' || c.is_ascii_digit() => "number",
            _ => "value",
        }
    }

    fn skip_whitespace(&mut self) {
        while let Some(byte) = self.peek() {
            if matches!(byte, b' ' | b'\t' | b'\n' | b'\r') {
                self.pos += 1;
            } else {
                break;
            }
        }
    }

    fn expect(&mut self, byte: u8) -> Result<()> {
        if self.peek() == Some(byte) {
            self.pos += 1;
            Ok(())
        } else {
            refuse!(
                "header-not-json",
                "Header is not valid JSON: expected {:?} at byte {}",
                byte as char,
                self.pos
            )
        }
    }

    fn parse_value(&mut self, depth: usize) -> Result<Value> {
        self.skip_whitespace();
        match self.peek() {
            Some(b'{') => self.parse_object(depth),
            Some(b'[') => self.parse_array(depth),
            Some(b'"') => Ok(Value::String(self.parse_string()?)),
            Some(b't') => self.parse_literal("true", Value::Bool(true)),
            Some(b'f') => self.parse_literal("false", Value::Bool(false)),
            Some(b'n') => self.parse_literal("null", Value::Null),
            Some(c) if c == b'-' || c.is_ascii_digit() => self.parse_number(),
            _ => refuse!(
                "header-not-json",
                "Header is not valid JSON: unexpected byte at {}",
                self.pos
            ),
        }
    }

    fn parse_literal(&mut self, word: &str, value: Value) -> Result<Value> {
        if self.bytes[self.pos..].starts_with(word.as_bytes()) {
            self.pos += word.len();
            Ok(value)
        } else {
            refuse!(
                "header-not-json",
                "Header is not valid JSON: bad literal at byte {}",
                self.pos
            )
        }
    }

    fn parse_object(&mut self, depth: usize) -> Result<Value> {
        self.expect(b'{')?;
        let mut map: BTreeMap<String, Value> = BTreeMap::new();
        self.skip_whitespace();
        if self.peek() == Some(b'}') {
            self.pos += 1;
            return Ok(Value::Object(map));
        }
        loop {
            self.skip_whitespace();
            if self.peek() != Some(b'"') {
                refuse!(
                    "header-not-json",
                    "Header is not valid JSON: object key must be a string at byte {}",
                    self.pos
                );
            }
            let key = self.parse_string()?;
            // The reason this parser exists. A map cannot represent the
            // duplicate, so the check has to happen while the text is still
            // being read.
            if map.contains_key(&key) {
                refuse!(
                    "header-key-duplicate",
                    "Header contains a duplicate JSON object key {key:?}; the two values \
                     disagree about what the file means and last-write-wins would pick one \
                     silently"
                );
            }
            self.skip_whitespace();
            self.expect(b':')?;
            let value = self.parse_value(depth + 1)?;
            map.insert(key, value);
            self.skip_whitespace();
            match self.peek() {
                Some(b',') => {
                    self.pos += 1;
                }
                Some(b'}') => {
                    self.pos += 1;
                    return Ok(Value::Object(map));
                }
                _ => refuse!(
                    "header-not-json",
                    "Header is not valid JSON: expected ',' or '}}' at byte {}",
                    self.pos
                ),
            }
        }
    }

    fn parse_array(&mut self, depth: usize) -> Result<Value> {
        self.expect(b'[')?;
        let mut items = Vec::new();
        self.skip_whitespace();
        if self.peek() == Some(b']') {
            self.pos += 1;
            return Ok(Value::Array(items));
        }
        loop {
            items.push(self.parse_value(depth + 1)?);
            self.skip_whitespace();
            match self.peek() {
                Some(b',') => {
                    self.pos += 1;
                }
                Some(b']') => {
                    self.pos += 1;
                    return Ok(Value::Array(items));
                }
                _ => refuse!(
                    "header-not-json",
                    "Header is not valid JSON: expected ',' or ']' at byte {}",
                    self.pos
                ),
            }
        }
    }

    fn parse_string(&mut self) -> Result<String> {
        self.expect(b'"')?;
        let mut out = String::new();
        loop {
            let byte = match self.peek() {
                Some(b) => b,
                None => refuse!(
                    "header-not-json",
                    "Header is not valid JSON: unterminated string"
                ),
            };
            self.pos += 1;
            match byte {
                b'"' => return Ok(out),
                b'\\' => {
                    let esc = match self.peek() {
                        Some(b) => b,
                        None => refuse!(
                            "header-not-json",
                            "Header is not valid JSON: unterminated escape"
                        ),
                    };
                    self.pos += 1;
                    match esc {
                        b'"' => out.push('"'),
                        b'\\' => out.push('\\'),
                        b'/' => out.push('/'),
                        b'b' => out.push('\u{0008}'),
                        b'f' => out.push('\u{000C}'),
                        b'n' => out.push('\n'),
                        b'r' => out.push('\r'),
                        b't' => out.push('\t'),
                        b'u' => out.push(self.parse_unicode_escape()?),
                        _ => refuse!(
                            "header-not-json",
                            "Header is not valid JSON: unknown escape \\{}",
                            esc as char
                        ),
                    }
                }
                _ => {
                    // Multi-byte UTF-8 passes through: the slice was validated as
                    // UTF-8 up front, so copying the raw bytes cannot split a
                    // code point.
                    let start = self.pos - 1;
                    let len = utf8_len(byte);
                    if len > 1 {
                        self.pos = start + len;
                    }
                    out.push_str(std::str::from_utf8(&self.bytes[start..self.pos]).map_err(
                        |_| AlfError::new("header-not-utf8", "Header is not valid UTF-8"),
                    )?);
                }
            }
        }
    }

    fn parse_unicode_escape(&mut self) -> Result<char> {
        if self.pos + 4 > self.bytes.len() {
            refuse!(
                "header-not-json",
                "Header is not valid JSON: truncated \\u escape"
            );
        }
        let hex = std::str::from_utf8(&self.bytes[self.pos..self.pos + 4])
            .map_err(|_| AlfError::new("header-not-json", "Header is not valid JSON"))?;
        let code = u32::from_str_radix(hex, 16)
            .map_err(|_| AlfError::new("header-not-json", "Header is not valid JSON: bad \\u"))?;
        self.pos += 4;
        // Surrogate pair: JSON spells astral characters as two escapes.
        if (0xD800..0xDC00).contains(&code) && self.bytes[self.pos..].starts_with(b"\\u") {
            self.pos += 2;
            let low_hex = std::str::from_utf8(&self.bytes[self.pos..self.pos + 4])
                .map_err(|_| AlfError::new("header-not-json", "Header is not valid JSON"))?;
            let low = u32::from_str_radix(low_hex, 16).map_err(|_| {
                AlfError::new("header-not-json", "Header is not valid JSON: bad \\u")
            })?;
            self.pos += 4;
            let combined = 0x10000 + ((code - 0xD800) << 10) + (low - 0xDC00);
            return char::from_u32(combined).ok_or_else(|| {
                AlfError::new("header-not-json", "Header is not valid JSON: bad surrogate")
            });
        }
        char::from_u32(code).ok_or_else(|| {
            AlfError::new(
                "header-not-json",
                "Header is not valid JSON: lone surrogate",
            )
        })
    }

    fn parse_number(&mut self) -> Result<Value> {
        let start = self.pos;
        if self.peek() == Some(b'-') {
            self.pos += 1;
        }
        while let Some(byte) = self.peek() {
            if byte.is_ascii_digit() || matches!(byte, b'.' | b'e' | b'E' | b'+' | b'-') {
                self.pos += 1;
            } else {
                break;
            }
        }
        let token = std::str::from_utf8(&self.bytes[start..self.pos])
            .map_err(|_| AlfError::new("header-not-json", "Header is not valid JSON"))?
            .to_string();
        let parsed: f64 = token.parse().map_err(|_| {
            AlfError::new(
                "header-not-json",
                format!("Header is not valid JSON: bad number {token:?}"),
            )
        })?;
        // Non-finite is NOT refused here. `1e309` overflows f64, but which rule
        // it breaks depends on WHERE it sits: under `device_position` it is
        // `geom-number-inexact`, anywhere else it is `header-not-json`. The
        // parser cannot know, so it keeps the value and lets the validators that
        // do know decide, in that order.
        Ok(Value::Number(parsed, token))
    }
}

fn utf8_len(first: u8) -> usize {
    if first < 0x80 {
        1
    } else if first >> 5 == 0b110 {
        2
    } else if first >> 4 == 0b1110 {
        3
    } else {
        4
    }
}
