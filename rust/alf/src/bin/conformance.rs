//! Reference runner: read one ALF file, print the canonical answer, exit 0.
//!
//! The contract every language's runner implements.
//!
//! A REFUSAL IS A RESULT. When the file is invalid this prints
//! `{"refused": "<identifier>"}` and still exits 0, so the harness diffs it like
//! any other answer. Exiting non-zero would collapse "refused correctly" and
//! "fell over" into one outcome, and telling those apart is what the invalid
//! corpus is for.
//!
//! Unlike the other runners there is NO substring table here. The identifier
//! travels inside `AlfError`, constructed at the site that enforces the rule, so
//! a reworded message cannot silently stop matching.

use std::process::ExitCode;

use alf::header::Value;
use alf::{read_alf, AlfFile, Channel};

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: alf-conformance <path> [extension]");
        return ExitCode::FAILURE;
    }
    let path = &args[1];
    let extension = args.get(2).map(String::as_str).unwrap_or("");

    match read_alf(path, extension) {
        Ok(file) => {
            println!("{}", canonical(&file));
            ExitCode::SUCCESS
        }
        Err(err) => {
            // An unnamed refusal is a GAP, not a pass: it matches no expectation
            // and fails with a readable diff rather than quietly counting as
            // success. `id` is empty exactly when no invalid case covers it.
            println!("{{\"refused\": {}}}", json_string(err.id));
            ExitCode::SUCCESS
        }
    }
}

fn canonical(file: &AlfFile) -> String {
    let mut fields: Vec<String> = Vec::new();
    fields.push(format!("\"version\": {}", file.version));
    fields.push(format!(
        "\"extension\": {}",
        json_string(file.channel.as_str())
    ));

    match file.channel {
        Channel::Alp => {
            fields.push(format!(
                "\"codec\": {}",
                file.geom_codec.map(json_string).unwrap_or("null".into())
            ));
            let points: Vec<String> = file
                .xyz
                .iter()
                .map(|point| {
                    let comps: Vec<String> = point.iter().map(|c| component(*c)).collect();
                    format!("[{}]", comps.join(", "))
                })
                .collect();
            fields.push(format!("\"points\": [{}]", points.join(", ")));
            fields.push(format!(
                "\"max_geom_error_m\": {}",
                match file.max_geom_error_m {
                    None => "null".to_string(),
                    Some(v) => number(round_decimals(v, 12)),
                }
            ));
            fields.push(format!(
                "\"frame_convention\": {}",
                match &file.frame_convention {
                    None => "null".to_string(),
                    Some(text) => json_string(text.as_str()),
                }
            ));
            if let Some(q) = file.frame_rotation {
                let members: Vec<String> = [("w", q.w), ("x", q.x), ("y", q.y), ("z", q.z)]
                    .iter()
                    .map(|(axis, value)| format!("\"{axis}\": {}", component(*value)))
                    .collect();
                fields.push(format!("\"frame_rotation\": {{{}}}", members.join(", ")));
            }
            let anchors: Vec<String> = ["device_position", "world_centroid_epsg3857"]
                .iter()
                .filter_map(|key| {
                    file.metadata
                        .get(*key)
                        .map(|value| format!("{}: {}", json_string(key), emit(value)))
                })
                .collect();
            if !anchors.is_empty() {
                fields.push(format!("\"anchors\": {{{}}}", anchors.join(", ")));
            }
        }
        Channel::Ali => {
            let values: Vec<String> = file.intensity.iter().map(|v| v.to_string()).collect();
            fields.push(format!("\"intensity\": [{}]", values.join(", ")));
        }
        Channel::Alc => {
            let palette: Vec<String> = file.palette.iter().map(|c| json_string(c)).collect();
            fields.push(format!("\"palette\": [{}]", palette.join(", ")));
            let indices: Vec<String> = file.palette_indices.iter().map(|v| v.to_string()).collect();
            fields.push(format!("\"palette_indices\": [{}]", indices.join(", ")));
        }
        Channel::All => {
            fields.push(format!(
                "\"format_version\": {}",
                file.format_version
                    .map(|v| v.to_string())
                    .unwrap_or("null".into())
            ));
            let cats: Vec<String> = file.categories.iter().map(|v| v.to_string()).collect();
            fields.push(format!("\"categories\": [{}]", cats.join(", ")));
            fields.push(format!(
                "\"instance_ids\": {}",
                match &file.instance_ids {
                    None => "null".to_string(),
                    Some(ids) => format!(
                        "[{}]",
                        ids.iter()
                            .map(|v| v.to_string())
                            .collect::<Vec<_>>()
                            .join(", ")
                    ),
                }
            ));
            fields.push(format!(
                "\"instance_map\": {}",
                match &file.instance_map {
                    None => "null".to_string(),
                    Some(map) => {
                        let entries: Vec<String> = map
                            .iter()
                            .map(|(k, v)| format!("{}: {}", json_string(k), emit(v)))
                            .collect();
                        format!("{{{}}}", entries.join(", "))
                    }
                }
            ));
        }
        Channel::Alx => {
            let kept: Vec<String> = file.kept_indices.iter().map(|v| v.to_string()).collect();
            fields.push(format!("\"kept_indices\": [{}]", kept.join(", ")));
            fields.push(format!(
                "\"source_total_points\": {}",
                file.metadata
                    .get("_source_total_points")
                    .map(emit)
                    .unwrap_or("null".into())
            ));
        }
    }
    format!("{{{}}}", fields.join(", "))
}

/// A coordinate, spelled the way every other runner spells it.
fn component(value: f64) -> String {
    // JSON HAS NO NaN, so the corpus spells non-finite components as STRINGS,
    // and keeps the three distinct: a reader that turned +Infinity into NaN
    // would have changed the file's meaning, and `null` could not say so.
    if value.is_nan() {
        return "\"NaN\"".to_string();
    }
    if value.is_infinite() {
        return if value > 0.0 {
            "\"Infinity\"".to_string()
        } else {
            "\"-Infinity\"".to_string()
        };
    }
    number(round_decimals(value, 6))
}

/// `round(value, digits)`, matching Python by rounding the EXACT binary value.
///
/// Not `(value * 10^digits).round() / 10^digits`. That multiply is itself
/// inexact: the double nearest `2.5e-06` sits just ABOVE the decimal midpoint,
/// so Python rounds it up to `3e-06`, while scaling first lands on exactly `2.5`
/// and any tie rule then answers `2e-06` — off by one in the last place, on a
/// coordinate. Formatting with an explicit precision rounds the true value
/// correctly, and parsing back recovers the rounded double.
fn round_decimals(value: f64, digits: u32) -> f64 {
    if !value.is_finite() {
        return value;
    }
    format!("{:.*}", digits as usize, value)
        .parse()
        .unwrap_or(value)
}

/// A float spelled exactly as `json.dumps` spells it.
///
/// Python switches to scientific notation outside `-4 <= exp10 < 16`, pads the
/// exponent to two digits and always signs it — `1.8446744073709552e+19`,
/// `3e-06`. Rust's `{}` is positional at every magnitude, so the two disagree on
/// precisely the wide-integer and tiny-scale cases this corpus is built from.
fn number(value: f64) -> String {
    if value == 0.0 {
        return "0.0".to_string();
    }
    // `{:e}` gives the shortest round-trip digits plus a base-10 exponent, which
    // is the same pair Python starts from.
    let scientific = format!("{value:e}");
    let (mantissa, exponent) = scientific
        .split_once('e')
        .expect("{:e} always has an exponent");
    let exponent: i32 = exponent.parse().expect("{:e} exponent is an integer");

    if (-4..16).contains(&exponent) {
        let text = format!("{value}");
        if text.contains('.') || text.contains('e') {
            return text;
        }
        return format!("{text}.0");
    }
    let sign = if exponent < 0 { '-' } else { '+' };
    format!("{}e{}{:02}", mantissa, sign, exponent.abs())
}

fn emit(value: &Value) -> String {
    match value {
        Value::Null => "null".to_string(),
        Value::Bool(b) => b.to_string(),
        Value::Number(parsed, token) => {
            if token.contains('.') || token.contains('e') || token.contains('E') {
                number(*parsed)
            } else {
                token.clone()
            }
        }
        Value::String(s) => json_string(s),
        Value::Array(items) => format!(
            "[{}]",
            items.iter().map(emit).collect::<Vec<_>>().join(", ")
        ),
        Value::Object(map) => format!(
            "{{{}}}",
            map.iter()
                .map(|(k, v)| format!("{}: {}", json_string(k), emit(v)))
                .collect::<Vec<_>>()
                .join(", ")
        ),
    }
}

fn json_string(text: &str) -> String {
    let mut out = String::with_capacity(text.len() + 2);
    out.push('"');
    for ch in text.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}
