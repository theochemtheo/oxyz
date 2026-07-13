//! Conformance against the extxyz comment-line value grammar: every
//! "should work" form parses to its canonical typed value. Axes are grouped
//! by scalar kind (integer, float, boolean) so each section stands alone.
//! Intentional divergences from a literal reading of the grammar are called
//! out inline with the reasoning, not just asserted silently.

use std::io::Cursor;

use oxyz_core::{ExtxyzError, Value, iter_frames_from};

/// Read a single comment-line value by wrapping it in a one-atom frame.
/// `Properties` is required on every comment line, so the harness supplies a
/// minimal fixed descriptor alongside the `key` under test.
fn read_value(src: &str) -> Result<Value, ExtxyzError> {
    let input = format!("1\nProperties=species:S:1:pos:R:3 key={src}\nX 0.0 0.0 0.0\n");
    let frame = iter_frames_from(Cursor::new(input.into_bytes()))
        .expect("reader constructs")
        .next()
        .expect("one frame")?;
    Ok(frame
        .metadata_value("key")
        .expect("key present in metadata")
        .clone())
}

/// Scalar integer forms: sign x digit-shape, plus the quoted and
/// brace-wrapped whitespace variants. All parse to Int(3) (or the signed
/// value) — leading zeros are decimal, never octal.
fn integer_cases() -> Vec<(String, Value)> {
    let signs = ["", "+", "-"];
    let digits = ["1", "12", "012"];
    let mut cases = Vec::new();
    for sign in signs {
        for d in digits {
            let src = format!("{sign}{d}");
            let expected: i64 = src.parse().expect("i64");
            cases.push((src, Value::Int(expected)));
        }
    }
    // Quoted and brace-wrapped forms of the bare integer 3, all -> Int(3).
    for wrapped in [
        "\"3\"", "\" 3\"", "\"3 \"", "\" 3 \"", "{3}", "{ 3}", "{3 }", "{ 3 }",
    ] {
        cases.push((wrapped.to_string(), Value::Int(3)));
    }
    cases
}

#[test]
fn should_work_integers() {
    for (src, expected) in integer_cases() {
        assert_eq!(read_value(&src).unwrap(), expected, "src = {src:?}");
    }
}

/// Scalar float forms: a decimal point or an exponent. Fortran `d`/`D`
/// exponents are accepted and normalised like `e`/`E`. Bare integer-shaped
/// values are deliberately excluded here — a bare integer types as Int, not
/// Real: the number is identical, but the type is finer and the distinction
/// is preserved rather than collapsed.
fn float_cases() -> Vec<(String, Value)> {
    let mut cases = Vec::new();
    for (src, val) in [
        ("1.0", 1.0),
        ("1.", 1.0),
        ("12.0", 12.0),
        ("012.0", 12.0),
        ("0.12", 0.12),
        ("00.12", 0.12),
        ("0.012", 0.012),
        (".012", 0.012),
    ] {
        for sign in ["", "+", "-"] {
            let s = format!("{sign}{src}");
            let signed = if sign == "-" { -val } else { val };
            cases.push((s, Value::Real(signed)));
        }
    }
    // Exponent combinations on -12.0: e/E/d/D are all equivalent.
    for exp_char in ["e", "E", "d", "D"] {
        for exp_sign in ["", "+", "-"] {
            for mag in ["0", "2", "02", "12"] {
                let src = format!("-12.0{exp_char}{exp_sign}{mag}");
                let reference = format!("-12.0e{exp_sign}{mag}");
                let expected: f64 = reference.parse().expect("f64");
                cases.push((src, Value::Real(expected)));
            }
        }
    }
    cases
}

#[test]
fn should_work_floats() {
    for (src, expected) in float_cases() {
        assert_eq!(read_value(&src).unwrap(), expected, "src = {src:?}");
    }
}

/// Quoting preserves a string's interior whitespace: a quoted single word
/// keeps its padding as a `Str`, while a quoted number still trims and types
/// numerically. Classification runs on the trimmed token, but the string
/// fallback keeps the original quote-stripped content.
#[test]
fn quoted_strings_preserve_padding_but_numbers_still_trim() {
    assert_eq!(
        read_value("\" hello \"").unwrap(),
        Value::Str(" hello ".into())
    );
    assert_eq!(read_value("\" 3 \"").unwrap(), Value::Int(3));
}

/// Scalar boolean forms: t/T/true/True/TRUE -> true; f/F/false/False/FALSE
/// -> false.
#[test]
fn should_work_booleans() {
    for src in ["t", "T", "true", "True", "TRUE"] {
        assert_eq!(read_value(src).unwrap(), Value::Bool(true), "src = {src:?}");
    }
    for src in ["f", "F", "false", "False", "FALSE"] {
        assert_eq!(
            read_value(src).unwrap(),
            Value::Bool(false),
            "src = {src:?}"
        );
    }
}
