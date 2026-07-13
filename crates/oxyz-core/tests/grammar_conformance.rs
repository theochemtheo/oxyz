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

/// kv_tests "scalar string": bare strings from the printable set, and quoted
/// strings. Single quotes are ordinary bare characters (only `"` quotes; see
/// the singlequote_metadata.extxyz fixture), so 'abc' is a valid bare string.
#[test]
fn should_work_strings() {
    for src in ["TRuE", "1.3k7", "-2.75e", "+2.75e-"] {
        assert_eq!(
            read_value(src).unwrap(),
            Value::Str(src.into()),
            "src = {src:?}"
        );
    }
    // Quoted string with spaces keeps its interior; quoting is stripped.
    assert_eq!(
        read_value("\"line one\"").unwrap(),
        Value::Str("line one".into())
    );
    // Single-quoted: a bare string, quotes retained (intentional divergence).
    assert_eq!(read_value("'abc'").unwrap(), Value::Str("'abc'".into()));
}

/// kv_tests "1-d array": whitespace (bare, quoted, braced) and new-style [ , ].
#[test]
fn should_work_1d_arrays() {
    assert_eq!(
        read_value("\"1 2 3\"").unwrap(),
        Value::IntArray(vec![1, 2, 3])
    );
    assert_eq!(
        read_value("{1 2 3}").unwrap(),
        Value::IntArray(vec![1, 2, 3])
    );
    // Bare (unquoted) interior whitespace would split into further key=value
    // pairs on a real comment line, so this form is only reachable quoted —
    // matching the quoted int case above.
    assert_eq!(
        read_value("\"1.0 2.0 3.0\"").unwrap(),
        Value::RealArray(vec![1.0, 2.0, 3.0])
    );
    assert_eq!(
        read_value("[a,b]").unwrap(),
        Value::StrArray(vec!["a".into(), "b".into()])
    );
    assert_eq!(
        read_value("[ \"a\", \"b\" ]").unwrap(),
        Value::StrArray(vec!["a".into(), "b".into()])
    );
    // Quoted commas/brackets inside string elements do not split.
    assert_eq!(
        read_value("[ \"a, b\", \"c]\" ]").unwrap(),
        Value::StrArray(vec!["a, b".into(), "c]".into()])
    );
    assert_eq!(
        read_value("[ T, F, bob ]").unwrap(),
        Value::StrArray(vec!["T".into(), "F".into(), "bob".into()])
    );
    // A `[` *inside* a quoted string element is part of the string, not a
    // nested row: the array stays 1-D. The 1-D/2-D decision runs on the
    // quote-aware split, so a quoted element like `"a[b"` starts with `"`,
    // never `[`.
    assert_eq!(
        read_value("[ \"a[b\", \"c\" ]").unwrap(),
        Value::StrArray(vec!["a[b".into(), "c".into()])
    );
    assert_eq!(
        read_value("[ \"path[0]\", \"path[1]\" ]").unwrap(),
        Value::StrArray(vec!["path[0]".into(), "path[1]".into()])
    );
}

/// kv_tests "2-d array": nested new-style brackets -> flat buffer + shape.
#[test]
fn should_work_2d_arrays() {
    assert_eq!(
        read_value("[[1,2],[3,4]]").unwrap(),
        Value::IntArray2D {
            rows: 2,
            cols: 2,
            data: vec![1, 2, 3, 4]
        }
    );
    assert_eq!(
        read_value("[[1,2],[3.0,4]]").unwrap(),
        Value::RealArray2D {
            rows: 2,
            cols: 2,
            data: vec![1.0, 2.0, 3.0, 4.0]
        }
    );
    // A literal `[` inside a quoted string *cell* of a 2-D array is part of
    // that string, not a nested (3-D) row — the row-level guard must be as
    // quote-aware as the top-level 1-D/2-D one.
    assert_eq!(
        read_value("[[\"a[b\",\"c\"],[\"d\",\"e\"]]").unwrap(),
        Value::StrArray2D {
            rows: 2,
            cols: 2,
            data: vec!["a[b".into(), "c".into(), "d".into(), "e".into()]
        }
    );
}

/// kv_tests "ones that should fail": malformed values must be Err. Cases that
/// oxyz intentionally accepts (single-quoted bare strings, bare-int-as-int) are
/// asserted as their conformant value elsewhere, with a decision on record.
#[test]
fn should_fail_cases() {
    // These are malformed *values* — the strict typer must reject them.
    let fail = [
        "\"abc'",       // almost quoted: opens " never closes → splitter Err
        "\"abc\"def",   // trailing junk after close quote → stray bare key Err
        "[ 1, 2, ]",    // trailing comma → empty element
        "[ , 2, 3 ]",   // leading comma → empty element
        "[[1,2],[3]]",  // ragged 2-D
        "[[1,2][1,2]]", // 2-D missing separating comma
    ];
    for src in fail {
        assert!(read_value(src).is_err(), "expected Err for src = {src:?}");
    }
    // Unbalanced groups → Err at the splitter.
    assert!(read_value("{1, 2").is_err());
    assert!(read_value("[1, 2").is_err());
}

/// The "no key-value =" fail class (a comment token with no `=`) is a splitter
/// concern, not a value-typer one — `read_value` always supplies `key=`, so it
/// cannot express it. Assert it directly against the splitter.
#[test]
fn should_fail_bare_key_without_equals() {
    use std::io::Cursor;
    // Comment line is a bare token with no '=' at all.
    let input = "1\nloneword\nX 0.0 0.0 0.0\n";
    let result = iter_frames_from(Cursor::new(input.as_bytes()))
        .unwrap()
        .next()
        .unwrap();
    assert!(
        result.is_err(),
        "bare comment token with no '=' must be an error"
    );
}
