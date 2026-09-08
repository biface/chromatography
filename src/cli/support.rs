//! Helpers shared by more than one `chrom-rs` command handler.
//!
//! - [`to_cli_err`]: the single conversion point from `anyhow::Error` (or
//!   any displayable error) into the [`DynamicCliError`] every
//!   `CommandHandler::execute` must return.
//! - [`resolve_source_optional`]: reads an optional `--source <role>
//!   file=...` occurrence — used by `check`, where every role is optional.
//! - [`path_to_str`]: converts a [`Path`] to `&str`, rejecting non-UTF-8
//!   paths with a clear error instead of panicking or silently lossy-ing.
//!
//! `run`'s own `resolve_source` (the *required*-source variant, with the
//! legacy-scalar fallback) and `resolve_new_outputs` stay in
//! [`crate::cli::run`] — nothing outside that module uses them.

use std::path::Path;

use anyhow::anyhow;
use dynamic_cli::error::ExecutionError;
use dynamic_cli::{DynamicCliError, ParsedArgs};

/// Wraps any displayable error into [`DynamicCliError`] via
/// [`ExecutionError::CommandFailed`].
///
/// This is the single conversion point used throughout every command
/// handler's `execute` to bridge `anyhow::Error` (and other error types)
/// into the error type required by `CommandHandler::execute`.
pub(crate) fn to_cli_err(e: impl Into<anyhow::Error>) -> DynamicCliError {
    ExecutionError::CommandFailed(e.into()).into()
}

/// Like `run`'s `resolve_source`, but consults *only* `--source <role>
/// file=...` (no legacy scalar fallback — commands other than `run` don't
/// define one) and returns `Ok(None)` rather than erroring when `role`
/// isn't given at all. Used by `check`, where every role is optional.
pub(crate) fn resolve_source_optional(
    args: &ParsedArgs,
    role: &str,
) -> anyhow::Result<Option<String>> {
    let mut via_source: Vec<String> = Vec::new();
    if let Some(occurrences) = args.get_repeated("source") {
        for occ in occurrences {
            if occ.discriminant == role {
                let file = occ.params.get("file").ok_or_else(|| {
                    anyhow!("'--source {role}' requires a 'file=...' sub-parameter")
                })?;
                via_source.push(file.clone());
            }
        }
    }
    match via_source.len() {
        0 => Ok(None),
        1 => Ok(Some(via_source.remove(0))),
        _ => Err(anyhow!(
            "'--source {role}' was given more than once — expected at most one"
        )),
    }
}

/// Converts a [`Path`] to `&str`, rejecting non-UTF-8 paths.
pub(crate) fn path_to_str(path: &Path) -> anyhow::Result<&str> {
    path.to_str()
        .ok_or_else(|| anyhow!("path '{}' contains non-UTF-8 characters", path.display()))
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    /// Builds a `ParsedArgs` with a single "source" key carrying the given
    /// (discriminant, file) occurrences, for testing `resolve_source*`
    /// without going through the full CLI parser.
    fn source_args(occurrences: &[(&str, &str)]) -> ParsedArgs {
        use dynamic_cli::parser::cli_parser::{OptionOccurrence, ParsedValue};

        let occs: Vec<OptionOccurrence> = occurrences
            .iter()
            .map(|(discriminant, file)| OptionOccurrence {
                discriminant: discriminant.to_string(),
                params: HashMap::from([("file".to_string(), file.to_string())]),
            })
            .collect();
        let mut map = HashMap::new();
        map.insert("source".to_string(), ParsedValue::Repeated(occs));
        ParsedArgs::new(map)
    }

    // ── to_cli_err ───────────────────────────────────────────────────────────

    #[test]
    fn test_to_cli_err_produces_command_failed() {
        use dynamic_cli::error::DynamicCliError;
        let err = to_cli_err(anyhow!("test error"));
        assert!(matches!(err, DynamicCliError::Execution(_)));
    }

    // ── path_to_str ──────────────────────────────────────────────────────────

    #[test]
    fn test_path_to_str_valid_utf8() {
        let p = std::path::PathBuf::from("/tmp/results.csv");
        assert_eq!(path_to_str(&p).unwrap(), "/tmp/results.csv");
    }

    #[test]
    fn test_path_to_str_rejects_non_utf8() {
        use std::ffi::OsStr;
        use std::os::unix::ffi::OsStrExt;
        let bad = OsStr::from_bytes(&[0xff, 0xfe]);
        let p = std::path::PathBuf::from(bad);
        assert!(path_to_str(&p).is_err());
    }

    // ── resolve_source_optional ─────────────────────────────────────────────

    #[test]
    fn test_resolve_source_optional_absent_is_none() {
        let args = ParsedArgs::from_scalars(HashMap::new());
        assert_eq!(resolve_source_optional(&args, "model").unwrap(), None);
    }

    #[test]
    fn test_resolve_source_optional_present_is_some() {
        let args = source_args(&[("model", "model.yml")]);
        assert_eq!(
            resolve_source_optional(&args, "model").unwrap(),
            Some("model.yml".to_string())
        );
    }

    #[test]
    fn test_resolve_source_optional_duplicate_errors() {
        let args = source_args(&[("model", "a.yml"), ("model", "b.yml")]);
        assert!(resolve_source_optional(&args, "model").is_err());
    }
}
