use std::fmt;

/// Why code generation or code memory failed.
#[derive(Clone, Debug)]
pub struct Error {
    kind: ErrorKind,
    message: String,
}

/// What kind of failure an `Error` is.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ErrorKind {
    /// The function is malformed.
    Verify,
    /// The function uses something the encoder for this target does not handle.
    Unsupported,
    /// Mapping or filling code or data memory failed.
    Memory,
}

impl Error {
    pub(crate) fn new(kind: ErrorKind, message: impl Into<String>) -> Error {
        Error { kind, message: message.into() }
    }

    /// What kind of failure this is.
    pub fn kind(&self) -> ErrorKind {
        self.kind
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for Error {}
