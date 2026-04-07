//! Error handling module
//!
//! Defines application-level errors and error handling utilities.

use thiserror::Error;

/// Application-level errors
#[derive(Error, Debug)]
pub enum AppError {
    /// PDF processing error
    #[error("PDF error: {0}")]
    Pdf(#[from] crate::pdf::PdfError),

    /// Model loading error
    #[error("Model error: {0}")]
    Model(String),

    /// Tokenization error
    #[error("Tokenizer error: {0}")]
    Tokenizer(String),

    /// Generation error
    #[error("Generation error: {0}")]
    Generation(String),

    /// IO error
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// Configuration error
    #[error("Configuration error: {0}")]
    Config(String),

    /// CLI argument error
    #[error("Argument error: {0}")]
    Argument(String),

    /// Device/GPU error
    #[error("Device error: {0}")]
    Device(String),

    /// Generic error
    #[error("{0}")]
    Other(String),
}

impl From<anyhow::Error> for AppError {
    fn from(e: anyhow::Error) -> Self {
        AppError::Other(e.to_string())
    }
}

/// Result type alias for this application
pub type AppResult<T> = Result<T, AppError>;

/// Error context extension trait
pub trait ErrorContext<T> {
    /// Add context to an error
    fn with_context<F, S>(self, f: F) -> Result<T, AppError>
    where
        F: FnOnce() -> S,
        S: Into<String>;
}

impl<T, E: std::error::Error + Send + Sync + 'static> ErrorContext<T> for Result<T, E> {
    fn with_context<F, S>(self, f: F) -> Result<T, AppError>
    where
        F: FnOnce() -> S,
        S: Into<String>,
    {
        self.map_err(|e| AppError::Other(format!("{}: {}", f().into(), e)))
    }
}

/// Helper macro for creating errors with context
#[macro_export]
macro_rules! app_error {
    ($kind:ident, $($arg:tt)*) => {
        $crate::error::AppError::$kind(format!($($arg)*))
    };
}

/// Helper macro for bail-style error handling
#[macro_export]
macro_rules! bail {
    ($kind:ident, $($arg:tt)*) => {
        return Err($crate::error::AppError::$kind(format!($($arg)*)))
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = AppError::Config("missing model path".to_string());
        assert!(err.to_string().contains("missing model path"));
    }

    #[test]
    fn test_error_from_io() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
        let app_err: AppError = io_err.into();
        assert!(matches!(app_err, AppError::Io(_)));
    }
}
