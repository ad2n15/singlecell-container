/// Errors returned by scfast kernels. Bindings turn these into R / Python errors.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// Inputs with inconsistent shapes or invalid sparse structure.
    #[error("invalid input: {0}")]
    InvalidInput(String),
    /// A parameter outside its valid range (e.g. more components than genes).
    #[error("invalid parameter: {0}")]
    InvalidParameter(String),
    /// Failure inside a dependency (linear algebra, clustering library).
    #[error("{context}: {message}")]
    Backend { context: &'static str, message: String },
}

pub type Result<T> = std::result::Result<T, Error>;
