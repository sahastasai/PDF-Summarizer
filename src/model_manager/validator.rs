//! Model Validator
//!
//! Validates downloaded LLaMA models for completeness and integrity.

use anyhow::{Context, Result};
use std::fs;
use std::path::Path;
use tracing::{debug, info, warn};

/// Required files for a valid LLaMA model
const REQUIRED_FILES: &[&str] = &["config.json", "tokenizer.json"];

/// Optional but recommended files
const OPTIONAL_FILES: &[&str] = &[
    "tokenizer_config.json",
    "special_tokens_map.json",
    "generation_config.json",
];

/// Model validator
pub struct ModelValidator {
    /// Strict mode - fail on warnings
    strict: bool,
}

impl Default for ModelValidator {
    fn default() -> Self {
        Self::new()
    }
}

impl ModelValidator {
    /// Create a new validator
    pub fn new() -> Self {
        Self { strict: false }
    }

    /// Enable strict validation mode
    pub fn strict(mut self) -> Self {
        self.strict = true;
        self
    }

    /// Validate a model directory
    pub fn validate(&self, model_path: &Path) -> Result<ValidationResult> {
        info!("Validating model at: {:?}", model_path);

        if !model_path.exists() {
            anyhow::bail!("Model path does not exist: {:?}", model_path);
        }

        if !model_path.is_dir() {
            anyhow::bail!("Model path is not a directory: {:?}", model_path);
        }

        let mut result = ValidationResult::default();

        // Check required files
        for &file in REQUIRED_FILES {
            let file_path = model_path.join(file);
            if file_path.exists() {
                debug!("Found required file: {}", file);
                result.files_present.push(file.to_string());

                // Validate file is not empty
                let metadata = fs::metadata(&file_path)?;
                if metadata.len() == 0 {
                    result
                        .errors
                        .push(format!("Required file is empty: {}", file));
                }
            } else {
                result.files_missing.push(file.to_string());
                result
                    .errors
                    .push(format!("Missing required file: {}", file));
            }
        }

        // Check optional files
        for &file in OPTIONAL_FILES {
            let file_path = model_path.join(file);
            if file_path.exists() {
                debug!("Found optional file: {}", file);
                result.files_present.push(file.to_string());
            } else {
                result
                    .warnings
                    .push(format!("Missing optional file: {}", file));
            }
        }

        // Check for weight files (safetensors)
        let has_safetensors = self.check_weight_files(model_path, &mut result)?;

        if !has_safetensors {
            result
                .errors
                .push("No model weight files found (.safetensors)".to_string());
        }

        // Validate config.json
        self.validate_config(model_path, &mut result)?;

        // Validate tokenizer
        self.validate_tokenizer(model_path, &mut result)?;

        // Calculate total size
        result.total_size = self.calculate_size(model_path)?;

        // Determine overall validity
        result.is_valid = result.errors.is_empty();

        if !result.is_valid {
            let errors = result.errors.join("\n  - ");
            anyhow::bail!("Model validation failed:\n  - {}", errors);
        }

        if self.strict && !result.warnings.is_empty() {
            let warnings = result.warnings.join("\n  - ");
            anyhow::bail!("Model validation failed (strict mode):\n  - {}", warnings);
        }

        for warning in &result.warnings {
            warn!("{}", warning);
        }

        info!("Model validation successful");
        info!(
            "  Total size: {:.2} GB",
            result.total_size as f64 / (1024.0 * 1024.0 * 1024.0)
        );
        info!("  Weight files: {}", result.weight_files.len());

        Ok(result)
    }

    /// Check for weight files
    fn check_weight_files(&self, model_path: &Path, result: &mut ValidationResult) -> Result<bool> {
        let mut found_weights = false;

        for entry in fs::read_dir(model_path)? {
            let entry = entry?;
            let path = entry.path();

            if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                if name.ends_with(".safetensors") {
                    debug!("Found weight file: {}", name);
                    result.weight_files.push(name.to_string());
                    result.files_present.push(name.to_string());
                    found_weights = true;
                }
            }
        }

        // Sort weight files for consistent ordering
        result.weight_files.sort();

        Ok(found_weights)
    }

    /// Validate config.json
    fn validate_config(&self, model_path: &Path, result: &mut ValidationResult) -> Result<()> {
        let config_path = model_path.join("config.json");

        if !config_path.exists() {
            return Ok(()); // Already reported as error
        }

        let content = fs::read_to_string(&config_path).context("Failed to read config.json")?;

        let config: serde_json::Value =
            serde_json::from_str(&content).context("Failed to parse config.json")?;

        // Check for essential config fields
        let required_fields = [
            "hidden_size",
            "num_hidden_layers",
            "num_attention_heads",
            "vocab_size",
        ];

        for field in required_fields {
            if config.get(field).is_none() {
                result
                    .warnings
                    .push(format!("config.json missing field: {}", field));
            }
        }

        // Extract model type
        if let Some(model_type) = config.get("model_type").and_then(|v| v.as_str()) {
            result.model_type = Some(model_type.to_string());
            debug!("Model type: {}", model_type);

            if !model_type.to_lowercase().contains("llama") {
                result.warnings.push(format!(
                    "Model type '{}' may not be compatible with LLaMA implementation",
                    model_type
                ));
            }
        }

        Ok(())
    }

    /// Validate tokenizer files
    fn validate_tokenizer(&self, model_path: &Path, _result: &mut ValidationResult) -> Result<()> {
        let tokenizer_path = model_path.join("tokenizer.json");

        if !tokenizer_path.exists() {
            return Ok(()); // Already reported as error
        }

        let content =
            fs::read_to_string(&tokenizer_path).context("Failed to read tokenizer.json")?;

        // Just check it's valid JSON
        let _: serde_json::Value =
            serde_json::from_str(&content).context("Failed to parse tokenizer.json")?;

        debug!("Tokenizer validation successful");

        Ok(())
    }

    /// Calculate total model size
    fn calculate_size(&self, model_path: &Path) -> Result<u64> {
        let mut total = 0u64;

        for entry in fs::read_dir(model_path)? {
            let entry = entry?;
            let metadata = entry.metadata()?;

            if metadata.is_file() {
                total += metadata.len();
            }
        }

        Ok(total)
    }
}

/// Validation result
#[derive(Debug, Default)]
pub struct ValidationResult {
    /// Whether the model is valid
    pub is_valid: bool,
    /// Files that were found
    pub files_present: Vec<String>,
    /// Required files that are missing
    pub files_missing: Vec<String>,
    /// Weight files found
    pub weight_files: Vec<String>,
    /// Model type from config
    pub model_type: Option<String>,
    /// Total size in bytes
    pub total_size: u64,
    /// Validation errors
    pub errors: Vec<String>,
    /// Validation warnings
    pub warnings: Vec<String>,
}

impl ValidationResult {
    /// Check if validation passed
    pub fn is_ok(&self) -> bool {
        self.is_valid && self.errors.is_empty()
    }

    /// Get a summary string
    pub fn summary(&self) -> String {
        if self.is_valid {
            format!(
                "Valid model with {} weight files ({:.2} GB)",
                self.weight_files.len(),
                self.total_size as f64 / (1024.0 * 1024.0 * 1024.0)
            )
        } else {
            format!("Invalid model: {} errors", self.errors.len())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::tempdir;

    #[test]
    fn test_validator_missing_files() {
        let temp = tempdir().unwrap();
        let validator = ModelValidator::new();

        let result = validator.validate(temp.path());
        assert!(result.is_err());
    }

    #[test]
    fn test_validator_with_required_files() {
        let temp = tempdir().unwrap();

        // Create minimal required files
        let config = r#"{"hidden_size": 4096, "num_hidden_layers": 32, "num_attention_heads": 32, "vocab_size": 128256}"#;
        let tokenizer = r#"{"version": "1.0"}"#;

        fs::write(temp.path().join("config.json"), config).unwrap();
        fs::write(temp.path().join("tokenizer.json"), tokenizer).unwrap();
        fs::write(temp.path().join("model.safetensors"), b"dummy weights").unwrap();

        let validator = ModelValidator::new();
        let result = validator.validate(temp.path());

        assert!(result.is_ok());
    }
}
