//! CLI argument parsing module
//!
//! Handles command-line arguments for the PDF summarizer tool.

use clap::Parser;
use std::path::PathBuf;

/// PDF Summarizer - Extract and summarize PDF documents using LLaMA 3
#[derive(Parser, Debug, Clone)]
#[command(name = "pdf-summarizer")]
#[command(author = "PDF Summarizer Team")]
#[command(version = "0.1.0")]
#[command(about = "Process PDFs and generate summaries using LLaMA 3", long_about = None)]
pub struct Args {
    /// Length of the summary to generate (in words)
    #[arg(short = 's', long = "summary-length", default_value = "250")]
    pub summary_length: usize,

    /// Paths to individual PDF files to process
    #[arg(short = 'f', long = "files", value_delimiter = ',', num_args = 1..)]
    pub files: Option<Vec<PathBuf>>,

    /// Path to a folder containing PDF files to process
    #[arg(short = 'F', long = "folder")]
    pub folder: Option<PathBuf>,

    /// Path to the output file (must be .txt, otherwise outputs to stdout)
    #[arg(short = 'o', long = "output")]
    pub output: Option<PathBuf>,

    /// Path to the LLaMA 3 model directory
    #[arg(short = 'm', long = "model", env = "LLAMA_MODEL_PATH")]
    pub model_path: Option<PathBuf>,

    /// Path to the tokenizer file (tokenizer.json)
    #[arg(short = 't', long = "tokenizer", env = "LLAMA_TOKENIZER_PATH")]
    pub tokenizer_path: Option<PathBuf>,

    /// HuggingFace API token for accessing gated models (like LLaMA 3)
    #[arg(long = "hf-token", env = "HF_TOKEN")]
    pub hf_token: Option<String>,

    /// Use CPU instead of GPU
    #[arg(long = "cpu", default_value = "false")]
    pub use_cpu: bool,

    /// GPU device index to use
    #[arg(long = "gpu-device", default_value = "0")]
    pub gpu_device: usize,

    /// Maximum context length for the model
    #[arg(long = "max-context", default_value = "4096")]
    pub max_context: usize,

    /// Temperature for text generation (0.0 - 2.0)
    #[arg(long = "temperature", default_value = "0.7")]
    pub temperature: f32,

    /// Top-p (nucleus) sampling parameter
    #[arg(long = "top-p", default_value = "0.9")]
    pub top_p: f32,

    /// Top-k sampling parameter
    #[arg(long = "top-k", default_value = "40")]
    pub top_k: usize,

    /// Verbose output
    #[arg(short = 'v', long = "verbose", action = clap::ArgAction::Count)]
    pub verbose: u8,

    /// Batch size for processing multiple PDFs
    #[arg(long = "batch-size", default_value = "1")]
    pub batch_size: usize,

    /// Skip PDFs that fail to parse
    #[arg(long = "skip-errors", default_value = "false")]
    pub skip_errors: bool,
}

impl Args {
    /// Validate the arguments
    pub fn validate(&self) -> Result<(), String> {
        // Check that at least one input source is provided
        if self.files.is_none() && self.folder.is_none() {
            return Err(
                "At least one of --files (-f) or --folder (-F) must be provided".to_string(),
            );
        }

        // Validate summary length
        if self.summary_length == 0 {
            return Err("Summary length must be greater than 0".to_string());
        }

        // Validate temperature
        if self.temperature < 0.0 || self.temperature > 2.0 {
            return Err("Temperature must be between 0.0 and 2.0".to_string());
        }

        // Validate top_p
        if self.top_p < 0.0 || self.top_p > 1.0 {
            return Err("Top-p must be between 0.0 and 1.0".to_string());
        }

        // Validate folder exists if provided
        if let Some(ref folder) = self.folder {
            if !folder.exists() {
                return Err(format!("Folder does not exist: {:?}", folder));
            }
            if !folder.is_dir() {
                return Err(format!("Path is not a directory: {:?}", folder));
            }
        }

        // Validate individual files exist
        if let Some(ref files) = self.files {
            for file in files {
                if !file.exists() {
                    return Err(format!("File does not exist: {:?}", file));
                }
                if !file.is_file() {
                    return Err(format!("Path is not a file: {:?}", file));
                }
            }
        }

        // Validate output file extension if provided
        if let Some(ref output) = self.output {
            if let Some(ext) = output.extension() {
                if ext != "txt" {
                    tracing::warn!(
                        "Output file extension is not .txt ({:?}), will output to stdout instead",
                        ext
                    );
                }
            }
        }

        Ok(())
    }

    /// Get all PDF files to process
    pub fn get_pdf_files(&self) -> Result<Vec<PathBuf>, String> {
        let mut pdf_files = Vec::new();

        // Add individual files
        if let Some(ref files) = self.files {
            for file in files {
                if file.extension().map(|e| e == "pdf").unwrap_or(false) {
                    pdf_files.push(file.clone());
                } else {
                    tracing::warn!("Skipping non-PDF file: {:?}", file);
                }
            }
        }

        // Add files from folder
        if let Some(ref folder) = self.folder {
            for entry in walkdir::WalkDir::new(folder)
                .follow_links(true)
                .into_iter()
                .filter_map(|e| e.ok())
            {
                let path = entry.path();
                if path.is_file() && path.extension().map(|e| e == "pdf").unwrap_or(false) {
                    pdf_files.push(path.to_path_buf());
                }
            }
        }

        if pdf_files.is_empty() {
            return Err("No PDF files found to process".to_string());
        }

        Ok(pdf_files)
    }

    /// Check if output should go to file
    pub fn should_output_to_file(&self) -> bool {
        if let Some(ref output) = self.output {
            output.extension().map(|e| e == "txt").unwrap_or(false)
        } else {
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_values() {
        let args = Args::parse_from(["pdf-summarizer", "-f", "test.pdf"]);
        assert_eq!(args.summary_length, 250);
        assert_eq!(args.temperature, 0.7);
        assert_eq!(args.top_p, 0.9);
    }
}
