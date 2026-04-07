//! PDF text extraction module
//!
//! Handles extraction of text content from PDF files.

use anyhow::{Context, Result};
use std::path::Path;
use thiserror::Error;
use tracing::{debug, info, warn};

/// Errors that can occur during PDF processing
#[derive(Error, Debug)]
pub enum PdfError {
    #[error("Failed to open PDF file: {0}")]
    OpenError(String),

    #[error("Failed to extract text from PDF: {0}")]
    ExtractionError(String),

    #[error("PDF file is empty or contains no extractable text")]
    EmptyPdf,

    #[error("PDF file is password protected")]
    PasswordProtected,

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
}

/// Represents extracted content from a PDF
#[derive(Debug, Clone)]
pub struct PdfContent {
    /// The source file path
    pub source_path: std::path::PathBuf,

    /// Extracted text content
    pub text: String,

    /// Number of pages in the PDF
    pub page_count: usize,

    /// Metadata if available
    pub metadata: Option<PdfMetadata>,
}

/// PDF metadata
#[derive(Debug, Clone, Default)]
pub struct PdfMetadata {
    pub title: Option<String>,
    pub author: Option<String>,
    pub subject: Option<String>,
    pub creator: Option<String>,
}

/// PDF text extractor
pub struct PdfExtractor {
    /// Whether to preserve formatting
    preserve_formatting: bool,

    /// Maximum pages to extract (0 = all)
    max_pages: usize,
}

impl Default for PdfExtractor {
    fn default() -> Self {
        Self {
            preserve_formatting: false,
            max_pages: 0,
        }
    }
}

impl PdfExtractor {
    /// Create a new PDF extractor
    pub fn new() -> Self {
        Self::default()
    }

    /// Set whether to preserve formatting
    pub fn with_formatting(mut self, preserve: bool) -> Self {
        self.preserve_formatting = preserve;
        self
    }

    /// Set maximum pages to extract
    pub fn with_max_pages(mut self, max: usize) -> Self {
        self.max_pages = max;
        self
    }

    /// Extract text from a PDF file
    pub fn extract<P: AsRef<Path>>(&self, path: P) -> Result<PdfContent> {
        let path = path.as_ref();
        info!("Extracting text from PDF: {:?}", path);

        // Read the PDF file
        let bytes =
            std::fs::read(path).with_context(|| format!("Failed to read PDF file: {:?}", path))?;

        // Try to load the PDF document using lopdf
        let doc =
            lopdf::Document::load_mem(&bytes).map_err(|e| PdfError::OpenError(e.to_string()))?;

        let page_count = doc.get_pages().len();
        debug!("PDF has {} pages", page_count);

        // Extract metadata
        let metadata = self.extract_metadata(&doc);

        // Extract text using pdf-extract
        let text = pdf_extract::extract_text_from_mem(&bytes)
            .map_err(|e| PdfError::ExtractionError(e.to_string()))?;

        // Clean up the extracted text
        let text = self.clean_text(&text);

        if text.trim().is_empty() {
            warn!("Extracted text is empty for: {:?}", path);
            return Err(PdfError::EmptyPdf.into());
        }

        info!("Extracted {} characters from {:?}", text.len(), path);

        Ok(PdfContent {
            source_path: path.to_path_buf(),
            text,
            page_count,
            metadata,
        })
    }

    /// Extract metadata from the PDF
    fn extract_metadata(&self, doc: &lopdf::Document) -> Option<PdfMetadata> {
        // Try to get Info dictionary from trailer
        let info_ref = doc.trailer.get(b"Info").ok()?;
        let info_id = info_ref.as_reference().ok()?;
        let info_dict = doc.get_dictionary(info_id).ok()?;

        let get_string = |key: &[u8]| -> Option<String> {
            info_dict
                .get(key)
                .ok()
                .and_then(|v| v.as_string().ok())
                .map(|s| s.to_string())
        };

        Some(PdfMetadata {
            title: get_string(b"Title"),
            author: get_string(b"Author"),
            subject: get_string(b"Subject"),
            creator: get_string(b"Creator"),
        })
    }

    /// Clean up extracted text
    fn clean_text(&self, text: &str) -> String {
        let mut result = String::with_capacity(text.len());
        let mut prev_char_was_whitespace = false;
        let mut prev_char_was_newline = false;

        for ch in text.chars() {
            match ch {
                // Handle newlines
                '\n' | '\r' => {
                    if !prev_char_was_newline {
                        result.push('\n');
                        prev_char_was_newline = true;
                        prev_char_was_whitespace = true;
                    }
                }
                // Handle other whitespace
                ' ' | '\t' => {
                    if !prev_char_was_whitespace {
                        result.push(' ');
                        prev_char_was_whitespace = true;
                    }
                    prev_char_was_newline = false;
                }
                // Handle form feed and other control characters
                '\x0C' | '\x0B' => {
                    if !prev_char_was_newline {
                        result.push('\n');
                        prev_char_was_newline = true;
                        prev_char_was_whitespace = true;
                    }
                }
                // Regular characters
                _ => {
                    // Skip non-printable characters except common ones
                    if ch.is_control() && ch != '\t' {
                        continue;
                    }
                    result.push(ch);
                    prev_char_was_whitespace = false;
                    prev_char_was_newline = false;
                }
            }
        }

        result.trim().to_string()
    }

    /// Extract text from multiple PDFs
    pub fn extract_multiple<P: AsRef<Path>>(
        &self,
        paths: &[P],
        skip_errors: bool,
    ) -> Result<Vec<PdfContent>> {
        let mut results = Vec::with_capacity(paths.len());

        for path in paths {
            match self.extract(path) {
                Ok(content) => results.push(content),
                Err(e) => {
                    if skip_errors {
                        warn!("Failed to extract PDF {:?}: {}", path.as_ref(), e);
                    } else {
                        return Err(e);
                    }
                }
            }
        }

        Ok(results)
    }
}

/// Batch process PDFs with progress reporting
pub fn process_pdfs_with_progress<P: AsRef<Path>>(
    paths: &[P],
    skip_errors: bool,
) -> Result<Vec<PdfContent>> {
    use indicatif::{ProgressBar, ProgressStyle};

    let pb = ProgressBar::new(paths.len() as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template(
                "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})",
            )
            .unwrap()
            .progress_chars("#>-"),
    );

    let extractor = PdfExtractor::new();
    let mut results = Vec::with_capacity(paths.len());

    for path in paths {
        pb.set_message(format!(
            "Processing: {:?}",
            path.as_ref().file_name().unwrap_or_default()
        ));

        match extractor.extract(path) {
            Ok(content) => results.push(content),
            Err(e) => {
                if skip_errors {
                    warn!("Failed to extract PDF {:?}: {}", path.as_ref(), e);
                } else {
                    pb.finish_with_message("Failed");
                    return Err(e);
                }
            }
        }

        pb.inc(1);
    }

    pb.finish_with_message("Complete");
    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clean_text() {
        let extractor = PdfExtractor::new();

        let text = "Hello    World\n\n\nTest";
        let cleaned = extractor.clean_text(text);
        assert_eq!(cleaned, "Hello World\nTest");
    }

    #[test]
    fn test_extractor_builder() {
        let extractor = PdfExtractor::new().with_formatting(true).with_max_pages(10);

        assert!(extractor.preserve_formatting);
        assert_eq!(extractor.max_pages, 10);
    }
}
