//! Output handling module
//!
//! Handles writing results to stdout or files.

use anyhow::Result;
use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::Path;

use crate::pipeline::SummaryResult;

/// Output format
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputFormat {
    /// Plain text
    Plain,
    /// JSON format
    Json,
    /// Markdown format
    Markdown,
}

/// Output writer abstraction
pub struct OutputWriter {
    writer: Box<dyn Write>,
    format: OutputFormat,
    is_file: bool,
}

impl OutputWriter {
    /// Create a new stdout writer
    pub fn stdout(format: OutputFormat) -> Self {
        Self {
            writer: Box::new(io::stdout()),
            format,
            is_file: false,
        }
    }

    /// Create a new file writer
    pub fn file<P: AsRef<Path>>(path: P, format: OutputFormat) -> Result<Self> {
        let file = File::create(path)?;
        let writer = BufWriter::new(file);
        Ok(Self {
            writer: Box::new(writer),
            format,
            is_file: true,
        })
    }

    /// Create writer based on output path
    pub fn from_path(path: Option<&Path>) -> Result<Self> {
        match path {
            Some(p) => {
                // Check if it's a .txt file
                let ext = p.extension().and_then(|e| e.to_str()).unwrap_or("");
                if ext != "txt" {
                    tracing::warn!("Output file is not .txt, writing to stdout instead");
                    Ok(Self::stdout(OutputFormat::Plain))
                } else {
                    Self::file(p, OutputFormat::Plain)
                }
            }
            None => Ok(Self::stdout(OutputFormat::Plain)),
        }
    }

    /// Write a single summary result
    pub fn write_result(&mut self, result: &SummaryResult) -> Result<()> {
        match self.format {
            OutputFormat::Plain => self.write_plain(result),
            OutputFormat::Json => self.write_json(result),
            OutputFormat::Markdown => self.write_markdown(result),
        }
    }

    /// Write multiple results
    pub fn write_results(&mut self, results: &[SummaryResult]) -> Result<()> {
        match self.format {
            OutputFormat::Plain => {
                for (i, result) in results.iter().enumerate() {
                    if i > 0 {
                        writeln!(self.writer, "\n{}", "=".repeat(80))?;
                    }
                    self.write_plain(result)?;
                }
            }
            OutputFormat::Json => {
                self.write_json_array(results)?;
            }
            OutputFormat::Markdown => {
                for (i, result) in results.iter().enumerate() {
                    if i > 0 {
                        writeln!(self.writer, "\n---\n")?;
                    }
                    self.write_markdown(result)?;
                }
            }
        }
        Ok(())
    }

    /// Write in plain text format
    fn write_plain(&mut self, result: &SummaryResult) -> Result<()> {
        writeln!(self.writer, "Source: {}", result.source.display())?;
        writeln!(
            self.writer,
            "Original Length: {} characters",
            result.original_length
        )?;
        writeln!(
            self.writer,
            "Summary Word Count: {} words",
            result.word_count
        )?;
        writeln!(self.writer)?;
        writeln!(self.writer, "Summary:")?;
        writeln!(self.writer, "{}", result.summary)?;
        Ok(())
    }

    /// Write in JSON format
    fn write_json(&mut self, result: &SummaryResult) -> Result<()> {
        let json = serde_json::json!({
            "source": result.source.display().to_string(),
            "original_length": result.original_length,
            "word_count": result.word_count,
            "summary": result.summary
        });
        writeln!(self.writer, "{}", serde_json::to_string_pretty(&json)?)?;
        Ok(())
    }

    /// Write JSON array
    fn write_json_array(&mut self, results: &[SummaryResult]) -> Result<()> {
        let json_array: Vec<serde_json::Value> = results
            .iter()
            .map(|result| {
                serde_json::json!({
                    "source": result.source.display().to_string(),
                    "original_length": result.original_length,
                    "word_count": result.word_count,
                    "summary": result.summary
                })
            })
            .collect();
        writeln!(
            self.writer,
            "{}",
            serde_json::to_string_pretty(&json_array)?
        )?;
        Ok(())
    }

    /// Write in Markdown format
    fn write_markdown(&mut self, result: &SummaryResult) -> Result<()> {
        writeln!(
            self.writer,
            "## Summary: {}",
            result
                .source
                .file_name()
                .unwrap_or_default()
                .to_string_lossy()
        )?;
        writeln!(self.writer)?;
        writeln!(self.writer, "**Source:** `{}`", result.source.display())?;
        writeln!(
            self.writer,
            "**Original Length:** {} characters",
            result.original_length
        )?;
        writeln!(
            self.writer,
            "**Summary Word Count:** {} words",
            result.word_count
        )?;
        writeln!(self.writer)?;
        writeln!(self.writer, "### Content")?;
        writeln!(self.writer)?;
        writeln!(self.writer, "{}", result.summary)?;
        Ok(())
    }

    /// Flush the writer
    pub fn flush(&mut self) -> Result<()> {
        self.writer.flush()?;
        Ok(())
    }

    /// Check if writing to a file
    pub fn is_file(&self) -> bool {
        self.is_file
    }
}

/// Format bytes as human-readable string
pub fn format_bytes(bytes: usize) -> String {
    const KB: usize = 1024;
    const MB: usize = KB * 1024;
    const GB: usize = MB * 1024;

    if bytes >= GB {
        format!("{:.2} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.2} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.2} KB", bytes as f64 / KB as f64)
    } else {
        format!("{} bytes", bytes)
    }
}

/// Format duration as human-readable string
pub fn format_duration(seconds: f64) -> String {
    if seconds >= 3600.0 {
        let hours = (seconds / 3600.0).floor();
        let mins = ((seconds % 3600.0) / 60.0).floor();
        format!("{:.0}h {:.0}m", hours, mins)
    } else if seconds >= 60.0 {
        let mins = (seconds / 60.0).floor();
        let secs = seconds % 60.0;
        format!("{:.0}m {:.1}s", mins, secs)
    } else {
        format!("{:.2}s", seconds)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_format_bytes() {
        assert_eq!(format_bytes(500), "500 bytes");
        assert_eq!(format_bytes(2048), "2.00 KB");
        assert_eq!(format_bytes(1_500_000), "1.43 MB");
    }

    #[test]
    fn test_format_duration() {
        assert_eq!(format_duration(30.5), "30.50s");
        assert_eq!(format_duration(90.0), "1m 30.0s");
        assert_eq!(format_duration(3700.0), "1h 1m");
    }
}
