//! PDF Summarizer - A CLI tool for PDF text extraction and summarization using LLaMA 3
//!
//! This application processes PDF files, extracts their text content, and generates
//! concise summaries using a LLaMA 3 model running on GPU via the Burn framework.

pub mod cli;
pub mod error;
pub mod llm;
pub mod model_manager;
pub mod output;
pub mod pdf;
pub mod pipeline;
pub mod tokenizer;

use anyhow::{Context, Result};
use burn::backend::wgpu::{Wgpu, WgpuDevice};
use clap::Parser;
use std::time::Instant;
use tracing::{info, warn, Level};
use tracing_subscriber::FmtSubscriber;

use cli::Args;
use model_manager::ensure_model_available_sync;
use output::OutputWriter;
use pdf::process_pdfs_with_progress;
use pipeline::{GenerationConfig, SummarizationPipeline};

/// Type alias for our backend (WebGPU for cross-platform GPU support)
type Backend = Wgpu;

/// Initialize logging based on verbosity level
fn init_logging(verbosity: u8) {
    let level = match verbosity {
        0 => Level::WARN,
        1 => Level::INFO,
        2 => Level::DEBUG,
        _ => Level::TRACE,
    };

    let subscriber = FmtSubscriber::builder()
        .with_max_level(level)
        .with_target(false)
        .with_thread_ids(false)
        .with_file(false)
        .with_line_number(false)
        .compact()
        .finish();

    tracing::subscriber::set_global_default(subscriber).expect("Failed to set tracing subscriber");
}

/// Select the best available WGPU device
fn select_device(use_cpu: bool, _gpu_device: usize) -> WgpuDevice {
    if use_cpu {
        info!("Using CPU backend (as requested)");
        return WgpuDevice::Cpu;
    }

    // Use DefaultDevice which automatically selects the best available device
    // based on the CUBECL_WGPU_DEFAULT_DEVICE env var or auto-detection.
    // This handles cases where no discrete GPU is available.
    info!("Selecting default compute device (auto-detection)...");
    
    let device = WgpuDevice::DefaultDevice;
    
    info!("Using device: {:?}", device);
    device
}

/// Main application entry point
fn main() -> Result<()> {
    // Parse command line arguments
    let args = Args::parse();

    // Initialize logging
    init_logging(args.verbose);

    // Validate arguments
    args.validate().map_err(|e| anyhow::anyhow!("{}", e))?;

    // Run the application
    run(args)
}

/// Run the main application logic
fn run(args: Args) -> Result<()> {
    let start_time = Instant::now();

    info!("PDF Summarizer v{}", env!("CARGO_PKG_VERSION"));
    info!("Summary length: {} words", args.summary_length);

    // Get PDF files to process
    let pdf_files = args.get_pdf_files().map_err(|e| anyhow::anyhow!("{}", e))?;

    info!("Found {} PDF files to process", pdf_files.len());

    // Extract text from PDFs
    info!("Extracting text from PDFs...");
    let pdf_contents = process_pdfs_with_progress(&pdf_files, args.skip_errors)?;

    if pdf_contents.is_empty() {
        warn!("No PDF content could be extracted");
        return Ok(());
    }

    info!(
        "Successfully extracted text from {} PDFs",
        pdf_contents.len()
    );

    // Initialize device with fallback logic
    let device = select_device(args.use_cpu, args.gpu_device);

    // Get model path (download if necessary)
    info!("Checking for model...");
    let model_path = ensure_model_available_sync(
        args.model_path.clone(),
        args.hf_token.clone(),
    ).context("Failed to ensure model is available")?;

    // Get tokenizer path (use the one from model directory if not specified)
    let tokenizer_path = args.tokenizer_path.clone()
        .or_else(|| Some(model_path.join("tokenizer.json")));
    let tokenizer_path = tokenizer_path.as_deref();

    // Create generation config
    let gen_config = GenerationConfig {
        max_new_tokens: estimate_tokens_for_words(args.summary_length),
        temperature: args.temperature,
        top_p: args.top_p,
        top_k: args.top_k,
        summary_length: args.summary_length,
        max_context: args.max_context,
    };

    // Initialize the pipeline
    info!("Initializing LLaMA model...");
    let pipeline: SummarizationPipeline<Backend> =
        SummarizationPipeline::from_path(&model_path, tokenizer_path, gen_config, device)?;

    // Process PDFs and generate summaries
    info!("Generating summaries...");
    let results = pipeline.summarize_batch(&pdf_contents)?;

    // Write output
    let output_path = args.output.as_deref();
    let mut writer = OutputWriter::from_path(output_path)?;

    if writer.is_file() {
        info!("Writing results to {:?}", args.output.as_ref().unwrap());
    }

    writer.write_results(&results)?;
    writer.flush()?;

    // Print summary statistics
    let elapsed = start_time.elapsed();
    let total_chars: usize = pdf_contents.iter().map(|p| p.text.len()).sum();
    let total_words: usize = results.iter().map(|r| r.word_count).sum();

    info!("Processing complete!");
    info!("  Files processed: {}", pdf_contents.len());
    info!("  Total input: {} characters", total_chars);
    info!("  Total output: {} words", total_words);
    info!("  Time elapsed: {:.2}s", elapsed.as_secs_f64());

    Ok(())
}

/// Estimate the number of tokens needed for a given word count
fn estimate_tokens_for_words(words: usize) -> usize {
    // Rough estimate: ~1.3 tokens per word for English text
    (words as f64 * 1.5).ceil() as usize
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_estimate_tokens() {
        assert!(estimate_tokens_for_words(100) >= 100);
        assert!(estimate_tokens_for_words(100) <= 200);
    }
}
