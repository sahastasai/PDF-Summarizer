//! Text generation and summarization pipeline
//!
//! Orchestrates the full pipeline from text to summary.

use anyhow::Result;
use burn::prelude::*;
use indicatif::{ProgressBar, ProgressStyle};
use tracing::{debug, info};

use crate::llm::{Llama, LlamaConfig, ModelLoader, SafeTensorData};
use crate::pdf::PdfContent;
use crate::tokenizer::{ChatTemplate, Tokenizer};
/// Generation parameters
#[derive(Debug, Clone)]
pub struct GenerationConfig {
    /// Maximum new tokens to generate
    pub max_new_tokens: usize,
    /// Temperature for sampling
    pub temperature: f32,
    /// Top-p (nucleus) sampling
    pub top_p: f32,
    /// Top-k sampling
    pub top_k: usize,
    /// Target summary length in words
    pub summary_length: usize,
    /// Maximum context length
    pub max_context: usize,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_new_tokens: 512,
            temperature: 0.7,
            top_p: 0.9,
            top_k: 40,
            summary_length: 250,
            max_context: 4096,
        }
    }
}

/// Summary result
#[derive(Debug, Clone)]
pub struct SummaryResult {
    /// Source file path
    pub source: std::path::PathBuf,
    /// Original text length (characters)
    pub original_length: usize,
    /// Generated summary
    pub summary: String,
    /// Summary word count
    pub word_count: usize,
}

/// Summarization pipeline
pub struct SummarizationPipeline<B: Backend> {
    /// The LLaMA model
    model: Llama<B>,
    /// The tokenizer
    tokenizer: Tokenizer,
    /// Generation configuration
    config: GenerationConfig,
    /// Model configuration
    model_config: LlamaConfig,
    /// Device
    device: B::Device,
}

impl<B: Backend> SummarizationPipeline<B> {
    /// Create a new summarization pipeline
    pub fn new(
        model: Llama<B>,
        tokenizer: Tokenizer,
        config: GenerationConfig,
        model_config: LlamaConfig,
        device: B::Device,
    ) -> Self {
        Self {
            model,
            tokenizer,
            config,
            model_config,
            device,
        }
    }

    /// Initialize the pipeline from model path with pretrained weights
    pub fn from_path(
        model_path: &std::path::Path,
        tokenizer_path: Option<&std::path::Path>,
        config: GenerationConfig,
        device: B::Device,
    ) -> Result<Self> {
        info!("Initializing summarization pipeline");

        // Load model configuration
        let loader = ModelLoader::new(model_path.to_path_buf())?;
        let model_config = loader.config().clone();

        // Validate model files exist
        loader.validate()?;

        // Load weights from SafeTensors
        info!("Loading model weights from {:?}...", model_path);
        let weights = SafeTensorData::load(&model_path.to_path_buf())?;

        // Initialize model with pretrained weights
        info!("Building LLaMA model with pretrained weights...");
        let model = Llama::from_pretrained(&weights, &model_config, &device)?;

        // Load tokenizer
        let tokenizer_path = tokenizer_path
            .map(|p| p.to_path_buf())
            .unwrap_or_else(|| loader.tokenizer_path());

        info!("Loading tokenizer from {:?}", tokenizer_path);
        let tokenizer = Tokenizer::from_file(&tokenizer_path)?;

        info!("Pipeline initialized successfully!");
        Ok(Self {
            model,
            tokenizer,
            config,
            model_config,
            device,
        })
    }

    /// Summarize a single piece of text
    pub fn summarize(&self, text: &str) -> Result<String> {
        debug!("Summarizing text of {} characters", text.len());

        // Prepare the prompt
        let prompt = ChatTemplate::summarization_prompt(text, self.config.summary_length);

        // Tokenize
        let input_ids = self.tokenizer.encode_with_bos(&prompt)?;

        // Truncate if necessary
        let actual_max_context = std::cmp::min(
            self.config.max_context,
            self.model_config.max_position_embeddings,
        );
        let max_input = actual_max_context.saturating_sub(self.config.max_new_tokens);
        let input_ids = if input_ids.len() > max_input {
            info!(
                "Truncating input from {} to {} tokens",
                input_ids.len(),
                max_input
            );
            self.tokenizer.truncate(input_ids, max_input)
        } else {
            input_ids
        };

        debug!("Input tokens: {}", input_ids.len());

        // Convert to tensor
        let input_tensor = self.ids_to_tensor(&input_ids);

        // Generate
        let output_ids = self.model.generate(
            input_tensor,
            self.config.max_new_tokens,
            self.config.temperature,
            self.config.top_p,
            self.config.top_k,
            &self.model_config,
        );

        // Decode output
        let output_data = output_ids.to_data();
        let output_vec: Vec<u32> = output_data.iter::<i64>().map(|v| v as u32).collect();

        // Extract only the generated tokens (after the input)
        let generated = &output_vec[input_ids.len()..];
        let summary = self.tokenizer.decode(generated, true)?;

        Ok(summary.trim().to_string())
    }

    /// Summarize PDF content
    pub fn summarize_pdf(&self, pdf: &PdfContent) -> Result<SummaryResult> {
        info!("Summarizing PDF: {:?}", pdf.source_path);

        let summary = self.summarize(&pdf.text)?;
        let word_count = summary.split_whitespace().count();

        Ok(SummaryResult {
            source: pdf.source_path.clone(),
            original_length: pdf.text.len(),
            summary,
            word_count,
        })
    }

    /// Summarize multiple PDFs with progress
    pub fn summarize_batch(&self, pdfs: &[PdfContent]) -> Result<Vec<SummaryResult>> {
        let pb = ProgressBar::new(pdfs.len() as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} Summarizing...")
                .unwrap()
                .progress_chars("#>-"),
        );

        let mut results = Vec::with_capacity(pdfs.len());

        for pdf in pdfs {
            pb.set_message(format!(
                "{}",
                pdf.source_path
                    .file_name()
                    .unwrap_or_default()
                    .to_string_lossy()
            ));

            let result = self.summarize_pdf(pdf)?;
            results.push(result);
            pb.inc(1);
        }

        pb.finish_with_message("Complete");
        Ok(results)
    }

    /// Convert token IDs to a tensor
    fn ids_to_tensor(&self, ids: &[u32]) -> Tensor<B, 2, Int> {
        let ids_i64: Vec<i64> = ids.iter().map(|&id| id as i64).collect();
        let len = ids.len();
        let tensor_data = burn::tensor::TensorData::new(ids_i64, vec![1, len]);
        Tensor::from_data(tensor_data, &self.device)
    }

    /// Get the tokenizer
    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }

    /// Get the model
    pub fn model(&self) -> &Llama<B> {
        &self.model
    }
}

/// Chunk text for processing long documents
pub fn chunk_text(text: &str, max_chunk_size: usize, overlap: usize) -> Vec<String> {
    if text.len() <= max_chunk_size {
        return vec![text.to_string()];
    }

    let mut chunks = Vec::new();
    let mut start = 0;

    while start < text.len() {
        let end = (start + max_chunk_size).min(text.len());

        // Try to find a sentence boundary
        let chunk_end = if end < text.len() {
            text[start..end]
                .rfind(|c| c == '.' || c == '!' || c == '?')
                .map(|pos| start + pos + 1)
                .unwrap_or(end)
        } else {
            end
        };

        chunks.push(text[start..chunk_end].to_string());

        // Move start position with overlap
        start = if chunk_end > overlap {
            chunk_end - overlap
        } else {
            chunk_end
        };

        // Prevent infinite loop
        if start >= text.len() || chunk_end == text.len() {
            break;
        }
    }

    chunks
}

/// Combine multiple summaries into one
pub fn combine_summaries(summaries: &[String], target_length: usize) -> String {
    if summaries.len() == 1 {
        return summaries[0].clone();
    }

    // Simple concatenation with section markers
    let combined: String = summaries
        .iter()
        .enumerate()
        .map(|(i, s)| format!("[Section {}] {}", i + 1, s))
        .collect::<Vec<_>>()
        .join("\n\n");

    // If combined is short enough, return it
    let word_count: usize = combined.split_whitespace().count();
    if word_count <= target_length * 2 {
        return combined;
    }

    // Otherwise, truncate to approximate target
    let words: Vec<&str> = combined.split_whitespace().collect();
    words[..target_length.min(words.len())].join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chunk_text() {
        let text = "First sentence. Second sentence. Third sentence. Fourth sentence.";
        let chunks = chunk_text(text, 30, 5);
        assert!(!chunks.is_empty());
        assert!(chunks[0].len() <= 30 + 20); // Some flexibility for sentence boundaries
    }

    #[test]
    fn test_combine_summaries() {
        let summaries = vec!["Summary one.".to_string(), "Summary two.".to_string()];
        let combined = combine_summaries(&summaries, 100);
        assert!(combined.contains("[Section 1]"));
        assert!(combined.contains("[Section 2]"));
    }
}
