//! Tokenizer module for LLaMA 3
//!
//! Handles text tokenization using the HuggingFace tokenizers library.

use anyhow::Result;
use std::path::Path;
use tokenizers::Tokenizer as HfTokenizer;
use tracing::{debug, info};

/// LLaMA 3 Tokenizer wrapper
pub struct Tokenizer {
    /// Underlying HuggingFace tokenizer
    inner: HfTokenizer,
    /// Beginning of sequence token ID
    bos_token_id: u32,
    /// End of sequence token ID
    eos_token_id: u32,
    /// Padding token ID
    pad_token_id: u32,
}

impl Tokenizer {
    /// Load tokenizer from a file
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();
        info!("Loading tokenizer from {:?}", path);

        let inner = HfTokenizer::from_file(path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;

        // LLaMA 3 special tokens
        let bos_token_id = 128000;
        let eos_token_id = 128001;
        let pad_token_id = 128002;

        debug!(
            "Tokenizer loaded with vocab size: {}",
            inner.get_vocab_size(true)
        );

        Ok(Self {
            inner,
            bos_token_id,
            eos_token_id,
            pad_token_id,
        })
    }

    /// Encode text to token IDs
    pub fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>> {
        let encoding = self
            .inner
            .encode(text, add_special_tokens)
            .map_err(|e| anyhow::anyhow!("Failed to encode text: {}", e))?;

        Ok(encoding.get_ids().to_vec())
    }

    /// Encode text with BOS token prepended
    pub fn encode_with_bos(&self, text: &str) -> Result<Vec<u32>> {
        let mut ids = vec![self.bos_token_id];
        let encoded = self.encode(text, false)?;
        ids.extend(encoded);
        Ok(ids)
    }

    /// Decode token IDs to text
    pub fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String> {
        self.inner
            .decode(ids, skip_special_tokens)
            .map_err(|e| anyhow::anyhow!("Failed to decode tokens: {}", e))
    }

    /// Get BOS token ID
    pub fn bos_token_id(&self) -> u32 {
        self.bos_token_id
    }

    /// Get EOS token ID
    pub fn eos_token_id(&self) -> u32 {
        self.eos_token_id
    }

    /// Get padding token ID
    pub fn pad_token_id(&self) -> u32 {
        self.pad_token_id
    }

    /// Get vocabulary size
    pub fn vocab_size(&self) -> usize {
        self.inner.get_vocab_size(true)
    }

    /// Encode a batch of texts
    pub fn encode_batch(&self, texts: &[&str], add_special_tokens: bool) -> Result<Vec<Vec<u32>>> {
        let encodings = self
            .inner
            .encode_batch(texts.to_vec(), add_special_tokens)
            .map_err(|e| anyhow::anyhow!("Failed to encode batch: {}", e))?;

        Ok(encodings
            .into_iter()
            .map(|e| e.get_ids().to_vec())
            .collect())
    }

    /// Decode a batch of token sequences
    pub fn decode_batch(
        &self,
        sequences: &[Vec<u32>],
        skip_special_tokens: bool,
    ) -> Result<Vec<String>> {
        sequences
            .iter()
            .map(|ids| self.decode(ids, skip_special_tokens))
            .collect()
    }

    /// Truncate tokens to a maximum length
    pub fn truncate(&self, tokens: Vec<u32>, max_length: usize) -> Vec<u32> {
        if tokens.len() <= max_length {
            tokens
        } else {
            tokens[..max_length].to_vec()
        }
    }

    /// Pad tokens to a specific length
    pub fn pad(&self, tokens: Vec<u32>, target_length: usize, pad_left: bool) -> Vec<u32> {
        if tokens.len() >= target_length {
            return tokens;
        }

        let padding_count = target_length - tokens.len();
        let padding = vec![self.pad_token_id; padding_count];

        if pad_left {
            [padding, tokens].concat()
        } else {
            [tokens, padding].concat()
        }
    }

    /// Create an attention mask for a sequence
    pub fn create_attention_mask(&self, tokens: &[u32]) -> Vec<u32> {
        tokens
            .iter()
            .map(|&id| if id == self.pad_token_id { 0 } else { 1 })
            .collect()
    }
}

/// Chat template for LLaMA 3 Instruct
pub struct ChatTemplate;

impl ChatTemplate {
    /// Format a system message
    pub fn system_message(content: &str) -> String {
        format!(
            "<|start_header_id|>system<|end_header_id|>\n\n{}<|eot_id|>",
            content
        )
    }

    /// Format a user message
    pub fn user_message(content: &str) -> String {
        format!(
            "<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|>",
            content
        )
    }

    /// Format an assistant message
    pub fn assistant_message(content: &str) -> String {
        format!(
            "<|start_header_id|>assistant<|end_header_id|>\n\n{}<|eot_id|>",
            content
        )
    }

    /// Start assistant response (for generation)
    pub fn assistant_start() -> String {
        "<|start_header_id|>assistant<|end_header_id|>\n\n".to_string()
    }

    /// Format a complete prompt for summarization
    pub fn summarization_prompt(text: &str, word_count: usize) -> String {
        let system = Self::system_message(
            "You are a helpful assistant that summarizes text. \
             Provide concise, accurate summaries that capture the key points.",
        );

        let user = Self::user_message(&format!(
            "Please provide a summary of the following text in approximately {} words. \
             Focus on the main ideas and key information.\n\n\
             Text to summarize:\n\n{}",
            word_count, text
        ));

        let assistant_start = Self::assistant_start();

        format!("<|begin_of_text|>{}{}{}", system, user, assistant_start)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chat_template() {
        let prompt = ChatTemplate::summarization_prompt("Hello world", 100);
        assert!(prompt.contains("<|begin_of_text|>"));
        assert!(prompt.contains("system"));
        assert!(prompt.contains("user"));
        assert!(prompt.contains("Hello world"));
    }

    #[test]
    fn test_truncate() {
        // Create a mock tokenizer test
        let tokens = vec![1, 2, 3, 4, 5];
        let truncated: Vec<u32> = if tokens.len() > 3 {
            tokens[..3].to_vec()
        } else {
            tokens
        };
        assert_eq!(truncated.len(), 3);
    }
}
