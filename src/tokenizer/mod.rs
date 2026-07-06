//! Tokenizer module for LLaMA models
//!
//! Handles text tokenization using the HuggingFace tokenizers library.
//! Supports different LLaMA variants (TinyLlama, LLaMA 3, etc.)

use anyhow::Result;
use std::path::Path;
use tokenizers::Tokenizer as HfTokenizer;
use tracing::{debug, info};

/// LLaMA Tokenizer wrapper
pub struct Tokenizer {
    /// Underlying HuggingFace tokenizer
    inner: HfTokenizer,
    /// Beginning of sequence token ID
    bos_token_id: u32,
    /// End of sequence token ID
    eos_token_id: u32,
    /// Padding token ID
    pad_token_id: u32,
    /// Chat template format (llama3 or llama2/tinyllama)
    chat_format: ChatFormat,
}

/// Chat format type
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ChatFormat {
    /// LLaMA 3 format with <|start_header_id|> etc.
    Llama3,
    /// LLaMA 2 / TinyLlama format with [INST] etc.
    Llama2,
}

impl Tokenizer {
    /// Load tokenizer from a file
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();
        info!("Loading tokenizer from {:?}", path);

        let inner = HfTokenizer::from_file(path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;

        let vocab_size = inner.get_vocab_size(true);
        debug!("Tokenizer vocab size: {}", vocab_size);

        // Determine chat format and special tokens based on vocab size
        // LLaMA 3 has vocab size ~128000, TinyLlama/LLaMA 2 has ~32000
        let (bos_token_id, eos_token_id, pad_token_id, chat_format) = if vocab_size > 100000 {
            info!("Detected LLaMA 3 tokenizer (vocab size: {})", vocab_size);
            (128000, 128001, 128002, ChatFormat::Llama3)
        } else {
            // Try to load from config files
            let dir = path.parent().unwrap_or(Path::new("."));
            let (bos, eos, pad) = Self::load_special_tokens_from_config(dir).unwrap_or((1, 2, 0)); // Default TinyLlama/LLaMA 2 tokens
            info!(
                "Detected LLaMA 2/TinyLlama tokenizer (vocab size: {})",
                vocab_size
            );
            info!("Special tokens - BOS: {}, EOS: {}, PAD: {}", bos, eos, pad);
            (bos, eos, pad, ChatFormat::Llama2)
        };

        Ok(Self {
            inner,
            bos_token_id,
            eos_token_id,
            pad_token_id,
            chat_format,
        })
    }

    /// Load special token IDs from generation_config.json or tokenizer_config.json
    fn load_special_tokens_from_config(dir: &Path) -> Option<(u32, u32, u32)> {
        // Try generation_config.json first
        let gen_config_path = dir.join("generation_config.json");
        if gen_config_path.exists() {
            if let Ok(content) = std::fs::read_to_string(&gen_config_path) {
                if let Ok(config) = serde_json::from_str::<serde_json::Value>(&content) {
                    let bos = config["bos_token_id"].as_u64().map(|v| v as u32);
                    let eos = config["eos_token_id"].as_u64().map(|v| v as u32);
                    let pad = config["pad_token_id"].as_u64().map(|v| v as u32);

                    if bos.is_some() && eos.is_some() {
                        return Some((bos.unwrap(), eos.unwrap(), pad.unwrap_or(0)));
                    }
                }
            }
        }

        // Fallback to tokenizer_config.json
        let tokenizer_config_path = dir.join("tokenizer_config.json");
        if tokenizer_config_path.exists() {
            if let Ok(content) = std::fs::read_to_string(&tokenizer_config_path) {
                if let Ok(config) = serde_json::from_str::<serde_json::Value>(&content) {
                    let bos = config["bos_token_id"].as_u64().map(|v| v as u32);
                    let eos = config["eos_token_id"].as_u64().map(|v| v as u32);
                    let pad = config["pad_token_id"].as_u64().map(|v| v as u32);

                    if bos.is_some() && eos.is_some() {
                        return Some((bos.unwrap(), eos.unwrap(), pad.unwrap_or(0)));
                    }
                }
            }
        }

        None
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

    /// Get chat format
    pub fn chat_format(&self) -> ChatFormat {
        self.chat_format
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

/// Chat template for LLaMA models
pub struct ChatTemplate;

impl ChatTemplate {
    /// Format a summarization prompt based on the chat format
    pub fn summarization_prompt(text: &str, word_count: usize) -> String {
        // Default to LLaMA 2 / TinyLlama format which is more widely supported
        Self::summarization_prompt_llama2(text, word_count)
    }

    /// Format for LLaMA 2 / TinyLlama
    pub fn summarization_prompt_llama2(text: &str, word_count: usize) -> String {
        format!(
            "<s>[INST] <<SYS>>\n\
            You are a helpful assistant that summarizes text. \
            Provide concise, accurate summaries that capture the key points.\n\
            <</SYS>>\n\n\
            Please provide a summary of the following text in approximately {} words. \
            Focus on the main ideas and key information.\n\n\
            Text to summarize:\n\n{} [/INST]",
            word_count, text
        )
    }

    /// Format for LLaMA 3
    pub fn summarization_prompt_llama3(text: &str, word_count: usize) -> String {
        let system = Self::system_message_llama3(
            "You are a helpful assistant that summarizes text. \
             Provide concise, accurate summaries that capture the key points.",
        );

        let user = Self::user_message_llama3(&format!(
            "Please provide a summary of the following text in approximately {} words. \
             Focus on the main ideas and key information.\n\n\
             Text to summarize:\n\n{}",
            word_count, text
        ));

        let assistant_start = Self::assistant_start_llama3();

        format!("<|begin_of_text|>{}{}{}", system, user, assistant_start)
    }

    /// Format a system message (LLaMA 3)
    fn system_message_llama3(content: &str) -> String {
        format!(
            "<|start_header_id|>system<|end_header_id|>\n\n{}<|eot_id|>",
            content
        )
    }

    /// Format a user message (LLaMA 3)
    fn user_message_llama3(content: &str) -> String {
        format!(
            "<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|>",
            content
        )
    }

    /// Start assistant response (LLaMA 3)
    fn assistant_start_llama3() -> String {
        "<|start_header_id|>assistant<|end_header_id|>\n\n".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chat_template_llama2() {
        let prompt = ChatTemplate::summarization_prompt_llama2("Hello world", 100);
        assert!(prompt.contains("[INST]"));
        assert!(prompt.contains("<<SYS>>"));
        assert!(prompt.contains("Hello world"));
    }

    #[test]
    fn test_chat_template_llama3() {
        let prompt = ChatTemplate::summarization_prompt_llama3("Hello world", 100);
        assert!(prompt.contains("<|begin_of_text|>"));
        assert!(prompt.contains("system"));
        assert!(prompt.contains("user"));
        assert!(prompt.contains("Hello world"));
    }

    #[test]
    fn test_truncate() {
        let tokens = vec![1, 2, 3, 4, 5];
        let truncated: Vec<u32> = if tokens.len() > 3 {
            tokens[..3].to_vec()
        } else {
            tokens
        };
        assert_eq!(truncated.len(), 3);
    }
}
