# PDF Summarizer

A production-ready CLI tool for PDF text extraction and summarization using LLaMA 3 with GPU acceleration via the Burn framework.

## Features

- Extract text from multiple PDF files or entire directories
- Generate concise summaries using LLaMA 3
- GPU acceleration via WebGPU (CUDA, Vulkan, Metal support)
- Configurable summary length and generation parameters
- Output to stdout or file
- Progress tracking for batch processing
- Robust error handling with skip option for problematic files

## Requirements

- Rust 1.75+ (2021 edition)
- GPU with Vulkan, CUDA, or Metal support (or use CPU mode)
- LLaMA 3 model weights in SafeTensors format
- LLaMA 3 tokenizer (tokenizer.json)

## Installation

### Quick Install (Recommended)

Use the installation script that automatically detects your OS and installs all dependencies:

```bash
# Clone the repository
git clone https://github.com/sahastasai/PDF-Summarizer.git
cd PDF-Summarizer

# Run the installer
./install.sh
```

The installer supports:
- **Linux**: Ubuntu/Debian, Fedora, CentOS/RHEL, Arch, openSUSE
- **macOS**: Intel and Apple Silicon (via Homebrew)
- **Windows**: WSL2 recommended, or MSYS2

### Manual Installation

```bash
# Clone the repository
git clone https://github.com/sahastasai/PDF-Summarizer.git
cd PDF-Summarizer

# Build in release mode for best performance
cargo build --release

# Install to /usr/local/bin (optional)
sudo cp target/release/pdf-summarizer /usr/local/bin/
```

### Dependencies

The project uses the following key dependencies:
- `burn` + `burn-wgpu` - ML framework with GPU support
- `clap` - CLI argument parsing
- `pdf-extract` + `lopdf` - PDF text extraction
- `tokenizers` - HuggingFace tokenizers
- `safetensors` - Model weight loading
- `indicatif` - Progress bars

## Usage

### Basic Usage

```bash
# Summarize a single PDF
pdf-summarizer -f document.pdf -m /path/to/llama3-model

# Summarize multiple PDFs
pdf-summarizer -f doc1.pdf,doc2.pdf,doc3.pdf -m /path/to/llama3-model

# Summarize all PDFs in a folder
pdf-summarizer -F /path/to/pdf/folder -m /path/to/llama3-model
```

### Command Line Options

```
USAGE:
    pdf-summarizer [OPTIONS]

OPTIONS:
    -s, --summary-length <NUMBER>
            Length of the summary to generate in words [default: 250]

    -f, --files <PATHS>
            Paths to individual PDF files to process (comma-separated)

    -F, --folder <PATH>
            Path to a folder containing PDF files to process

    -o, --output <PATH>
            Path to the output file (must be .txt, otherwise outputs to stdout)

    -m, --model <PATH>
            Path to the LLaMA 3 model directory [env: LLAMA_MODEL_PATH]
            If not provided, model will be auto-downloaded to ~/.pdf_summarizer/models/

    -t, --tokenizer <PATH>
            Path to the tokenizer file [env: LLAMA_TOKENIZER_PATH]

        --hf-token <TOKEN>
            HuggingFace API token for downloading gated models [env: HF_TOKEN]

        --cpu
            Use CPU instead of GPU [default: false]

        --gpu-device <NUMBER>
            GPU device index to use [default: 0]

        --max-context <NUMBER>
            Maximum context length for the model [default: 4096]

        --temperature <FLOAT>
            Temperature for text generation (0.0 - 2.0) [default: 0.7]

        --top-p <FLOAT>
            Top-p (nucleus) sampling parameter [default: 0.9]

        --top-k <NUMBER>
            Top-k sampling parameter [default: 40]

    -v, --verbose
            Verbose output (-v info, -vv debug, -vvv trace)

        --batch-size <NUMBER>
            Batch size for processing multiple PDFs [default: 1]

        --skip-errors
            Skip PDFs that fail to parse [default: false]

    -h, --help
            Print help information

    -V, --version
            Print version information
```

### Examples

```bash
# Generate a 100-word summary
pdf-summarizer -f report.pdf -m ./llama3-8b -s 100

# Process folder and save to file
pdf-summarizer -F ./documents -m ./llama3-8b -o summaries.txt

# Use CPU mode with verbose output
pdf-summarizer -f doc.pdf -m ./llama3-8b --cpu -vv

# Custom generation parameters
pdf-summarizer -f doc.pdf -m ./llama3-8b --temperature 0.5 --top-p 0.95 --top-k 50

# Skip problematic PDFs in batch processing
pdf-summarizer -F ./mixed-docs -m ./llama3-8b --skip-errors
```

## Model Setup

### Automatic Model Download

The application automatically downloads a model if none is provided:

**Without HuggingFace token** (uses TinyLlama 1.1B - open model, no signup required):
```bash
# Just run it - TinyLlama will be downloaded automatically
pdf-summarizer -f document.pdf
```

**With HuggingFace token** (uses LLaMA 3 8B Instruct - better quality):
```bash
# With HuggingFace token for LLaMA 3
pdf-summarizer -f document.pdf --hf-token YOUR_HF_TOKEN

# Or set environment variable
export HF_TOKEN=YOUR_HF_TOKEN
pdf-summarizer -f document.pdf
```

**Note:** LLaMA 3 is a gated model. To use it:
1. Visit https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct and accept the license
2. Create an access token at https://huggingface.co/settings/tokens
3. Provide the token via `--hf-token` or `HF_TOKEN` environment variable

Models are cached at `~/.pdf_summarizer/models/` and reused for subsequent runs.

### Manual Model Setup

If you prefer to download the model manually:

1. Request access to LLaMA 3 from Meta
2. Download the model weights in SafeTensors format
3. Ensure you have the following files in your model directory:
   - `config.json` - Model configuration
   - `tokenizer.json` - Tokenizer file
   - `*.safetensors` - Model weight files

### Using Environment Variables

```bash
# Set model path globally
export LLAMA_MODEL_PATH=/path/to/llama3-8b
export LLAMA_TOKENIZER_PATH=/path/to/llama3-8b/tokenizer.json

# Then run without -m flag
pdf-summarizer -f document.pdf
```

## GPU Support

The application uses WebGPU for cross-platform GPU acceleration:

- **CUDA** (NVIDIA GPUs) - Best performance on NVIDIA hardware
- **Vulkan** (Cross-platform) - Works on most modern GPUs
- **Metal** (macOS) - Native Apple Silicon support

### Selecting GPU Device

```bash
# Use first GPU (default)
pdf-summarizer -f doc.pdf -m ./model --gpu-device 0

# Use second GPU
pdf-summarizer -f doc.pdf -m ./model --gpu-device 1

# Fall back to CPU
pdf-summarizer -f doc.pdf -m ./model --cpu
```

## Architecture

```
pdf-summarizer/
├── src/
│   ├── main.rs           # Application entry point
│   ├── cli/              # CLI argument parsing
│   ├── pdf/              # PDF text extraction
│   ├── llm/              # LLaMA 3 model implementation
│   │   ├── config.rs     # Model configuration
│   │   ├── model.rs      # Main model
│   │   ├── attention.rs  # GQA implementation
│   │   ├── layers.rs     # RMSNorm, MLP layers
│   │   └── loader.rs     # Weight loading
│   ├── model_manager/    # Auto-download & caching
│   │   ├── mod.rs        # Main module
│   │   ├── cache.rs      # Cache directory management
│   │   ├── downloader.rs # HuggingFace download
│   │   └── validator.rs  # Model validation
│   ├── tokenizer/        # Tokenization
│   ├── pipeline/         # Summarization pipeline
│   ├── output/           # Output formatting
│   └── error/            # Error handling
├── Cargo.toml
└── README.md
```

## Performance Tips

1. **Use Release Build**: Always compile with `--release` for optimal performance
2. **GPU Memory**: LLaMA 3 8B requires ~16GB VRAM. Use `--cpu` for systems with less VRAM
3. **Batch Processing**: Process multiple PDFs at once for better throughput
4. **Context Length**: Reduce `--max-context` if running out of memory

## Troubleshooting

### Common Issues

**GPU not detected:**
```bash
# Check available GPUs
vulkaninfo | grep -i "device name"  # For Vulkan
nvidia-smi                           # For CUDA
```

**Out of memory:**
```bash
# Use CPU mode or reduce context
pdf-summarizer -f doc.pdf -m ./model --cpu
# or
pdf-summarizer -f doc.pdf -m ./model --max-context 2048
```

**PDF extraction fails:**
```bash
# Skip problematic files
pdf-summarizer -F ./docs -m ./model --skip-errors
```

## License

MIT License - See LICENSE file for details.

## Contributing

Contributions are welcome! Please read the contributing guidelines before submitting PRs.
