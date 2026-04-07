#!/usr/bin/env bash
#
# PDF Summarizer Installation Script
# Automatically detects OS and installs dependencies + the application
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print colored output
info() { echo -e "${BLUE}[INFO]${NC} $1"; }
success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }

# Detect OS
detect_os() {
    case "$(uname -s)" in
        Linux*)     OS="linux";;
        Darwin*)    OS="macos";;
        CYGWIN*|MINGW*|MSYS*) OS="windows";;
        *)          OS="unknown";;
    esac
    
    # Detect Linux distribution
    if [ "$OS" = "linux" ]; then
        if [ -f /etc/os-release ]; then
            . /etc/os-release
            DISTRO=$ID
        elif [ -f /etc/debian_version ]; then
            DISTRO="debian"
        elif [ -f /etc/redhat-release ]; then
            DISTRO="rhel"
        else
            DISTRO="unknown"
        fi
    fi
    
    info "Detected OS: $OS"
    [ "$OS" = "linux" ] && info "Detected Distribution: $DISTRO"
}

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Install Rust if not present
install_rust() {
    if command_exists rustc; then
        RUST_VERSION=$(rustc --version | cut -d' ' -f2)
        info "Rust is already installed (version $RUST_VERSION)"
        return 0
    fi
    
    info "Installing Rust..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source "$HOME/.cargo/env"
    success "Rust installed successfully"
}

# Install system dependencies based on OS
install_dependencies() {
    info "Installing system dependencies..."
    
    case "$OS" in
        linux)
            install_linux_dependencies
            ;;
        macos)
            install_macos_dependencies
            ;;
        windows)
            install_windows_dependencies
            ;;
        *)
            error "Unsupported operating system: $OS"
            ;;
    esac
}

# Linux dependencies
install_linux_dependencies() {
    case "$DISTRO" in
        ubuntu|debian|linuxmint|pop)
            info "Installing dependencies for Debian/Ubuntu..."
            sudo apt-get update
            sudo apt-get install -y \
                build-essential \
                pkg-config \
                libssl-dev \
                libvulkan-dev \
                vulkan-tools \
                cmake \
                git \
                curl
            ;;
        fedora)
            info "Installing dependencies for Fedora..."
            sudo dnf install -y \
                gcc \
                gcc-c++ \
                make \
                pkg-config \
                openssl-devel \
                vulkan-loader-devel \
                vulkan-tools \
                cmake \
                git \
                curl
            ;;
        centos|rhel|rocky|almalinux)
            info "Installing dependencies for RHEL/CentOS..."
            sudo yum install -y epel-release
            sudo yum install -y \
                gcc \
                gcc-c++ \
                make \
                pkgconfig \
                openssl-devel \
                vulkan-loader-devel \
                cmake \
                git \
                curl
            ;;
        arch|manjaro)
            info "Installing dependencies for Arch Linux..."
            sudo pacman -Syu --noconfirm \
                base-devel \
                openssl \
                vulkan-icd-loader \
                vulkan-tools \
                cmake \
                git \
                curl
            ;;
        opensuse*|suse*)
            info "Installing dependencies for openSUSE..."
            sudo zypper install -y \
                gcc \
                gcc-c++ \
                make \
                pkg-config \
                libopenssl-devel \
                libvulkan1 \
                vulkan-tools \
                cmake \
                git \
                curl
            ;;
        *)
            warn "Unknown Linux distribution: $DISTRO"
            warn "Please install the following manually:"
            echo "  - C/C++ compiler (gcc/g++)"
            echo "  - pkg-config"
            echo "  - OpenSSL development libraries"
            echo "  - Vulkan development libraries"
            echo "  - cmake, git, curl"
            read -p "Press Enter to continue anyway, or Ctrl+C to abort..."
            ;;
    esac
    
    success "System dependencies installed"
}

# macOS dependencies
install_macos_dependencies() {
    if ! command_exists brew; then
        info "Installing Homebrew..."
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    fi
    
    info "Installing dependencies via Homebrew..."
    brew install openssl pkg-config cmake
    
    # MoltenVK for Vulkan support on macOS
    if ! brew list molten-vk &>/dev/null; then
        info "Installing MoltenVK for Vulkan support..."
        brew install molten-vk
    fi
    
    success "System dependencies installed"
}

# Windows dependencies (via MSYS2/MinGW or WSL)
install_windows_dependencies() {
    warn "Windows detected. For best results, use WSL2 with Ubuntu."
    warn "If using native Windows:"
    echo "  1. Install Visual Studio Build Tools"
    echo "  2. Install Vulkan SDK from https://vulkan.lunarg.com/"
    echo "  3. Ensure 'cargo' is in your PATH"
    
    if command_exists pacman; then
        info "MSYS2 detected, installing dependencies..."
        pacman -Syu --noconfirm
        pacman -S --noconfirm \
            mingw-w64-x86_64-toolchain \
            mingw-w64-x86_64-openssl \
            mingw-w64-x86_64-pkg-config \
            mingw-w64-x86_64-cmake \
            git \
            curl
        success "Dependencies installed via MSYS2"
    else
        read -p "Press Enter to continue anyway, or Ctrl+C to abort..."
    fi
}

# Build the application
build_application() {
    info "Building PDF Summarizer in release mode..."
    
    # Ensure we're in the project directory
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    cd "$SCRIPT_DIR"
    
    # Build in release mode
    cargo build --release
    
    success "Build completed successfully"
}

# Install the binary
install_binary() {
    info "Installing pdf-summarizer binary..."
    
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    BINARY_PATH="$SCRIPT_DIR/target/release/pdf-summarizer"
    
    if [ ! -f "$BINARY_PATH" ]; then
        error "Binary not found at $BINARY_PATH. Build may have failed."
    fi
    
    # Determine install location
    if [ "$OS" = "macos" ] || [ "$OS" = "linux" ]; then
        INSTALL_DIR="/usr/local/bin"
        
        if [ -w "$INSTALL_DIR" ]; then
            cp "$BINARY_PATH" "$INSTALL_DIR/"
        else
            info "Installing to $INSTALL_DIR (requires sudo)..."
            sudo cp "$BINARY_PATH" "$INSTALL_DIR/"
        fi
        
        success "Installed to $INSTALL_DIR/pdf-summarizer"
    else
        # Windows: copy to user's local bin
        INSTALL_DIR="$HOME/.local/bin"
        mkdir -p "$INSTALL_DIR"
        cp "$BINARY_PATH" "$INSTALL_DIR/"
        
        success "Installed to $INSTALL_DIR/pdf-summarizer"
        warn "Make sure $INSTALL_DIR is in your PATH"
    fi
}

# Verify installation
verify_installation() {
    info "Verifying installation..."
    
    if command_exists pdf-summarizer; then
        VERSION=$(pdf-summarizer --version 2>/dev/null || echo "unknown")
        success "pdf-summarizer is installed and accessible!"
        echo ""
        pdf-summarizer --help | head -20
    else
        warn "pdf-summarizer is not in PATH"
        warn "You may need to add /usr/local/bin to your PATH or restart your shell"
    fi
}

# Print GPU information
check_gpu() {
    info "Checking GPU availability..."
    
    case "$OS" in
        linux)
            if command_exists vulkaninfo; then
                echo ""
                vulkaninfo --summary 2>/dev/null | head -20 || warn "Could not get Vulkan info"
            fi
            if command_exists nvidia-smi; then
                echo ""
                nvidia-smi --query-gpu=name,memory.total --format=csv 2>/dev/null || true
            fi
            ;;
        macos)
            info "macOS uses Metal via MoltenVK for GPU acceleration"
            system_profiler SPDisplaysDataType 2>/dev/null | grep -A2 "Chipset Model" || true
            ;;
    esac
}

# Main installation flow
main() {
    echo ""
    echo "=========================================="
    echo "   PDF Summarizer Installation Script"
    echo "=========================================="
    echo ""
    
    detect_os
    
    echo ""
    echo "This script will:"
    echo "  1. Install system dependencies"
    echo "  2. Install Rust (if not present)"
    echo "  3. Build the application"
    echo "  4. Install the binary to /usr/local/bin"
    echo ""
    
    read -p "Continue? [Y/n] " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]] && [[ ! -z $REPLY ]]; then
        info "Installation cancelled"
        exit 0
    fi
    
    echo ""
    install_dependencies
    echo ""
    install_rust
    echo ""
    build_application
    echo ""
    install_binary
    echo ""
    verify_installation
    echo ""
    check_gpu
    
    echo ""
    echo "=========================================="
    success "Installation complete!"
    echo "=========================================="
    echo ""
    echo "Quick start:"
    echo "  pdf-summarizer -f document.pdf"
    echo ""
    echo "For LLaMA 3 (better quality, requires HuggingFace token):"
    echo "  export HF_TOKEN=your_token_here"
    echo "  pdf-summarizer -f document.pdf"
    echo ""
    echo "For more options:"
    echo "  pdf-summarizer --help"
    echo ""
}

# Run main function
main "$@"
