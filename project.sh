#!/bin/bash
# Unified Management Script for Detectron2 Project
# Handles setup, testing, cleanup and more

# Check script name - allows the script to behave differently based on symlinks
SCRIPT_NAME=$(basename "$0")

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

# Print banner with script name
print_banner() {
    echo -e "${BLUE}================================================================${NC}"
    echo -e "${BLUE}    DETECTRON2 PROJECT - $1${NC}"
    echo -e "${BLUE}================================================================${NC}"
    echo ""
}

# Function to check command status
check_status() {
    if [ $? -ne 0 ]; then
        echo -e "${RED}Error: $1 failed. Please check the error messages above.${NC}"
        exit 1
    fi
}

# Function to check if a Python package is installed
check_package() {
    python -c "import $1" &> /dev/null
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ $1 is installed${NC}"
        return 0
    else
        echo -e "${YELLOW}✗ $1 is not installed${NC}"
        return 1
    fi
}

# Setup function - creates environment and installs dependencies
setup_environment() {
    print_banner "ENVIRONMENT SETUP"
    echo "This will set up the necessary environment for the Detectron2 project."
    
    # Check if Python is installed
    echo "Checking if Python is installed..."
    python --version
    check_status "Python check"
    echo -e "${GREEN}✓ Python is installed${NC}"

    # Create virtual environment if it doesn't exist
    if [ ! -d "detectron2-env" ]; then
        echo "Creating virtual environment..."
        python -m venv detectron2-env
        check_status "Virtual environment creation"
        echo -e "${GREEN}✓ Virtual environment created${NC}"
    else
        echo -e "${GREEN}✓ Virtual environment already exists${NC}"
    fi

    # Activate virtual environment
    echo "Activating virtual environment..."
    source detectron2-env/bin/activate
    check_status "Virtual environment activation"
    echo -e "${GREEN}✓ Virtual environment activated${NC}"

    # Update pip
    echo "Updating pip..."
    pip install --upgrade pip
    check_status "Pip update"
    echo -e "${GREEN}✓ Pip updated${NC}"

    # Install dependencies
    echo "Installing required packages (this may take a while)..."
    pip install -r requirements.txt
    check_status "Package installation"
    echo -e "${GREEN}✓ Packages installed${NC}"

    # Check if CUDA is available
    echo "Checking for CUDA availability..."
    python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"

    echo ""
    echo -e "${GREEN}===== Setup Complete =====${NC}"
    echo "To activate the environment in the future, run:"
    echo "source detectron2-env/bin/activate"
    echo ""
    echo "To start training with minimal configuration:"
    echo "python train.py --data-dir ./data --class-list ./class.names"
    echo ""
    echo "For more options, see README.md or run:"
    echo "python train.py --help"
}

# Test function - checks if everything is set up correctly
test_environment() {
    print_banner "ENVIRONMENT TEST"
    echo "This will test if the core functionality of the project is working."

    # Controleer of Python beschikbaar is
    echo "Checking for Python..."
    python --version
    if [ $? -ne 0 ]; then
        echo -e "${RED}ERROR: Python not found${NC}"
        exit 1
    fi
    echo -e "${GREEN}✓ Python available${NC}"

    # Controleer of de virtuele omgeving bestaat
    echo "Checking for virtual environment..."
    if [ ! -d "detectron2-env" ]; then
        echo -e "${YELLOW}WARNING: detectron2-env directory not found. Use 'project.sh setup' to create the environment.${NC}"
    else
        echo -e "${GREEN}✓ Virtual environment found${NC}"
    fi

    # Activeer de omgeving als deze bestaat
    if [ -f "detectron2-env/bin/activate" ]; then
        echo "Activating virtual environment..."
        source detectron2-env/bin/activate
        echo -e "${GREEN}✓ Virtual environment activated${NC}"
    fi

    # Controleer benodigde bestanden
    echo "Checking for essential files..."
    ESSENTIALS=("train.py" "predict.py" "util.py" "loss.py" "class.names" "requirements.txt")
    MISSING=0
    for file in "${ESSENTIALS[@]}"; do
        if [ ! -f "$file" ]; then
            echo -e "${RED}ERROR: $file not found${NC}"
            MISSING=1
        fi
    done

    if [ $MISSING -eq 0 ]; then
        echo -e "${GREEN}✓ All essential files found${NC}"
    else
        echo -e "${YELLOW}WARNING: Some essential files are missing${NC}"
    fi

    # Controleer of de virtuele omgeving correct is geïnstalleerd
    echo "Checking for PyTorch and other dependencies..."
    python -c "import torch; print(f'PyTorch version: {torch.__version__}')" 2>/dev/null
    if [ $? -ne 0 ]; then
        echo -e "${YELLOW}WARNING: PyTorch not found. Use 'project.sh setup' to install dependencies.${NC}"
    else
        echo -e "${GREEN}✓ PyTorch installed${NC}"
    fi

    python -c "import detectron2; print(f'Detectron2 available')" 2>/dev/null
    if [ $? -ne 0 ]; then
        echo -e "${YELLOW}WARNING: Detectron2 not found. Use 'project.sh setup' to install dependencies.${NC}"
    else
        echo -e "${GREEN}✓ Detectron2 installed${NC}"
    fi

    # Controleer of de datastructuur klopt
    echo "Checking data structure..."
    if [ -d "data/train/imgs" ] && [ -d "data/train/anns" ] && [ -d "data/val/imgs" ] && [ -d "data/val/anns" ]; then
        echo -e "${GREEN}✓ Data structure is correct${NC}"
        TRAIN_IMGS=$(ls data/train/imgs 2>/dev/null | wc -l)
        TRAIN_ANNS=$(ls data/train/anns 2>/dev/null | wc -l)
        VAL_IMGS=$(ls data/val/imgs 2>/dev/null | wc -l)
        VAL_ANNS=$(ls data/val/anns 2>/dev/null | wc -l)
        echo "  - Training images: $TRAIN_IMGS"
        echo "  - Training annotations: $TRAIN_ANNS"
        echo "  - Validation images: $VAL_IMGS"
        echo "  - Validation annotations: $VAL_ANNS"

        # Check for mismatch between images and annotations
        if [ $TRAIN_IMGS -ne $TRAIN_ANNS ]; then
            echo -e "${YELLOW}WARNING: Number of training images ($TRAIN_IMGS) does not match annotations ($TRAIN_ANNS)${NC}"
        fi
        if [ $VAL_IMGS -ne $VAL_ANNS ]; then
            echo -e "${YELLOW}WARNING: Number of validation images ($VAL_IMGS) does not match annotations ($VAL_ANNS)${NC}"
        fi
    else
        echo -e "${YELLOW}WARNING: Data structure does not seem to be complete${NC}"
    fi

    # Testafbeeldingen controleren
    if [ -d "img" ]; then
        IMG_COUNT=$(ls img 2>/dev/null | wc -l)
        echo -e "${GREEN}✓ Test images directory found with $IMG_COUNT images${NC}"
    else
        echo -e "${YELLOW}WARNING: No img/ directory found for test images${NC}"
    fi

    # Check GPU/CPU for training recommendation
    echo ""
    echo "Hardware detection for training:"
    if python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null | grep -q "True"; then
        echo -e "${GREEN}✓ GPU detected, can use CUDA for faster training${NC}"
        echo "  Recommended: python train.py --device cuda"
    else
        echo -e "${YELLOW}✓ No GPU detected, will use CPU for training${NC}"
        echo "  Recommended: python train.py --device cpu --mini-batch-size 2"
    fi

    echo ""
    echo -e "${GREEN}===== TEST COMPLETED =====${NC}"
    echo ""
    echo "To train a model, use:"
    echo "  python train.py"
    echo ""
    echo "For predictions after training, use:"
    echo "  python predict.py --weights ./output/model_final.pth --input ./img --output ./predictions"
    echo ""
}

# Cleanup function - backs up and cleans the project
cleanup_project() {
    print_banner "PROJECT CLEANUP"
    echo "This will organize your project files and create a backup."

    # Create backup directory
    BACKUP_DIR="$(pwd)_backup_$(date +%Y%m%d_%H%M%S)"
    CLEANUP_LOG="$BACKUP_DIR/cleanup_log.txt"
    mkdir -p "$BACKUP_DIR"
    echo "Created backup directory: $BACKUP_DIR"

    # Function to log actions
    log() {
        echo "$1" | tee -a "$CLEANUP_LOG"
    }

    log "Cleanup started at $(date)"

    # Core files (absolutely essential for project functionality)
    CORE_FILES=(
        "train.py"
        "predict.py"
        "util.py"
        "loss.py"
        "class.names"
        "requirements.txt"
        "project.sh"
    )

    # Documentation files (minimal essential documentation)
    DOC_FILES=(
        "README.md"
        "HANDOVER.md"
    )

    # All essential files combined
    KEEP_FILES=("${CORE_FILES[@]}" "${DOC_FILES[@]}")

    # Create a list of all files (excluding directories and . files)
    ALL_FILES=$(find . -maxdepth 1 -type f -not -path "*/\.*" | sed 's|^\./||')

    # Identify files to backup/remove
    BACKUP_FILES=()
    for file in $ALL_FILES; do
        if [[ ! " ${KEEP_FILES[@]} " =~ " ${file} " ]]; then
            BACKUP_FILES+=("$file")
        fi
    done

    # Log the plan
    log ""
    log "=== CLEANUP PLAN ==="
    log "Files to keep:"
    for file in "${KEEP_FILES[@]}"; do
        if [ -f "$file" ]; then
            log "  ✓ $file"
        else
            log "  ! $file (missing)"
        fi
    done

    log ""
    log "Files to move to backup:"
    for file in "${BACKUP_FILES[@]}"; do
        log "  - $file"
    done

    # Data directories to preserve:
    log ""
    log "Directories:"
    log "  ✓ data/ (training data)"
    log "  ✓ img/ (test images)"
    log "  ✓ detectron2-env/ (virtual environment)"

    # Ask for confirmation
    echo ""
    echo "This will move ${#BACKUP_FILES[@]} files to backup directory $BACKUP_DIR"
    echo "Please review the plan above."
    read -p "Continue with cleanup? (y/n): " confirm
    if [[ $confirm != [Yy]* ]]; then
        log "Cleanup aborted by user"
        exit 1
    fi

    # Backup files
    echo ""
    echo "Moving files to backup..."
    for file in "${BACKUP_FILES[@]}"; do
        if [ -f "$file" ]; then
            cp "$file" "$BACKUP_DIR/" && log "  - Backed up: $file"
        fi
    done

    # Only remove if backup was successful
    echo "Removing backed up files..."
    for file in "${BACKUP_FILES[@]}"; do
        if [ -f "$BACKUP_DIR/$file" ]; then
            rm "$file" && log "  - Removed: $file"
        else
            log "  ! Failed to backup $file, not removing"
        fi
    done

    log ""
    log "Cleanup completed at $(date)"
    echo -e "${GREEN}===== CLEANUP COMPLETED =====${NC}"
    echo "Backup saved to: $BACKUP_DIR"
    echo "Cleanup log: $CLEANUP_LOG"
}

# Help function - shows available commands
show_help() {
    print_banner "HELP"
    echo "Usage: ./project.sh [command]"
    echo ""
    echo "Available commands:"
    echo "  setup       - Set up the environment and install dependencies"
    echo "  test        - Test if everything is set up correctly"
    echo "  cleanup     - Back up and clean the project" 
    echo "  all         - Run setup and test in sequence (recommended)"
    echo "  help        - Show this help message"
    echo ""
    echo "Examples:"
    echo "  ./project.sh          # Run complete setup and test (same as 'all')"
    echo "  ./project.sh setup    # Only set up the project"
    echo "  ./project.sh test     # Only test the project setup"
    echo ""
}

# Function to run all steps in sequence
run_all() {
    print_banner "COMPLETE SETUP AND TEST"
    echo -e "${PURPLE}This will run setup, test, and prepare the system for training.${NC}"
    echo ""
    
    # Ask for confirmation
    read -p "Do you want to proceed with complete setup? (y/n): " confirm
    if [[ $confirm != [Yy]* ]]; then
        echo "Operation cancelled."
        exit 0
    fi
    
    echo -e "\n${PURPLE}Step 1: Setting up environment...${NC}"
    setup_environment
    
    echo -e "\n${PURPLE}Step 2: Testing environment...${NC}"
    test_environment
    
    # Check if data exists
    if [ ! -d "data/train/imgs" ] || [ ! -d "data/train/anns" ] || [ $(ls data/train/imgs 2>/dev/null | wc -l) -eq 0 ]; then
        echo -e "\n${YELLOW}WARNING: No training data found or data folder is empty.${NC}"
        echo "You need to add data before training."
        echo "Make sure your data follows this structure:"
        echo "  data/train/imgs/ - Contains training images"
        echo "  data/train/anns/ - Contains training annotations"
        echo "  data/val/imgs/ - Contains validation images"
        echo "  data/val/anns/ - Contains validation annotations"
    else
        echo -e "\n${GREEN}Data structure looks good. You can start training.${NC}"
    fi
    
    # Final instructions
    echo -e "\n${PURPLE}======================================${NC}"
    echo -e "${GREEN}✓ Setup complete!${NC}"
    echo -e "${PURPLE}======================================${NC}"
    echo ""
    echo "To start training:"
    echo "  python train.py"
    echo ""
    echo "For predictions after training:"
    echo "  python predict.py --weights ./output/model_final.pth --input ./img --output ./predictions"
    echo ""
}

# Main execution logic
case "$1" in
    setup)
        setup_environment
        ;;
    test)
        test_environment
        ;;
    cleanup)
        cleanup_project
        ;;
    all)
        run_all
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        # If no arguments provided or unknown command
        if [ -z "$1" ]; then
            run_all
        else
            show_help
        fi
        ;;
esac
