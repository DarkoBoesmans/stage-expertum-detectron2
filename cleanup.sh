#!/bin/bash

# Cleanup script for Detectron2 project
# This script reorganizes the project files, removing unnecessary files
# and creating a clean structure for handover.

echo "===== DETECTRON2 PROJECT CLEANUP ====="
echo "This script will organize your project files for handover."

# Create backup directory
BACKUP_DIR="/home/dboesmans/projects/detectron2_backup_$(date +%Y%m%d_%H%M%S)"
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
    "check_model.py"
    "predict.py"
    "util.py"
    "loss.py"
    "class.names"
    "requirements.txt"
)

# Documentation files (minimal essential documentation)
DOC_FILES=(
    "README.md"
    "HANDOVER.md"
)

# Utility scripts (only the most important)
UTIL_FILES=(
    "setup.sh"
)

# All essential files combined
KEEP_FILES=("${CORE_FILES[@]}" "${DOC_FILES[@]}" "${UTIL_FILES[@]}")

# Move to project directory
cd /home/dboesmans/projects/detectron2

# Create a list of all files (excluding directories)
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

# Consolidate documentation before removing files
echo ""
echo "Consolidating documentation into README.md..."

# Extract useful information from TROUBLESHOOTING.md if it exists
if [ -f "TROUBLESHOOTING.md" ]; then
    log "Extracting troubleshooting information"
    echo -e "\n\n## Troubleshooting\n" >> README.md
    echo "Common issues and their solutions can be found in the backup directory in TROUBLESHOOTING.md" >> README.md
    echo -e "\nKey troubleshooting tips:\n" >> README.md
    
    # Extract just the headers from TROUBLESHOOTING.md to add to README
    grep "^###" TROUBLESHOOTING.md | sed 's/^### /- /' >> README.md
fi

# Extract useful information from COMMAND_REFERENCE.md if it exists
if [ -f "COMMAND_REFERENCE.md" ]; then
    log "Extracting command reference information"
    echo -e "\n\n## Command References\n" >> README.md
    echo "Detailed command references can be found in the backup directory in COMMAND_REFERENCE.md" >> README.md
    echo -e "\nAll scripts support --help to show available options.\n" >> README.md
fi

# Move files to backup
for file in "${BACKUP_FILES[@]}"; do
    if [ -f "$file" ]; then
        log "Moving $file to backup"
        mv "$file" "$BACKUP_DIR/"
    fi
done

# Clean up __pycache__ directories
if [ -d "__pycache__" ]; then
    log "Cleaning __pycache__ directory"
    find . -name "__pycache__" -type d -exec rm -rf {} +
fi

# Clean up any leftover output directories
read -p "Do you want to clean up the output/ and predictions_all/ directories? (y/n): " clean_output
if [[ $clean_output == [Yy]* ]]; then
    if [ -d "output" ]; then
        log "Backing up output/ directory"
        cp -r output "$BACKUP_DIR/"
        rm -rf output/*
        log "Emptied output/ directory"
    fi
    
    if [ -d "predictions_all" ]; then
        log "Backing up predictions_all/ directory"
        mv predictions_all "$BACKUP_DIR/"
    fi
    
    # Create empty directories as needed
    mkdir -p output
    mkdir -p predictions
    log "Created empty output directories"
fi

# Create a simple README in the backup directory explaining what's there
cat > "$BACKUP_DIR/README.md" << EOF
# Detectron2 Backup Files

This directory contains backup files from the Detectron2 project cleanup performed on $(date).

## Contents

### Documentation
- TROUBLESHOOTING.md: Common issues and solutions
- COMMAND_REFERENCE.md: Command-line options for all scripts

### Scripts
- Various analysis and visualization scripts
- Alternative training and prediction implementations

These files were removed from the main project to create a streamlined version
for handover, but are preserved here for reference if needed.

See cleanup_log.txt for a detailed list of all files and actions taken.
EOF

# Final summary
log ""
log "=== CLEANUP COMPLETED ==="
log "$(date)"
log "Essential files kept in project directory:"
for file in "${KEEP_FILES[@]}"; do
    if [ -f "$file" ]; then
        log "  - $file"
    fi
done
log "Non-essential files moved to: $BACKUP_DIR"

echo ""
echo "===== CLEANUP COMPLETED SUCCESSFULLY ====="
echo ""
echo "The project has been streamlined to just the essential components:"
echo "  - Core training script (train.py)"
echo "  - Memory-efficient training (check_model.py)"
echo "  - Prediction script (predict.py)"
echo "  - Core utilities (util.py, loss.py)"
echo "  - Documentation (README.md, HANDOVER.md)"
echo "  - Environment setup (setup.sh, requirements.txt)"
echo ""
echo "All other files were backed up to: $BACKUP_DIR"
echo "Key information from supplementary documentation has been added to README.md"
echo ""
echo "If you need any of the backed up files, you can find them in the backup directory."
echo "See $CLEANUP_LOG for a detailed log of the cleanup process."
