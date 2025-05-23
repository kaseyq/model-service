#!/bin/bash

# Output file
OUTPUT_FILE="compiled-overview.txt"

# Configuration: Directories and their traversal depths
# Format: "directory:depth" (depth=0 for directory only, 1 for one level deep, -1 for recursive)
# Add or modify entries as needed for different projects
DIRECTORIES=(
    "src:-1"          # Recursive for src
    "util:-1"         # Recursive for util
    "storage:1"       # One level deep for storage
    "venv:0"          # Only the venv directory, no contents
)

# Run cleanup script
if [ -f "./util/clean.sh" ]; then
    chmod +x ./util/clean.sh
    ./util/clean.sh
else
    echo "Warning: ./util/clean.sh not found, skipping cleanup" >&2
fi

# Clear output file if it exists
rm -f "$OUTPUT_FILE"

# Section: Project Description
echo "Project Description:" >> "$OUTPUT_FILE"
if [ -f "project-description.txt" ]; then
    cat "project-description.txt" >> "$OUTPUT_FILE"
else
    echo "project-description.txt not found" >> "$OUTPUT_FILE"
fi
echo "" >> "$OUTPUT_FILE"

# Section: Directory and File Structure
echo "Directory and File Structure:" >> "$OUTPUT_FILE"
{
    # Top-level files and directories (excluding configured directories and venv contents)
    EXCLUDE_TOPLEVEL=""
    for dir_entry in "${DIRECTORIES[@]}"; do
        dir="${dir_entry%%:*}"
        EXCLUDE_TOPLEVEL="$EXCLUDE_TOPLEVEL -not -path ./$dir"
    done
    find . -maxdepth 1 -not -name "__pycache__" -not -name ".DS_Store" -not -name "._*.*" \
        $EXCLUDE_TOPLEVEL -not -path "./venv/*" -not -path "./__pycache__/*" \
        | sort | sed 's|^\./||'

    # Process configured directories
    for dir_entry in "${DIRECTORIES[@]}"; do
        dir="${dir_entry%%:*}"
        depth="${dir_entry#*:}"
        if [ -d "./$dir" ] || [ -f "./$dir" ]; then
            if [ "$depth" -eq 0 ]; then
                echo "$dir"
            elif [ "$depth" -eq 1 ]; then
                find "./$dir" -maxdepth 1 -not -name "__pycache__" -not -name ".DS_Store" -not -name "._*.*" \
                    -not -path "./$dir/*/*" | sort | sed 's|^\./||'
            else  # depth=-1 (recursive)
                find "./$dir" -not -name "__pycache__" -not -name ".DS_Store" -not -name "._*.*" \
                    -not -path "*/__pycache__/*" | sort | sed 's|^\./||'
            fi
        fi
    done
} >> "$OUTPUT_FILE"

echo "" >> "$OUTPUT_FILE"
echo "Generated on $(date)" >> "$OUTPUT_FILE"