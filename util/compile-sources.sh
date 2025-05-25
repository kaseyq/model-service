#!/bin/bash

# Output file
OUTPUT_FILE="compiled_sources.txt"

# Run cleanup script
if [ -f "./util/clean.sh" ]; then
    chmod +x ./util/clean.sh
    ./util/clean.sh
else
    echo "Warning: ./util/clean.sh not found, skipping cleanup" >&2
fi

# Clear output file if it exists
rm -f "$OUTPUT_FILE"


# Delete .DS_Store and ._*.* files, excluding storage/, venv/, .git/, .venv/, and __pycache__
#find . -type f \( -name ".DS_Store" -o -name "._*.*" \) -not -path "./storage/*" -not -path "./venv/*" -not -path "./.git/*" -not -path "./.venv/*" -not -path "./__pycache__/*" -delete
#echo "Cleanup complete: .DS_Store and ._*.* files deleted."

echo "Compiled Sources:" >> "$OUTPUT_FILE"

# Find .py, .yml, .yaml, requirements.txt, Dockerfile*, requirements_docker.txt, and project-description.txt files, excluding specified directories, and append to compiled_sources.txt
find . -type f \( -name "*.py" -o -name "*.yml" -o -name "*.yaml" -o \( -name "requirements.txt" -path "./requirements.txt" \) -o -name "Dockerfile*" -o -name "requirements_docker.txt" -o -name "project-description.txt" \) -not -path "./storage/*" -not -path "./tests/*" -not -path "./venv/*" -not -path "./.git/*" -not -path "./.venv/*" -not -path "./__pycache__/*" -exec sh -c 'echo "File: {}"; cat "{}"; echo ""' \; > compiled_sources.txt


echo "" >> "$OUTPUT_FILE"
echo "Generated on $(date)" >> "$OUTPUT_FILE"