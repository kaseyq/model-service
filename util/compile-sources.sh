#!/bin/bash

# Delete compiled_sources.txt if it exists
rm -f compiled_sources.txt

# Delete .DS_Store and ._*.* files, excluding storage/, venv/, .git/, .venv/, and __pycache__
find . -type f \( -name ".DS_Store" -o -name "._*.*" \) -not -path "./storage/*" -not -path "./venv/*" -not -path "./.git/*" -not -path "./.venv/*" -not -path "./__pycache__/*" -delete
echo "Cleanup complete: .DS_Store and ._*.* files deleted."

# Find .py, .yml, .yaml, requirements.txt, and Dockerfile* files, excluding specified directories, and append to compiled_sources.txt
find . -type f \( -name "*.py" -o -name "*.yml" -o -name "*.yaml" -o \( -name "requirements.txt" -path "./requirements.txt" \) -o -name "Dockerfile*" \) -not -path "./storage/*" -not -path "./venv/*" -not -path "./.git/*" -not -path "./.venv/*" -not -path "./__pycache__/*" -exec sh -c 'echo "File: {}"; cat "{}"; echo ""' \; > compiled_sources.txt