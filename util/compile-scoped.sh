#!/bin/bash

# Default output file and settings
OUTPUT_FILE="compiled-scoped.txt"
RUN_CLEANUP=1  # Default: run cleanup
LOG_LINES=50   # Default: last 50 lines for *.log files

# Usage message
usage() {
    echo "Usage: $0 [-o output_file] [-c {0|1}] [-f path_file] [-l log_lines] [file_path ...]"
    echo "  -o output_file: Specify output file (default: compiled-scoped.txt)"
    echo "  -c {0|1}: Run cleanup (1) or skip (0) (default: 1)"
    echo "  -f path_file: Read file paths from a file"
    echo "  -l log_lines: Number of lines for *.log files (default: 50)"
    echo "  file_path: List of file paths to concatenate (any file, including storage/ and venv/)"
    exit 1
}

# Parse options
while getopts "o:c:f:l:" opt; do
    case $opt in
        o) OUTPUT_FILE="$OPTARG" ;;
        c) RUN_CLEANUP="$OPTARG" ;;
        f) PATH_FILE="$OPTARG" ;;
        l) LOG_LINES="$OPTARG" ;;
        *) usage ;;
    esac
done
shift $((OPTIND-1))

# Validate RUN_CLEANUP
if [[ "$RUN_CLEANUP" != "0" && "$RUN_CLEANUP" != "1" ]]; then
    echo "Error: -c must be 0 or 1" >&2
    exit 1
fi

# Validate LOG_LINES
if ! [[ "$LOG_LINES" =~ ^[0-9]+$ ]] || [ "$LOG_LINES" -lt 0 ]; then
    echo "Error: -l must be a non-negative integer" >&2
    exit 1
fi

# Collect file paths
FILE_PATHS=()
if [ -n "$PATH_FILE" ]; then
    if [ -f "$PATH_FILE" ]; then
        while IFS= read -r line; do
            # Skip empty lines or comments
            [[ -z "$line" || "$line" =~ ^[[:space:]]*# ]] && continue
            FILE_PATHS+=("$line")
        done < "$PATH_FILE"
    else
        echo "Error: Path file '$PATH_FILE' not found" >&2
        exit 1
    fi
fi
# Add command-line file paths
FILE_PATHS+=("$@")

# Check if any file paths were provided
if [ ${#FILE_PATHS[@]} -eq 0 ]; then
    echo "Error: No file paths provided" >&2
    usage
fi

# Run cleanup script if enabled
if [ "$RUN_CLEANUP" -eq 1 ]; then
    if [ -f "./util/clean.sh" ]; then
        chmod +x ./util/clean.sh
        ./util/clean.sh
    else
        echo "Warning: ./util/clean.sh not found, skipping cleanup" >&2
    fi
fi

# Clear output file
rm -f "$OUTPUT_FILE"

# Process each file
for file in "${FILE_PATHS[@]}"; do
    if [ -f "$file" ]; then
        echo "File: $file" >> "$OUTPUT_FILE"
        # Check if file ends with .log
        if [[ "$file" =~ \.log$ ]]; then
            tail -n "$LOG_LINES" "$file" >> "$OUTPUT_FILE"
        else
            cat "$file" >> "$OUTPUT_FILE"
        fi
        echo "" >> "$OUTPUT_FILE"
    else
        echo "Warning: File '$file' not found, skipping" >&2
    fi
done

echo "Output generated in $OUTPUT_FILE"