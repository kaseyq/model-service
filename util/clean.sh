#!/bin/bash

# Find and delete .DS_Store files
find . -type f -name ".DS_Store" -delete

# Find and delete ._*.* files
find . -type f -name "._*.*" -delete

echo "Cleanup complete: .DS_Store and ._*.* files deleted."
