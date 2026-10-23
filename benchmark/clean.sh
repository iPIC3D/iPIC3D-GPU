#!/bin/bash

read -p "This will delete the output and build. Are you sure? (y/N): " confirm

if [[ "$confirm" =~ ^[Yy]$ ]]; then

    rm -rf build

    for dir in */; do
        find "$dir" -type f ! -name '*.inp' -exec rm -f {} +
        find "$dir" -type d -not -path "$dir" -exec rm -rf {} +
    done

    echo "[*]Clean up completed."

else
    echo "Operation canceled."
fi