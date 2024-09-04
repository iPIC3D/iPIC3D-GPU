#!/bin/bash

if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <path1> <path2> <float>"
    exit 1
fi

path1="$1"
path2="$2"
float="$3"

if [ ! -d "$path1" ]; then
    echo "Directory $path1 does not exist."
    exit 1
fi

if [ ! -d "$path2" ]; then
    echo "Directory $path2 does not exist."
    exit 1
fi

compareExecutable="./build/CompareVTKFiles"

for file1 in "$path1"/*.vtk; do
    fileName=$(basename "$file1")
    file2="$path2/$fileName"

    # file with the same name
    if [ -f "$file2" ]; then
        echo "[*]$fileName: "
        "$compareExecutable" "$file1" "$file2" "$float"
    fi
done
