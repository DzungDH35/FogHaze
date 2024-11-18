#!/bin/bash

: '
This file is used to partition files into GT and hazy folder pairs, 
each folder contains up to FILE_BATCH_SIZE files (for example FILE_BATCH_SIZE = 20).

NOTE: Only run it to preprocess the dataset fetch from the original paper/repo
'

readonly DATASET_DIR="./HSTS"

cd "$DATASET_DIR" || exit 1
echo "Start processing $DATASET_DIR..."

rename_files_in_ascending_order() {
    current_dir=$(pwd)
    target_dir="${1:-$(pwd)}"
    
    cd "$target_dir" || return 1

    counter=1
    for file in *; do
        mv "$file" "$counter.${file##*.}"
        ((counter++))
    done

    cd "$current_dir" || return 1
}

reorganize_synthetic_files() {
    cd "synthetic" || return 1
    mv "./original" "./GT"
    mv "./synthetic" "./hazy"

    rename_files_in_ascending_order "./GT"
    rename_files_in_ascending_order "./hazy"
}

rename_files_in_ascending_order "./real-world"
reorganize_synthetic_files
