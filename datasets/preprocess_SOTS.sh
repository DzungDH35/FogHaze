#!/bin/bash

: '
This file is used to partition files into GT and hazy folder pairs, 
each folder contains up to FILE_BATCH_SIZE files (for example FILE_BATCH_SIZE = 20).

NOTE: Only run it to preprocess the dataset fetch from the original paper/repo
'

readonly DATASET_DIR="./SOTS"
readonly FILE_BATCH_SIZE=20

cd "$DATASET_DIR" || exit 1
echo "Start processing $DATASET_DIR..."

paritition_files_into_dirs() {
    current_dir=$(pwd)
    target_dir="${1:-$(pwd)}"
    local dir_prefix=$2

    cd "$target_dir" || return 1

    dir_index=1
    file_count=0

    for file in *; do
        mkdir -p "${dir_prefix}_${dir_index}"
        mv "$file" "${dir_prefix}_${dir_index}/"

        ((file_count++))

        if ((file_count % FILE_BATCH_SIZE == 0)); then
            ((dir_index++))
        fi
    done

    cd "$current_dir" || return 1
}

partition_indoor_hazy_files() {
    cd "./hazy" || return 1

    for file in *; do
        filename=$(basename "$file")
        filename="${filename%.*}"
        suffix=${filename#*_}

        degree_dir="degree_$suffix"
        new_file=$(echo "$file" | cut -d'_' -f1).${file##*.}

        mkdir -p "$degree_dir"
        mv "$file" "$degree_dir/$new_file"
    done

    for degree_dir in *; do
        paritition_files_into_dirs "$degree_dir" "hazy"
    done

    cd ".." || return 1
}

partition_outdoor_hazy_files() {
    cd "./hazy" || return 1

    for file in *; do
        new_name=$(echo "$file" | cut -d'_' -f1).${file##*.}

        if [ -e "$new_name" ]; then
            rm "$file"
            echo "Remove duplicate file $file"
        else
            mv "$file" "$new_name"
        fi
    done

    paritition_files_into_dirs "." "hazy"

    cd ".." || return 1
}

# Preprocess indoor dataset
cd ./indoor || exit
mv "./gt" "./GT"
paritition_files_into_dirs "./GT" "GT"
partition_indoor_hazy_files
mv ./GT/* ./ && rmdir ./GT

# Preprocess outdoor dataset
cd ../outdoor || exit
mv "./gt" "./GT"
paritition_files_into_dirs "./GT" "GT"
partition_outdoor_hazy_files
mv ./GT/* ./ && rmdir ./GT
mv ./hazy/* ./ && rmdir ./hazy
