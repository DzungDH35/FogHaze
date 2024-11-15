#!/bin/bash

: '
This file is used to partition files into GT and hazy folder pairs, 
each folder contains up to FILE_BATCH_SIZE files (for example FILE_BATCH_SIZE = 20).

NOTE: Only run it to preprocess the dataset fetch from the original paper/repo
'

readonly DATASET_DIR="./O-HAZE"
readonly FILE_BATCH_SIZE=20

cd "$DATASET_DIR" || exit 1
echo "Start processing $DATASET_DIR..."

hazy_files=()
gt_files=()

for file in *; do
    if [[ -f "$file" ]]; then
        if [[ "$file" == *hazy* ]]; then
            hazy_files+=("$file")
        elif [[ "$file" == *GT* ]]; then
            gt_files+=("$file")
        fi
    fi
done

[ ${#hazy_files[@]} -ne ${#gt_files[@]} ] && { echo "Error: The number of hazy files does not match the number of GT files."; exit 1; }
num_file_pairs=${#hazy_files[@]}

create_dir_pairs() {
    num_dirs=$(( ($num_file_pairs + $FILE_BATCH_SIZE - 1) / $FILE_BATCH_SIZE ))

    for ((i=1; i<=num_dirs; ++i)); do
        mkdir "GT_$i"
        mkdir "hazy_$i"
    done
}

# Partition hazy and GT files into pair of directories
partition_files() {
    dir_index=1

    for ((i=0; i<num_file_pairs; ++i)); do
        mv "${gt_files[$i]}" "./GT_$dir_index/$((i+1)).${gt_files[$i]##*.}"
        mv "${hazy_files[$i]}" "./hazy_$dir_index/$((i+1)).${hazy_files[$i]##*.}"

        (( (i+1) % FILE_BATCH_SIZE == 0 )) && ((dir_index++))
    done
}

create_dir_pairs
partition_files

echo "The dataset is processed succesfully!"
