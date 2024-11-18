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

# rename_files_in_ascending_order() {
#     current_dir=$(pwd)
#     target_dir="${1:-$(pwd)}"
    
#     cd "$target_dir" || return 1

#     counter=1
#     for file in *; do
#         mv "$file" "$counter.${file##*.}"
#         ((counter++))
#     done

#     cd "$current_dir" || return 1
# }

partition_indoor_hazy_files() {
    cd "./hazy" || return 1

    for file in *; do
        filename=$(basename "$file")
        filename="${filename%.*}"
        suffix=${filename#*_}
        mkdir -p "degree_$suffix"
        mv "$file" "degree_$suffix/"
    done

    dir_index=1
    for ((i=0; i<num_file_pairs; ++i)); do
        mv "${gt_files[$i]}" "./GT_$dir_index/$((i+1)).${gt_files[$i]##*.}"
        mv "${hazy_files[$i]}" "./hazy_$dir_index/$((i+1)).${hazy_files[$i]##*.}"

        (( (i+1) % FILE_BATCH_SIZE == 0 )) && ((dir_index++))
    done

    cd ".." || return 1
}

partition_outdoor_hazy_files() {
    cd "./hazy" || return 1

    for file in *; do
        new_name=$(echo "$file" | cut -d'_' -f1).${file##*.}

        if [ -e "$new_name" ]; then
            rm "$file"
        else
            mv "$file" "$new_name"
        fi
    done

    cd ".." || return 1
}

# Preprocess indoor dataset
cd ./indoor || exit
mv "./gt" "./GT"
partition_indoor_hazy_files

# Preprocess outdoor dataset
# cd ./outdoor || exit
# mv "./gt" "./GT"
# rename_files_in_ascending_order GT
# partition_outdoor_hazy_files
