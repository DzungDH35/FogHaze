#!/bin/bash
readonly PROJECT_DIR="/home/dzungdh/FogHaze"
readonly DEFOGHAZE_PATH="${PROJECT_DIR}/defoghaze.py"
readonly ALGO="msialdcp"

cd "$PROJECT_DIR" || exit

defoghaze() {
    if [ -z "$1" ] || [ -z "$2" ] || [ -z "$3" ]; then
        echo "Error: base_dir, fusion_weight and arf are required."
        exit 1
    fi
    local base_dir=$1
    local fusion_weight=$2
    local arf=$3
    echo "Base directory: $base_dir"
    echo "Fusion Weight: $fusion_weight"
    echo "AtmLight Resize Factor: $arf"

    cd "$base_dir" || exit

    num_GT_dirs=$(find . -type d -name "*GT*" | wc -l)

    local hazy_dir="./hazy"
    local gt_dir="./GT"
    local result_dir="./results_fw_${fusion_weight}"

    if [ "$(basename "$base_dir")" == "indoor" ]; then
        for degree_dir in "$hazy_dir"/degree_*; do
            for ((i = 1; i <= num_GT_dirs; i++)); do
                hazy_dir="${degree_dir}/hazy_$i"
                result_dir="./results_$(basename "$degree_dir")_fw_${fusion_weight}" && mkdir -p "$result_dir"

                python3 "$DEFOGHAZE_PATH" "$hazy_dir" -gp="${gt_dir}_$i" -op="$result_dir" -dm=0 -ps=30 -fw="$fusion_weight" -arf="$arf" -pp=1 <<< "$ALGO"
            done
        done
    else
        mkdir -p "$result_dir"
        if [ "$num_GT_dirs" -eq 1 ]; then
            python3 "$DEFOGHAZE_PATH" "$hazy_dir" -gp="$gt_dir" -op="$result_dir" -dm=0 -ps=30 -fw="$fusion_weight" -arf="$arf" -pp=1 <<< "$ALGO"
        else
            for ((i = 1; i <= num_GT_dirs; i++)); do
                python3 "$DEFOGHAZE_PATH" "${hazy_dir}_$i" -gp="${gt_dir}_$i" -op="$result_dir" -dm=0 -ps=30 -fw="$fusion_weight" -arf="$arf" -pp=1 <<< "$ALGO"
            done
        fi
    fi

    cd "$PROJECT_DIR" || exit
}

fusion_weights=(0.1 0.5 0.9)

# O-HAZE
base_dir="./datasets/O-HAZE"
arf=0.05
echo -e "\033[0;32mStart executing on O-HAZE...\033[0m" && echo
for fw in "${fusion_weights[@]}"; do
    defoghaze "$base_dir" "$fw" "$arf"
done
echo && echo -e "\033[0;32mDone on O-HAZE\033[0m" && echo -e "\n"

# Dense-HAZE
base_dir="./datasets/Dense_Haze"
arf=0.1
echo -e "\033[0;32mStart executing on Dense_Haze...\033[0m" && echo
for fw in "${fusion_weights[@]}"; do
    defoghaze "$base_dir" "$fw" "$arf"
done
echo && echo -e "\033[0;32mDone on Dense_Haze\033[0m" && echo -e "\n"

# NH-HAZE
base_dir="./datasets/NH-HAZE"
arf=0.1
echo -e "\033[0;32mStart executing on NH-HAZE...\033[0m" && echo
for fw in "${fusion_weights[@]}"; do
    defoghaze "$base_dir" "$fw" "$arf"
done
echo && echo -e "\033[0;32mDone on NH-HAZE\033[0m" && echo -e "\n"

# HSTS synthetic
base_dir="./datasets/HSTS/synthetic"
arf=0.2
echo -e "\033[0;32mStart executing on HSTS synthetic...\033[0m" && echo
for fw in "${fusion_weights[@]}"; do
    defoghaze "$base_dir" "$fw" "$arf"
done
echo && echo -e "\033[0;32mDone on HSTS synthetic\033[0m"

# SOTS indoor
base_dir="./datasets/SOTS/indoor"
arf=0.3
echo -e "\033[0;32mStart executing on SOTS indoor...\033[0m" && echo
for fw in "${fusion_weights[@]}"; do
    defoghaze "$base_dir" "$fw" "$arf"
done
echo && echo -e "\033[0;32mDone on SOTS indoor\033[0m"

# SOTS outdoor
base_dir="./datasets/SOTS/outdoor"
arf=0.2
echo -e "\033[0;32mStart executing on SOTS outdoor...\033[0m" && echo
for fw in "${fusion_weights[@]}"; do
    defoghaze "$base_dir" "$fw" "$arf"
done
echo && echo -e "\033[0;32mDone on SOTS outdoor\033[0m"
