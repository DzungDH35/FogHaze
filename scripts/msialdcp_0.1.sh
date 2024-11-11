#!/bin/bash

algo='msialdcp'
defoghaze_path='defoghaze.py'

# O-HAZE
echo 'Start executing on O-HAZE...'
base='datasets/O-HAZE'
python $defoghaze_path $base/'hazy_1' -gp=$base/'GT_1' -op=$base/'results/1' -dm=0 -ps=30 -fw=0.1 -arf=0.05 -pp=1 <<< $algo
python $defoghaze_path $base/'hazy_2' -gp=$base/'GT_2' -op=$base/'results/2' -dm=0 -ps=30 -fw=0.1 -arf=0.05 -pp=1 <<< $algo
python $defoghaze_path $base/'hazy_3' -gp=$base/'GT_3' -op=$base/'results/3' -dm=0 -ps=30 -fw=0.1 -arf=0.05 -pp=1 <<< $algo
echo 'Done on O-HAZE'

# DENSE-HAZE
echo 'Start executing on DENSE-HAZE...'
base='datasets/DENSE-HAZE'
python $defoghaze_path $base/'hazy_1' -gp=$base/'GT_1' -op=$base/'results/1' -dm=0 -ps=30 -fw=0.1 -arf=0.1 -pp=1 <<< $algo
python $defoghaze_path $base/'hazy_2' -gp=$base/'GT_2' -op=$base/'results/2' -dm=0 -ps=30 -fw=0.1 -arf=0.1 -pp=1 <<< $algo
wait
echo 'Done on DENSE-HAZE'

# NH-HAZE
echo 'Start executing on NH-HAZE...'
base='datasets/NH-HAZE'
python $defoghaze_path $base/'hazy_1' -gp=$base/'GT_1' -op=$base/'results/1' -dm=0 -ps=30 -fw=0.1 -arf=0.1 -pp=1 <<< $algo
python $defoghaze_path $base/'hazy_2' -gp=$base/'GT_2' -op=$base/'results/2' -dm=0 -ps=30 -fw=0.1 -arf=0.1 -pp=1 <<< $algo
echo 'Done on NH-HAZE'
