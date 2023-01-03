#!/bin/bash

#SBATCH -p short
#SBATCH -t 2:00:00
#SBATCH --mem=32G
#SBATCH -c 1
#SBATCH -n 1
#SBATCH --job-name=AffineStitch
#SBATCH --output=Name_%J.out
#SBATCH --mail-type ALL
#SBATCH --mail-user jack.featherstone@oist.jp

source "/home/j/john-featherstone/.bashrc"
export PYTHONPATH="/home/j/john-featherstone/apps/lib/python3.9/site-packages/"
conda activate ml

python3 stitch.py "/bucket/DaniU/Members/Jack Featherstone/Test/map_2_images/" -o 2022-11-25_B_map_sift.png -d "sift"
python3 stitch.py "/bucket/DaniU/Members/Jack Featherstone/Test/map_2_images/" -o 2022-11-25_B_map_fast.png -d "fast"
python3 stitch.py "/bucket/DaniU/Members/Jack Featherstone/Test/map_2_images/" -o 2022-11-25_B_map_orb.png -d "orb"
