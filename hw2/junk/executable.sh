#!/bin/bash
#SBATCH --account=p31490
#SBATCH --partition gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 24:00:00
#SBATCH --mem=64G
#SBTACH --output=/home/wzi1638/home/wzi1638/simulation/job.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=weijia.zhao@kellogg.northwestern.edu

module purge all
module load python/anaconda3.6
source activate my_env
cd /home/wzi1638/home/wzi1638/simulation
python 2_pong.py
