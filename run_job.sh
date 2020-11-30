#!/bin/bash
#SBATCH -N 1	# Total num of nodes (1 for serial)
#SBATCH -n 1	# Total num of mpi tasks (1 for serial)
#SBATCH --ntasks-per-node 1
#SBATCH -t 00:01:00
#SBATCH -p gtx
#SBATCH -J csnet_gpt2_sbatch	# Job Name
#SBATCH -o csnet_gpt2.out	# Stdout file
#SBATCH -e csnet_gpt2.err	# stderr file
#SBATCH --mail-user=fotios.lygerakis@mavs.uta.edu
#SBATCH --mail-type=all	#Send email at begin and end of the job

script/singularity/console

python -c "import tensorflow as tf"

nvidia-smi

python src/train.py --model gpt2 --quiet ../resources/saved_models/ ../resources/data/python/final/jsonl/train ../resources/data/python/final/jsonl/valid ../resources/data/python/final/jsonl/test

