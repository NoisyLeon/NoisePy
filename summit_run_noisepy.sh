#!/bin/bash
#SBATCH -J Nopy
#SBATCH -o Nopy_%j.out
#SBATCH -e Nopy_%j.err
#SBATCH -N 1
#SBATCH --ntasks-per-node=12
#SBATCH --time=168:00:00
#SBATCH --mem=MaxMemPerNode

dir=/projects/life9360/code/NoisePy
cd $dir
python ref_Alaska.py
