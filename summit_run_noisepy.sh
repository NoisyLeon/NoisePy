#!/bin/bash
#SBATCH -J Nopy
#SBATCH -o Nopy_%j.out
#SBATCH -e Nopy_%j.err
#SBATCH -N 1
#SBATCH --ntasks-per-node=24
#SBATCH --time=1:00:00
#SBATCH --mem=MaxMemPerNode

#unset DISPLAY XAUTHORITY
dir=/projects/life9360/code/NoisePy
cd $dir
python quake_Alaska.py
