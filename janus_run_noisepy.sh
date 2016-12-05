#!/bin/bash
#SBATCH -J Nopy
#SBATCH -o Nopy_%j.out
#SBATCH -e Nopy_%j.err
#SBATCH -N 1
#SBATCH --ntasks-per-node=12
#SBATCH --time=12:00:00
#SBATCH --mem=MaxMemPerNode
module load gcc/gcc-4.9.1
module load fftw
module load python/anaconda-2.1.0
#module load fftw/fftw-3.3.3_openmpi-1.7.4_gcc-4.8.2_double_ib
dir=/projects/life9360/code/NoisePy
cd $dir
python noisepy_addslowness.py
