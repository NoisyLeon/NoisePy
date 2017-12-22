#!/bin/bash

#SBATCH -J DMT
#SBATCH -o DMT_%j.out
#SBATCH -e DMT_%j.err
#SBATCH -N 1
#SBATCH --ntasks-per-node=24
#SBATCH --time=168:00:00
#SBATCH --mem=MaxMemPerNode
#SBATCH --mail-user=lili.feng@colorado.edu
#SBATCH --mail-type=ALL


obspyDMT --datapath /scratch/summit/life9360/ALASKA_work/p_wave_19910101_20170831 --bulk --req_parallel --req_np=24 --parallel_process --process_np=24 --min_date 1991-01-01 --max_date 2017-08-31 --min_mag 5.5 --read_catalog /scratch/summit/life9360/ALASKA_work/quakeml/alaska_2017_aug.ml --data_source IRIS  --station_rect -172./-122./52./72.5 --loc "*" --cha "BHZ,BHE,BHN" --cut_time_phase --preset 30 --offset 60 --pre_filt '(0.04, 0.05, 20., 25.)' --instrument_correction --sampling_rate=40.

