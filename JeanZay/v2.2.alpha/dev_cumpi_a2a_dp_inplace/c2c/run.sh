#!/usr/bin/env bash

cd 1
sbatch 0128.slurm
cd ../

for i in 2 4 8
do
	cd $i
	sbatch 0256a.slurm
	cd ../
done

for i in 16 32 64
do
	cd $i
	sbatch 1024a.slurm
	cd ../
done

cd 128
sbatch 2048a.slurm
cd ../
