#!/bin/bash
#SBATCH --job-name=c2c_x_2048_b     # Arbitrary name of the Slurm job
#SBATCH --output=%x.%j.out          # Standard output file of the job
#SBATCH --error=%x.%j.err           # Standard error file of the job
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=1
#SBATCH --constraint=h100
#SBATCH --qos=qos_gpu_h100-t3
#SBATCH --account=dhu@h100
#SBATCH --exclusive
# /!\ Important:  The following line is misleading but in Slurm vocabulary,
# "multithread" refers to hyperthreading.
#SBATCH --hint=nomultithread        # 1 MPI process per physical core (no hyperthreading)

# Loading the modules of your choice
module load arch/h100 nvidia-compilers/26.3 cuda/12.8.0 openmpi/4.1.8-cuda

UCX_PROTO_ENABLE=n srun ./fft_c2c_x 32 2 2048 2048 2048 100

sbatch 2048c.slurm
