# 2DECOMP&FFT: Scalability tests performed on Jean Zay

The following tests were performed on the machine Jean Zay using 2decomp&FFT version 2.2 alpha :

- [3D R2C FFT ](https://github.com/2decomp-fft/2decomp-fft/blob/v2.1/examples/fft_physical_x/README.md) starting from a X-pencil decomposition
- [3D C2C FFT ](https://github.com/2decomp-fft/2decomp-fft/blob/v2.1/examples/fft_physical_x/README.md) starting from a X-pencil decomposition
- [3D R2C FFT ](https://github.com/2decomp-fft/2decomp-fft/blob/v2.1/examples/fft_physical_z/README.md) starting from a Z-pencil decomposition

The tests were performed in single precision and double precision using CUDA-aware MPI_ALLTOALL and CUDA-aware MPI_ALLTOALLV.
Each test was performed one time to initialize the memory and then 100 times to measure the average performance.
The partition H100 contains 364 nodes, each with 4 GPU NVIDIA H100 SXM5 80 Go.

## Version of 2DECOMP&FFT used in the tests

The version used for the tests is slightly ahead of 2.1 and is very close to 69c1c60a172507740d81ae9abbf3d4f9ed13dd9c. The following diff was present :

```
diff --git a/cmake/D2D_GPU.cmake b/cmake/D2D_GPU.cmake
index e4b12d8..1c4b6d3 100644
--- a/cmake/D2D_GPU.cmake
+++ b/cmake/D2D_GPU.cmake
@@ -30,6 +30,7 @@ if (ENABLE_CUDA)
     set(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH_COMP} CACHE STRING "Set the correct CUDA architecture" FORCE)
   else()
     set(CUDA_ARCH_COMP ${SET_CUDA_ARCH})
+    set(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH_COMP} CACHE STRING "Set the correct CUDA architecture" FORCE)
   endif()
 endif()
 
diff --git a/cmake/D2D_MPI.cmake b/cmake/D2D_MPI.cmake
index 549fed3..8aacecb 100644
--- a/cmake/D2D_MPI.cmake
+++ b/cmake/D2D_MPI.cmake
@@ -66,9 +66,9 @@ if (MPI_FOUND)
        set(NP ${MPIEXEC_MAX_NUMPROCS})
     endif()
     # For even we'll test with a power of 2 number of MPI RANKS
-    if (EVEN)
-      closest_power_of_2(${NP} NP)
-    endif()
+    #if (EVEN)
+    #  closest_power_of_2(${NP} NP)
+    #endif()
     message(STATUS "NUMBER OF PROCS USED FOR TESTING ${NP}")
     set(MPI_NUMPROCS ${NP} CACHE STRING "SAVE NRANKS FOR MPIRUN" FORCE)
     set(MPI_NUMPROCS_SET 1 CACHE INTERNAL "MPI Ranks set" FORCE)
```

## Building 2DECOMP&FFT:

The build script below provides the version number for the compiler and the libraries.
```
#!/usr/bin/env bash
module purge
module load arch/h100
module load git/2.53.0
module load cmake/3.31.4
module load nvidia-compilers/26.3
module load cuda/12.8.0
module load openmpi/4.1.8-cuda

rm -rf include/ lib64/ tmp/
mkdir tmp
cd tmp

FC=mpif90 CC=mpicc CXX=mpicxx cmake -S ../.. -B build_dp_mpi_h100 -DCMAKE_INSTALL_PREFIX=xxxxxxx -DBUILD_SHARED_LIBS=off -DCMAKE_BUILD_TYPE=release -DBUILD_TESTING=ON -DBUILD_TARGET=gpu -DENABLE_NCCL=no -DSET_CUDA_ARCH=90 -DEVEN=ON -DCOMPLEX_TESTS=OFF -DDOUBLE_PRECISION=OFF -DENABLE_INPLACE=ON -DENABLE_PROFILER=caliper -Dcaliper_DIR=xxxxxxxxxxx
FC=mpif90 CC=mpicc CXX=mpicxx cmake --build build_dp_mpi_h100 --verbose
FC=mpif90 CC=mpicc CXX=mpicxx cmake --install build_dp_mpi_h100
```

## Running the tests

Most of the tests, when completed, submit a job for the next one.
The script `run.sh`, available in each case, can be used to run all the tests associated with a case.

## Processing the tests

The python script `process.py` available in the present folder can process the dataset and analyse the performance.

# Results of the test

## Tests on a single GPU

For grids 128^3, 256^3 and 512^3 show a simple trend : the higher the number of cells, the longer it takes to run the test.

![Complex-to-complex FFT, forward + backward. One GPU. Various grids.](./images/c2c_gpu_1.png)

The table below provides the relative increase of the timer when the number of cells increases by a factor 8.

| Size of the grid  | 128^3 => 256^3 | 256^3 => 512^3 |
| ----------------- | :------------: | :------------: |
| Double precision  | x 2.6          | x 4            |
| Single precision  | x 2.2          | x 3.4          |

![Real-to-complex FFT, physical in X, forward + backward. One GPU. Various grids.](./images/r2c_x_gpu_1.png)

The table below provides the relative increase of the timer when the number of cells increases by a factor 8.

| Size of the grid | 128^3 => 256^3 | 256^3 => 512^3 |
| ----------------- | :------------: | :------------: |
| Double precision  | x 2.3          | x 3.5          |
| Single precision  | x 2.1          | x 2.8          |

![Real-to-complex FFT, physical in Z, forward + backward. One GPU. Various grids.](./images/r2c_z_gpu_1.png)

The table below provides the relative increase of the timer when the number of cells increases by a factor 8.

| Size of the grid  | 128^3 => 256^3 | 256^3 => 512^3 |
| ----------------- | :------------: | :------------: |
| Double precision  | x 2.5          | x 4            |
| Single precision  | x 2.2          | x 3.2          |

## Impact of the pencil decomposition

Tests on 64 GPUs with the grid 1024^3 illustrate the critical impact of the pencil decomposition on the performance.
Pencil decomposition with 4 GPUs in a row or column can perform better because each node has 4 GPUs.
However, this general observation is not always accurate.
Each application using 2DECOMP&FFT should analyse carefully their workflow and adapt their pencil decomposition accordingly.

![Real-to-complex FFT, physical in Z, forward + backward. 64 GPUs. Grid 1024^3. Various pencil decomposition.](./images/r2c_z_gpu_64_nx_1024.png)

## Strong scaling

### Small number of GPUs and small grids

The strong scaling for a grid 512^3 with up to 2 nodes (8 GPUs) is not very good.
The NCCL backend might improve the performance.

![Real-to-complex FFT, physical in Z, forward + backward. Grid 512^3. Various number of GPUs.](./images/best_dp_c2c_nx_512.png)

### Large number of GPUs and large grids

The strong scaling for a larger grid (1024^3) using 2 to 16 nodes is more interesting.

![Real-to-complex FFT, physical in Z, forward + backward. Grid 1024^3. Various number of GPUs.](./images/best_dp_r2c_z_nx_1024.png)

The table below provides the relative decrease of the timer when the number of GPUs increases by a factor 2.

| Numbze of GPUs   | 8 => 16 | 16 => 32 | 32 => 64 |
| ---------------- | :-----: | :------: | :------: |
| Double precision | / 1.16  | / 1.31   | / 1.36   |
| Single precision | / 1.21  | / 1.23   | / 1.22   |
