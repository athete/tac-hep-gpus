# **TAC-HEP Module: Introduction to GPU Programming, Fall 2024**

This directory contains the code written for the final project. The project is divided into three parts: first,
a two-dimensional stencil operation and a matrix multiplication are written in C++, run, and profiled on a CPU; next, the kernels are rewritten in CUDA and iteratively profiled and optimized using the NSight software packages; and finally, a hardware-agnostic implementation is written using the Alpaka portability library. 

Across all three implementations, we use arrays of size 518x518 as inputs, and a stencil radius of 3. It as also assumed that these applications are compliled and run on GPU-enabled nodes on the physics machines at UW-Madison.  

## Setting up the environment

Standalone C++ applications can be compiled and run with 
```bash
g++ -std=c++17 -o <object> main.cpp
./<object>
```

To run the GPU application, you must first setup a CUDA environment by running
```bash
export LD_LIBRARY_PATH=/usr/local/cuda/lib
export PATH=$PATH:/usr/local/cuda/bin
```
CUDA applications can be compiled and run by executing
```bash
nvcc -o <object> <file>.cu 
./<object>
```

To run the Alpaka application, one muse first setup the working environment by running
```bash
source scl_source enable devtoolset-11
export BOOST_BASE=~abocci/public/boost
export ALPAKA_BASE=$(pwd)
```
To compile and run with the CPU as the device
```bash
g++ -DALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED \
-std=c++17 -O2 -g -I$BOOST_BASE/include -I$ALPAKA_BASE/include \
main.cc \
-o <CPU_executable>
./<CPU_executable>
```
To compile and run with the GPU as the device
```bash
nvcc -x cu –expt-relaxed-constexpr -DALPAKA_ACC_GPU_CUDA_ENABLED \
-std=c++17 -O2 -g -I$BOOST_BASE/include -I$ALPAKA_BASE/include \
main.cc \
-o <GPU_executable>
./<GPU_executable>
```

Outputs from profiling tools (VTune and NSight System) are stored in `cpp/vtune` and `cuda/nsys`. 

## C++ and CPU Profiling
## Porting to CUDA
## Optimizing performance in CUDA
## Making use of Alpaka