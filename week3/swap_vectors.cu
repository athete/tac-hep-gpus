#include <cstdio>
#include <iostream>


const int DSIZE = 40960;
const int block_size = 256;
const int grid_size = DSIZE/block_size;

__device__ void swap(float &a, float &b)
{
    a = a * b;
    b = a / b;
    a = a / b;
}


__global__ void vector_swap(float* A, float* B) 
{

    // Express the vector index in terms of threads and blocks
    int idx =  blockDim.x * blockIdx.x + threadIdx.x;
    // Swap the vector elements - make sure you are not out of range
    if (idx < DSIZE)
        swap(A[idx], B[idx]);
}


int main() {


    float *h_A, *h_B, *d_A, *d_B;
    h_A = new float[DSIZE];
    h_B = new float[DSIZE];


    for (int i = 0; i < DSIZE; i++) 
    {
        h_A[i] = rand()/(float)RAND_MAX;
        h_B[i] = rand()/(float)RAND_MAX;
    }

    // Print elements before swap
    std::cout << "Before swap\nA : ";
    for (int i = 0; i < 5; i++)
        std::cout << h_A[i] << " ";
    std::cout << "..." << std::endl;

    std::cout << "B : ";
    for (int i = 0; i < 5; i++)
        std::cout << h_B[i] << " ";
    std::cout << "..." << std::endl;


    // Allocate memory for device pointers 
    cudaMalloc((void**) &d_A, DSIZE * sizeof(float));
    cudaMalloc((void**) &d_B, DSIZE * sizeof(float));

    // Copy from host to device
    cudaMemcpy(d_A, h_A, DSIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, DSIZE * sizeof(float), cudaMemcpyHostToDevice);

    // Launch the kernel
    vector_swap<<<grid_size, block_size>>>(d_A, d_B);

    // Copy back to host 
    cudaMemcpy(h_A, d_A, DSIZE * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_B, d_B, DSIZE * sizeof(float), cudaMemcpyDeviceToHost);

    // Print and check some elements to make sure swapping was successfull

    std::cout << "After swap\nA : ";
    for (int i = 0; i < 5; i++)
        std::cout << h_A[i] << " ";
    std::cout << "..." << std::endl;

    std::cout << "B : ";
    for (int i = 0; i < 5; i++)
        std::cout << h_B[i] << " ";
    std::cout << "..." << std::endl;

    // Free the memory 
    free(h_A);
    free(h_B);
    cudaFree(d_A);
    cudaFree(d_B);

    return 0;
}
