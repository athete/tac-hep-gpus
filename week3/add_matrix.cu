#include <stdio.h>
#include <iostream>

const int DSIZE_X = 256;
const int DSIZE_Y = 256;

__global__ void add_matrix(float* A, float* B, float* C)
{
    // Express in terms of threads and blocks
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    // Add the two matrices - make sure you are not out of range
    int element = DSIZE_Y * idx + idy;
    if (idx <  DSIZE_X && idy < DSIZE_Y)
        C[element] = A[element] + B[element];

}

int main()
{
    // Create and allocate memory for host and device pointers 
    float **h_A, **h_B, **h_C, *d_A, *d_B, *d_C;
    size_t size = DSIZE_X * DSIZE_Y * sizeof(float);

    // Tiny nuance when allocating 2D arrays to copy over to the device:
    // http://www.trevorsimonton.com/blog/2016/11/16/transfer-2d-array-memory-to-cuda.html
    h_A = new float*[DSIZE_X];
    h_A[0] = new float[DSIZE_X * DSIZE_Y];
    h_B = new float*[DSIZE_X];
    h_B[0] = new float[DSIZE_X * DSIZE_Y];
    h_C = new float*[DSIZE_X];
    h_C[0] = new float[DSIZE_X * DSIZE_Y];
    for (int i = 0; i < DSIZE_X; i++)
    {
        h_A[i] = h_A[i - 1] + DSIZE_Y;
        h_B[i] = h_B[i - 1] + DSIZE_Y;
        h_C[i] = h_C[i - 1] + DSIZE_Y;
    }
    

    for (int i = 0; i < DSIZE_X; i++)
    {
        h_A[i] = new float[DSIZE_Y];
        h_B[i] = new float[DSIZE_Y];
        h_C[i] = new float[DSIZE_Y];
    }

    // Fill in the matrices
    for (int i = 0; i < DSIZE_X; i++) 
    {
        for (int j = 0; j < DSIZE_Y; j++) 
        {
            h_A[i][j] = (float) rand()/RAND_MAX;
            h_B[i][j] = (float) rand()/RAND_MAX;
            h_C[i][j] = 0;
        }
    }

    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    // Copy from host to device
    cudaMemcpy(d_A, h_A[0], size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B[0], size, cudaMemcpyHostToDevice);

    // Launch the kernel
    // dim3 is a built in CUDA type that allows you to define the block 
    // size and grid size in more than 1 dimentions
    // Syntax : dim3(Nx,Ny,Nz)
    dim3 blockSize(10, 10); 
    dim3 gridSize((DSIZE_X + blockSize.x-1)/blockSize.x, (DSIZE_Y + blockSize.y-1)/blockSize.y); 
    
    add_matrix<<<gridSize, blockSize>>>(d_A, d_B, d_C);
    cudaDeviceSynchronize();

    // Copy back to host 
    cudaMemcpy(h_C[0], d_C, size, cudaMemcpyDeviceToHost);
    // Print and check some elements to make the addition was succesfull
    for(int i = 0; i < 3; i++)
    {
        for(int j = 0; j < 3; j++)
            std::cout << h_A[i][j] << " ";
        std::cout << std::endl;
    }
    std::cout << " + " << std::endl;
    for(int i = 0; i < 3; i++)
    {
        for(int j = 0; j < 3; j++)
            std::cout << h_B[i][j] << " ";
        std::cout << std::endl;
    }
    std::cout << " = " << std::endl;
    for(int i = 0; i < 3; i++)
    {
        for(int j = 0; j < 3; j++)
            std::cout << h_C[i][j] << " ";
        std::cout << std::endl;
    }

    // Free the memory    
    cudaFree(d_A); 
    cudaFree(d_B); 
    cudaFree(d_C); 

    delete[] h_A[0];
    delete[] h_A;
    delete[] h_B[0];
    delete[] h_B;
    delete[] h_C[0];
    delete[] h_C;

    return 0;
}