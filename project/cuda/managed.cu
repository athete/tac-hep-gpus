#include <stdio.h>
#include <algorithm>
#include <iostream>

using namespace std;

const int N = 512;
const int RADIUS = 3;
const int BLOCK_SIZE = 32;
const int A_val = 1;
const int B_val = 2;


__global__ void stencil_2d(int *in, int *out) 
{
	int gindex_x = threadIdx.x + blockIdx.x * blockDim.x;
	int gindex_y = threadIdx.y + blockIdx.y * blockDim.y;

	// Read input elements into shared memory
	int size = N + 2 * RADIUS;

	// Apply the stencil
    int result = 0;
    for (int offset = -RADIUS; offset <= RADIUS; offset++){
        result += in[gindex_y+(gindex_x+offset)*size];
        result += in[gindex_y+offset+gindex_x*size];
    }
    // Avoid double-counting the center
    result -= in[gindex_y+gindex_x*size];

	// Store the result
	out[gindex_y+size*gindex_x] = result;
}

__global__ void matrix_mul_gpu(const int *A, const int *B, int *C, int size) 
{
    // create thread x index
    // create thread y index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    // Make sure we are not out of range
    if ((idx < size) && (idy < size)) 
    {
        int temp = 0;
        for (int i = 0; i < size; i++)
            temp += A[i + idy * size] * B[idx + i * size];
        C[idy*size+idx] = temp;                    
    }

}

void fill_ints(int *x, int n, int val) 
{
   // Store the result
   // https://en.cppreference.com/w/cpp/algorithm/fill_n
   fill_n(x, n, val);
}

int main(void) 
{

	int *in_A, *in_B, *out_A, *out_B, *C; // host copies of a, b, c

	// Alloc space for host copies and setup values
	int size = (N + 2*RADIUS)*(N + 2*RADIUS) * sizeof(int);
    int DSIZE = N + 2 * RADIUS;

	cudaMallocManaged((void **)&in_A, size);
    cudaMallocManaged((void **)&in_B, size);
	cudaMallocManaged((void **)&out_A, size);
    cudaMallocManaged((void **)&out_B, size);
    cudaMallocManaged((void **)&C, size);

    // Initialize arrays
    fill_ints(in_A, (N + 2*RADIUS)*(N + 2*RADIUS), A_val);
    fill_ints(in_B, (N + 2*RADIUS)*(N + 2*RADIUS), B_val);
    fill_ints(out_A, (N + 2*RADIUS)*(N + 2*RADIUS), A_val);
    fill_ints(out_B, (N + 2*RADIUS)*(N + 2*RADIUS), B_val);
    fill_ints(C, (N + 2*RADIUS)*(N + 2*RADIUS), 0);


	// Launch stencil_2d() kernel on GPU
	int gridSize = (N + BLOCK_SIZE-1)/BLOCK_SIZE;
	dim3 grid(gridSize, gridSize);
	dim3 block(BLOCK_SIZE, BLOCK_SIZE);
	// Launch the kernel 
	// Properly set memory address for first element on which the stencil will be applied
	stencil_2d<<<grid,block>>>(in_A + RADIUS*(N + 2*RADIUS) + RADIUS , out_A + RADIUS*(N + 2*RADIUS) + RADIUS);
    stencil_2d<<<grid,block>>>(in_B + RADIUS*(N + 2*RADIUS) + RADIUS , out_B + RADIUS*(N + 2*RADIUS) + RADIUS);

    // Launch matrix multiplication kernel on GPU
    int multGridSize = (DSIZE + BLOCK_SIZE-1)/BLOCK_SIZE;
	dim3 multGrid(multGridSize, multGridSize);
	dim3 multBlock(BLOCK_SIZE, BLOCK_SIZE);
    // Launch the kernel 
    matrix_mul_gpu<<<multGrid, multBlock>>>(out_A, out_B, C, DSIZE);

    cudaDeviceSynchronize();

	// Error Checking
	int exp_edge = A_val*B_val*((RADIUS*4+1)*(DSIZE-2*RADIUS)+2*RADIUS);
    int exp_center = A_val*B_val*((RADIUS*4+1)*(RADIUS*4+1)*(DSIZE-2*RADIUS)+2*RADIUS);
    for (int i = 0; i < N + 2 * RADIUS; ++i) 
    {
        for (int j = 0; j < N + 2 * RADIUS; ++j) 
        {

            if ((i < RADIUS || i >= N + RADIUS) && (j < RADIUS || j >= N+RADIUS)) 
            {
                if (C[j+i*(N + 2 * RADIUS)] != A_val*B_val*DSIZE) 
                {
                    printf("Mismatch at index [%d,%d], was: %d, should be: %d\n", i,j, C[j+i*(N + 2 * RADIUS)], A_val*B_val*DSIZE);
                    return -1;
                }
            }
            else if ((j < RADIUS || j >= N + RADIUS) && (i >= RADIUS && i< N+RADIUS))
            {
                if (C[j+i*(N + 2 * RADIUS)] != exp_edge) 
                {
                    printf("Mismatch at index [%d,%d], was: %d, should be: %d\n", i,j, C[j+i*(N + 2 * RADIUS)], exp_edge);
                    return -1;
                }
            }        
            else if ((i < RADIUS || i >= N + RADIUS) && (j >= RADIUS && j< N+RADIUS))
            {
                if (C[j+i*(N + 2 * RADIUS)] != exp_edge) 
                {
                    printf("Mismatch at index [%d,%d], was: %d, should be: %d\n", i,j, C[j+i*(N + 2 * RADIUS)], exp_edge);
                    return -1;
                }
            }
            else {
                if (C[j+i*(N + 2 * RADIUS)] != exp_center) 
                {
                    printf("Mismatch at index [%d,%d], was: %d, should be: %d\n", i,j, C[j+i*(N + 2 * RADIUS)], exp_center);
                    return -1;
                }
            }
        }
    }
	// Cleanup
	cudaFree(in_A);
    cudaFree(in_B);
	cudaFree(out_A);
    cudaFree(out_B);
    cudaFree(C);
	printf("Success!\n");

	return 0;
}