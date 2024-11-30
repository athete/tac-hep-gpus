#include <iostream>
#include <cassert>

const int DSIZE = 518;
const int RADIUS = 3;
const int A_val = 1;
const int B_val = 2;

void matrix_multiply(int A[][DSIZE], int B[][DSIZE], int C[][DSIZE])
{
    for (int i = 0; i < DSIZE; i++)
    {
        for (int j = 0; j < DSIZE; j++)
        {
            for (int k = 0; k < DSIZE; k++)
                C[i][j] += A[i][k] * B[k][j];
        }
    }
}

void stencil_2d(int in[][DSIZE], int out[][DSIZE])
{
    for (int i = RADIUS; i < (DSIZE - RADIUS); i++)
    {
        for (int j = RADIUS; j < (DSIZE - RADIUS); j++)
        {
            int sum = in[i][j];
            for (int r = 1; r < (RADIUS + 1); r++)
            {
                sum += in[i - r][j];
                sum += in[i + r][j];
                sum += in[i][j - r];
                sum += in[i][j + r];
            }
            out[i][j] = sum;   
        }
    }
}

void test_main(int out[][DSIZE])
{
    auto r1r1 = B_val * A_val * (RADIUS * 4 + 1) * (RADIUS * 4 + 1) * (DSIZE - 2 * RADIUS);
    r1r1 += 2 * RADIUS * A_val * B_val;
    assert(r1r1 == out[RADIUS+1][RADIUS + 1]);
    std::cout << "All tests passed!" << std::endl;
}

int main(int argc, char const *argv[])
{
    int A[DSIZE][DSIZE], B[DSIZE][DSIZE];
    int stencil_A[DSIZE][DSIZE], stencil_B[DSIZE][DSIZE];
    int out[DSIZE][DSIZE];
    for (int i = 0; i < DSIZE; i++)
    {
        for (int j = 0; j < DSIZE; j++)
        {
            A[i][j] = A_val;
            B[i][j] = B_val;
            stencil_A[i][j] = A_val;
            stencil_B[i][j] = B_val;
            out[i][j] = 0;
        }
    }

    stencil_2d(A, stencil_A);
    stencil_2d(B, stencil_B);
    matrix_multiply(stencil_A, stencil_B, out);
    test_main(out);

    return 0;
}
