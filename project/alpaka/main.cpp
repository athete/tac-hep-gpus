#include <iostream>
#include <alpaka/alpaka.hpp>
#include "config.h"
#include "workdivision.h"

const int DSIZE = 518;
const int RADIUS = 3;
const int BLOCK_SIZE = 32;
const int A_val = 1;
const int B_val = 2;

struct 2DStencilKernel
{
    template <typename TAcc, typename T>
    ALPAKA_FN_ACC void operator()(TAcc const& acc, T const* __restrict__ in, T* __restrict__ out, Vec2D size) const {
        for (auto ndindex : elements_with_stride_nd(acc, size))
        {
            if ((ndindex[0] >= RADIUS && ndindex[0] < size[0]-RADIUS) && (ndindex[1] >= RADIUS && ndindex[1] < size[1]-RADIUS))
            {
                int result = in[ndindex[0]*size[1] + ndindex[1]];
                for (int r = 1; r <= RADIUS; r++)
                {
                    result += in[(ndindex[0]+r)*size[1]+ndindex[1]];
                    result += in[(ndindex[0]-r)*size[1]+ndindex[1]];
                    result += in[ndindex[0]*size[1]+ndindex[1]+r];
                    result += in[ndindex[0]*size[1]+ndindex[1]-r];
                }
                out[ndindex[0]*size[1]+ndindex[1]] = result;
            }
        }
    }
};

struct MatMulKernel
{
    template <typename TAcc, typename T>
    ALPAKA_FN_ACC void operator()(TAcc const& acc, T const* __restrict__ A, T const* __restrict__ B, T* __restrict__ out, Vec2D size) const {
        for (auto ndindex : elements_with_stride_nd(acc, size))
        {
            if(ndindex[0] < DSIZE && ndindex[1] < DSIZE)
            {
                int sum = 0;
                for (int i = 0; i < size[1]; i++)
                {
                    auto A_idx = ndindex[0] * size[1] + i;
                    auto B_idx = i * size[1] + nindex[1];
                    sum += A[A_idx] * B[B_idx];
                }
                auto out_idx = ndindex[0] * size[1] + ndindex[1];
                out[out_idx] = sum;
            }
        }
    }
};

int main(int argc, char const *argv[])
{
    std::size_t num_devices = alpaka::getDevCount<Platform>();
    if(num_devices == 0)
        exit(EXIT_FAILURE);

    Host host = alpaka:getDevByIdx<HostPlatform>(0u);
    std::cout << "Host: " << alpaka::getname(host) << std::endl;

    Device device = alpaka::getDevByIdx<Platform>(0u);
    std::cout << "Host: " << alpaka::getname(device) << std::endl;

    constexpr Vec2D m_ndsize = {DSIZE, DSIZE};
    constexpr size_t m_size = m_ndsize.prod();

    auto A_in_h = alpaka::allocMappedBuf<Platform, int, uint32_t>(host, Vec1D{m_size});
    auto A_out_h = alpaka::allocMappedBuf<Platform, int, uint32_t>(host, Vec1D{m_size});
    auto B_in_h = alpaka::allocMappedBuf<Platform, int, uint32_t>(host, Vec1D{m_size});
    auto B_out_h = alpaka::allocMappedBuf<Platform, int, uint32_t>(host, Vec1D{m_size});
    auto C_h = alpaka::allocMappedBuf<Platform, int, uint32_t>(host, Vec1D{m_size});

    for (auto i = 0; i < m_size; i++)
    {
        A_in_h[i] = A_val;
        A_out_h[i] = A_val;
        B_in_h[i] = B_val;
        B_out_h[i] = B_val;
        C_h[i] = 0;
    }

    auto queue = Queue{device};
    auto A_in_d = alpaka::allocAsyncBuf<int, uint32_t>(queue, Vec1D{m_size});
    auto A_out_d = alpaka::allocAsyncBuf<int, uint32_t>(queue, Vec1D{m_size});
    auto B_in_d = alpaka::allocAsyncBuf<int, uint32_t>(queue, Vec1D{m_size});
    auto B_out_d = alpaka::allocAsyncBuf<int, uint32_t>(queue, Vec1D{m_size});
    auto C_d = alpaka::allocAsyncBuf<int, uint32_t>(queue, Vec1D{m_size});

    alpaka::memcpy(queue, A_in_d, A_in_h);
    alpaka::memcpy(queue, A_out_d, A_out_h);
    alpaka::memcpy(queue, B_in_d, B_in_h);
    alpaka::memcpy(queue, B_out_d, B_out_h);

    alpaka::memset(queue, C_d, 0x00);

    int m_gridsize = (DSIZE + BLOCK_SIZE-1)/BLOCK_SIZE;
    auto m_div = make_workdiv<Acc2D>({m_gridsize, m_gridsize}, {BLOCK_SIZE, BLOCK_SIZE})
    alpaka::exec<Acc2D>(queue, m_div, StencilKernel{}, A_in_d.data(), A_out_d.data(), m_ndsize);
    alpaka::exec<Acc2D>(queue, m_div, StencilKernel{}, B_in_d.data(), B_out_d.data(), m_ndsize);
    alpaka::exec<Acc2D>(queue, m_div, MatMulKernel{}, A_out_d.data(), B_out_d.data(), C_d.data(), m_ndsize);

    alpaka::memcpy(queue, C_h, C_d);
    alpaka::wait(queue);

    int exp_edge = A_val*B_val*((RADIUS*4+1)*(DSIZE-2*RADIUS)+2*RADIUS);
    int exp_center = A_val*B_val*((RADIUS*4+1)*(RADIUS*4+1)*(DSIZE-2*RADIUS)+2*RADIUS);
    for (int i = 0; i < DSIZE; ++i) {
        for (int j = 0; j < DSIZE; ++j) {
            if ((i < RADIUS || i >= DSIZE-RADIUS) && (j < RADIUS || j >= DSIZE-RADIUS)) {
                if (C_h[j+i*DSIZE] != A_val*B_val*DSIZE) {
                    printf("Mismatch at index [%i,%i], was: %i, should be: %i\n", i,j, C_h[j+i*DSIZE], A_val*B_val*DSIZE);
                    return -1;
                }
            }
            else if ((j < RADIUS || j >= DSIZE-RADIUS) && (i >= RADIUS && i< DSIZE-RADIUS)){
                if (C_h[j+i*DSIZE] != exp_edge) {
                    printf("Mismatch at index [%i,%i], was: %d, should be: %i\n", i,j, C_h[j+i*DSIZE], exp_edge);
                    return -1;
                }
            }
            else if ((i < RADIUS || i >= DSIZE-RADIUS) && (j >= RADIUS && j< DSIZE-RADIUS)){
                if (C_h[j+i*DSIZE] != exp_edge) {
                    printf("Mismatch at index [%i,%i], was: %i, should be: %i\n", i,j, C_h[j+i*DSIZE], exp_edge);
                    return -1;
                }
            }
            else {
                if (C_h[j+i*DSIZE] != exp_center) {
                    printf("Mismatch at index [%i,%i], was: %i, should be: %i\n", i,j, C_h[j+i*DSIZE], exp_center);
                    return -1;
                }
            }
        }
    }
    std::cout << "Success!" << std::endl;


    return 0;
}
