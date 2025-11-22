#include "CudaORBextractor.h"
#include <cuda_runtime.h>
#include <iostream>
#include <stdio.h>

namespace ORB_SLAM3 
{
__global__ void helloCUDA()
{
    printf("Salut din Thread-ul GPU %d!\n", threadIdx.x);
}

CudaORBextractor::CudaORBextractor()
{
    std::cout << "CudaORBextractor initializat." << std::endl;
}

void CudaORBextractor::WarmupGPU()
{
    std::cout << "[CUDA] Lansam kernel-ul de test..." << std::endl;
    helloCUDA<<<1, 5>>>();
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        std::cerr << "[CUDA ERROR] " << cudaGetErrorString(err) << std::endl;
    }
    else
    {
        std::cout << "[CUDA] Kernel executat cu succes." << std::endl;
    }
}

} // namespace ORB_SLAM3