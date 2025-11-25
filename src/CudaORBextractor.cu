#include "CudaORBextractor.h"
#include <iostream>
#include <stdio.h>

namespace ORB_SLAM3
{
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            printf("CUDA Error: %s la linia %d\n", cudaGetErrorString(err), __LINE__); \
            exit(1); \
        } \
    } while (0)


// Kernel 1:
__global__ void ResizeKernel(unsigned char* src, int srcStep, int srcWidth, int srcHeight,
                             unsigned char* dst, int dstStep, int dstWidth, int dstHeight,
                             float scale)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= dstWidth || y >= dstHeight) return;

    // Calculam coordonata in sursa
    int srcX = (int)(x * scale);
    int srcY = (int)(y * scale);

    if (srcX >= srcWidth) srcX = srcWidth - 1;
    if (srcY >= srcHeight) srcY = srcHeight - 1;

    // Copiem pixelul
    dst[y * dstStep + x] = src[srcY * srcStep + srcX];
}

// Kernel 2:
__global__ void GaussianBlurKernel(unsigned char* src, unsigned char* dst, 
                                   int width, int height, int pitch)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < 1 || x >= width - 1 || y < 1 || y >= height - 1) 
        return;

    int centerIdx = y * pitch + x;
    int upIdx     = (y - 1) * pitch + x;
    int downIdx   = (y + 1) * pitch + x;

    unsigned int sum = 0;
    // Masca 3x3:
    // 1 2 1
    // 2 4 2
    // 1 2 1
    sum += src[upIdx - 1] * 1 + src[upIdx] * 2 + src[upIdx + 1] * 1;
    sum += src[centerIdx - 1] * 2 + src[centerIdx] * 4 + src[centerIdx + 1] * 2;
    sum += src[downIdx - 1] * 1 + src[downIdx] * 2 + src[downIdx + 1] * 1;

    dst[centerIdx] = (unsigned char)(sum / 16);
}

// Constructor  
CudaORBextractor::CudaORBextractor(int nLevels, float fScaleFactor)
    : mnLevels(nLevels), mfScaleFactor(fScaleFactor), mbGPUInitialized(false)
{
    float scale = 1.0f;
    float invScale = 1.0f;
    for (int i = 0; i < mnLevels; i++)
    {
        mvInvScaleFactor.push_back(invScale);
        scale *= mfScaleFactor;
        invScale = 1.0f / scale;
    }
    std::cout << "[CUDA] Extractor initializat." << std::endl;
}

// Destructor
CudaORBextractor::~CudaORBextractor()
{
    FreeMemory();
}

// Eliberam memoria alocata pe GPU
void CudaORBextractor::FreeMemory()
{
    // Eliberam fiecare pointer alocat
    for (auto ptr : mvdImagePyramid) {
        if(ptr) cudaFree(ptr);
    }
    mvdImagePyramid.clear();
    mvWidths.clear();
    mvHeights.clear();
    mvPitches.clear();
    mbGPUInitialized = false;
    std::cout << "[CUDA] Memorie eliberata." << std::endl;
}

// Initializam memoria pe GPU pentru piramida de imagini
void CudaORBextractor::InitGPU(int width, int height)
{
    if (mbGPUInitialized) return;

    std::cout << "[CUDA] Alocare memorie pentru " << width << "x" << height << "..." << std::endl;

    for (int i = 0; i < mnLevels; i++)
    {
        // Calculam dimensiunile pentru fiecare nivel
        int w = ceil((double)width * mvInvScaleFactor[i]);
        int h = ceil((double)height * mvInvScaleFactor[i]);

        unsigned char* d_img;
        size_t pitch;

        // Alocam memorie pe GPU cu padding (Pitch)
        CUDA_CHECK(cudaMallocPitch((void**)&d_img, &pitch, w * sizeof(unsigned char), h));

        mvdImagePyramid.push_back(d_img);
        mvWidths.push_back(w);
        mvHeights.push_back(h);
        mvPitches.push_back((int)pitch);
    }
    
    mbGPUInitialized = true;
    std::cout << "[CUDA] Memorie alocata cu succes." << std::endl;
}

// Descarcam un nivel specific al piramidei de pe GPU catre CPU
void CudaORBextractor::ApplyGaussianBlur(unsigned char* src, unsigned char* dst, int width, int height, int pitch)
{
    dim3 blockSize(32, 32);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
    GaussianBlurKernel<<<gridSize, blockSize>>>(src, dst, width, height, pitch);
    CUDA_CHECK(cudaGetLastError());
}

// Construim piramida de imagini pe GPU
void CudaORBextractor::ComputePyramid(const cv::Mat& image)
{
    // 1. Upload Nivel 0
    CUDA_CHECK(cudaMemcpy2D(mvdImagePyramid[0], mvPitches[0],
                            image.data, image.step,
                            image.cols, image.rows,
                            cudaMemcpyHostToDevice));

    // 2. Procesare Niveluri 1 -> N
    for (int i = 1; i < mnLevels; ++i)
    {
        // Configurare Grila
        dim3 blockSize(32, 32);
        dim3 gridSize((mvWidths[i] + blockSize.x - 1) / blockSize.x, 
                      (mvHeights[i] + blockSize.y - 1) / blockSize.y);

        // Resize (Din nivelul anterior i-1 in nivelul curent i)
        ResizeKernel<<<gridSize, blockSize>>>(mvdImagePyramid[i-1], mvPitches[i-1], mvWidths[i-1], mvHeights[i-1],
                                              mvdImagePyramid[i], mvPitches[i], mvWidths[i], mvHeights[i],
                                              mfScaleFactor);
        
        // Blur (Pe nivelul curent i)
        ApplyGaussianBlur(mvdImagePyramid[i], mvdImagePyramid[i], mvWidths[i], mvHeights[i], mvPitches[i]);
    }
    
    cudaDeviceSynchronize();
}

void CudaORBextractor::ExtractORB(cv::InputArray _image)
{
    cv::Mat image = _image.getMat();
    if(image.empty()) return;
    if(!mbGPUInitialized) InitGPU(image.cols, image.rows);
    ComputePyramid(image);
}

void CudaORBextractor::DownloadPyramidLevel(int level, cv::Mat& output)
{
    if(level >= mnLevels || level < 0) return;

    // Alocam imaginea OpenCV cu dimensiunile corecte
    output.create(mvHeights[level], mvWidths[level], CV_8UC1);

    // Copiem din VRAM in RAM
    CUDA_CHECK(cudaMemcpy2D(output.data, output.step, 
                            mvdImagePyramid[level], mvPitches[level], 
                            mvWidths[level], mvHeights[level], 
                            cudaMemcpyDeviceToHost));
}

} // namespace ORB_SLAM3