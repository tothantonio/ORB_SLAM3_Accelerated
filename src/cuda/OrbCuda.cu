#include "OrbCuda.h"
#include <cuda_runtime.h>
#include <iostream>
#include <algorithm>

// Constante
__constant__ int c_pattern[256 * 4];  // 256 perechi de puncte (x1,y1,x2,y2) pentru descriptorii ORB
#define DEG2RAD 0.0174532925f

// Buffere reutilizabile
static unsigned char* g_d_image = nullptr; // Imaginea pe GPU
static size_t g_d_image_size = 0; // Dimensiunea imaginii pe GPU
static GpuPoint* g_d_keypoints = nullptr; // Keypoint-urile pe GPU
static size_t g_d_keypoints_capacity = 0; // Capacitatea buffer-ului de keypoint-uri
static unsigned char* g_d_descriptors = nullptr; // Descriptorii pe GPU

// __restrict__ hint pt compilator ca pointerii nu se suprapun in memorie
__global__ void computeDescriptorsKernel(const unsigned char* __restrict__ image, int step, 
    const GpuPoint* __restrict__ keypoints, unsigned char* __restrict__ descriptors, int numKeypoints) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numKeypoints) return;

    float x = keypoints[idx].x;
    float y = keypoints[idx].y;
    float angle = keypoints[idx].angle * DEG2RAD;

    float a = cosf(angle); float b = sinf(angle);

    int cx = (int)(x + 0.5f); int cy = (int)(y + 0.5f); // aflam unde e centrul punctului din imagine

    int base_offset = cy * step + cx; // base address in memorie

    for (int i = 0; i < 32; ++i) { // calculez cei 32 de bytes ai descriptorului
        unsigned char byteVal = 0;
        for (int j = 0; j < 8; ++j) {
            int p_idx = ((i * 8) + j) * 4; 
            int t1_x = (int)(c_pattern[p_idx] * a - c_pattern[p_idx+1] * b + 0.5f);
            int t1_y = (int)(c_pattern[p_idx] * b + c_pattern[p_idx+1] * a + 0.5f);
            int t2_x = (int)(c_pattern[p_idx+2] * a - c_pattern[p_idx+3] * b + 0.5f);
            int t2_y = (int)(c_pattern[p_idx+2] * b + c_pattern[p_idx+3] * a + 0.5f);

            // compar pixelii si setez bitul corespunzator
            // image[] ia direct din vram
            if (image[base_offset + t1_y * step + t1_x] < image[base_offset + t2_y * step + t2_x]) {
                byteVal |= (1 << j);
            }
        }
        descriptors[idx * 32 + i] = byteVal;
    }
}

void copyPatternToGpu(const int* hostPattern, int nPoints) {cudaMemcpyToSymbol(c_pattern, hostPattern, 256 * 4 * sizeof(int));}

void launchOrbKernel(const cv::Mat& image, const std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors)
{
    int numKeypoints = keypoints.size();
    if (numKeypoints == 0) return;

    size_t image_size = image.rows * image.step;
    if (!g_d_image) {
        cudaMalloc((void**)&g_d_image, image_size);
        g_d_image_size = image_size;
    } 

    if (numKeypoints > g_d_keypoints_capacity) {
        if (g_d_keypoints) cudaFree(g_d_keypoints);
        if (g_d_descriptors) cudaFree(g_d_descriptors);
        
        size_t new_capacity = (size_t)(numKeypoints * 1.5);
        cudaMalloc((void**)&g_d_keypoints, new_capacity * sizeof(GpuPoint));
        cudaMalloc((void**)&g_d_descriptors, new_capacity * 32);
        g_d_keypoints_capacity = new_capacity;
    }

    std::vector<GpuPoint> rawPoints(numKeypoints);
    for (int i = 0; i < numKeypoints; ++i) {
        rawPoints[i].x = keypoints[i].pt.x;
        rawPoints[i].y = keypoints[i].pt.y;
        rawPoints[i].angle = keypoints[i].angle;
    }

    cudaMemcpy(g_d_image, image.data, image_size, cudaMemcpyHostToDevice);
    cudaMemcpy(g_d_keypoints, rawPoints.data(), numKeypoints * sizeof(GpuPoint), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (numKeypoints + threadsPerBlock - 1) / threadsPerBlock;

    computeDescriptorsKernel<<<blocksPerGrid, threadsPerBlock>>>(g_d_image, image.step, g_d_keypoints, g_d_descriptors, numKeypoints);

    // cudaDeviceSynchronize();

    cudaMemcpy(descriptors.data, g_d_descriptors, numKeypoints * 32, cudaMemcpyDeviceToHost);
}