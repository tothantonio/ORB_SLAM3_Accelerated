#include "OrbCuda.h"
#include <cuda_runtime.h>
#include <iostream>
#include <algorithm>

// Constante
__constant__ int c_pattern[256 * 4];  // 256 perechi de puncte (x1,y1,x2,y2) pentru descriptorii ORB
__constant__ int c_circle_offsets[16][2]; // Offset-urile cercului FAST
#define DEG2RAD 0.0174532925f

// Buffere reutilizabile (static)
static unsigned char* g_d_image = nullptr; // Imaginea pe GPU
static size_t g_d_image_size = 0; // Dimensiunea imaginii pe GPU
static GpuPoint* g_d_keypoints = nullptr; // Keypoint-urile pe GPU
static size_t g_d_keypoints_capacity = 0; // Capacitatea buffer-ului de keypoint-uri
static unsigned char* g_d_descriptors = nullptr; // Descriptorii pe GPU
static size_t g_d_descriptors_capacity = 0; // Capacitatea buffer-ului de descriptori

// Buffere FAST
static unsigned char* d_img_fast = nullptr; // Imaginea pentru FAST pe GPU
static size_t d_img_fast_size = 0; // Dimensiunea imaginii pentru FAST pe GPU
static short2* d_corners = nullptr; // Colțurile detectate de FAST pe GPU
static int* d_counter = nullptr; // Contorul colțurilor detectate pe GPU
const int MAX_CORNERS = 20000; // Limita hard de puncte de pe GPU

__global__ void fastKernel(const unsigned char* __restrict__ image, int step, 
    int rows, int cols, int threshold, short2* __restrict__ corners, int* __restrict__ counter, int max_points) 
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < 4 || x >= cols - 4 || y < 4 || y >= rows - 4) return;

    const int center_idx = y * step + x; // Indexul pixelului central
    const unsigned char p = image[center_idx]; // Valoarea pixelului central

    // Definirea pragurilor (mai intunecat sau mai luminos)
    int lower = (int)p - threshold;
    int upper = (int)p + threshold;

    // 1. Quick Test (Cardinale)
    int cardinal_hits = 0;
    if (image[(y-3)*step + x] < lower || image[(y-3)*step + x] > upper) cardinal_hits++;
    if (image[y*step + (x+3)] < lower || image[y*step + (x+3)] > upper) cardinal_hits++;
    if (image[(y+3)*step + x] < lower || image[(y+3)*step + x] > upper) cardinal_hits++;
    if (image[y*step + (x-3)] < lower || image[y*step + (x-3)] > upper) cardinal_hits++;

    if (cardinal_hits < 3) return;

    // 2. Full Test
    int consecutive = 0;
    bool is_corner = false;
    
    // verific daca exista 9 pixeli consecutivi mai intunecati
    for(int i=0; i<27; i++) { 
        int k = i % 16;
        int val = image[(y + c_circle_offsets[k][1]) * step + (x + c_circle_offsets[k][0])];
        if (val < lower) {
            consecutive++;
            if (consecutive >= 9) { is_corner = true; break; }
        } else { consecutive = 0; }
    }

    if (!is_corner) {
        consecutive = 0;
        // Check Brighter
        for(int i=0; i<27; i++) {
            int k = i % 16;
            int val = image[(y + c_circle_offsets[k][1]) * step + (x + c_circle_offsets[k][0])];
            if (val > upper) {
                consecutive++;
                if (consecutive >= 9) { is_corner = true; break; }
            } else { consecutive = 0; }
        }
    }

    if (is_corner) {
        // folosesc atomic pentru a evita conflictele intre thread-uri
        int idx = atomicAdd(counter, 1);
        if (idx < max_points) {
            corners[idx] = make_short2((short)x, (short)y);
        }
    }
}

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

void copyCircleOffsetsToGpu() {
    const int host_offsets[16][2] = {{0,-3},{1,-3},{2,-2},{3,-1},{3,0},{3,1},{2,2},{1,3},{0,3},{-1,3},{-2,2},{-3,1},{-3,0},{-3,-1},{-2,-2},{-1,-3}};
    cudaMemcpyToSymbol(c_circle_offsets, host_offsets, 16 * 2 * sizeof(int));
}

void launchFastGpu(const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints, int threshold) 
{
    // Alocare
    size_t needed_size = image.step * image.rows;
    if (needed_size > d_img_fast_size) {
        if (d_img_fast) cudaFree(d_img_fast);
        if (d_corners) cudaFree(d_corners);
        if (d_counter) cudaFree(d_counter);
        cudaMalloc((void**)&d_img_fast, needed_size);
        cudaMalloc((void**)&d_corners, MAX_CORNERS * sizeof(short2));
        cudaMalloc((void**)&d_counter, sizeof(int));
        d_img_fast_size = needed_size;
    }

    // Upload
    cudaMemcpy(d_img_fast, image.data, needed_size, cudaMemcpyHostToDevice);
    int zero = 0;
    cudaMemcpy(d_counter, &zero, sizeof(int), cudaMemcpyHostToDevice);

    // Launch
    dim3 threads(32, 32);
    dim3 blocks((image.cols + 31) / 32, (image.rows + 31) / 32);
    fastKernel<<<blocks, threads>>>(d_img_fast, image.step, image.rows, image.cols, threshold, d_corners, d_counter, MAX_CORNERS);
    cudaDeviceSynchronize();

    // Download
    int num_detected = 0;
    cudaMemcpy(&num_detected, d_counter, sizeof(int), cudaMemcpyDeviceToHost);
    num_detected = std::min(num_detected, MAX_CORNERS);

    if (num_detected > 0) {
        std::vector<short2> h_corners(num_detected);
        cudaMemcpy(h_corners.data(), d_corners, num_detected * sizeof(short2), cudaMemcpyDeviceToHost);
        
        // Conversie (CPU)
        keypoints.clear();
        keypoints.reserve(num_detected);
        for(int i=0; i<num_detected; i++) {
            cv::KeyPoint kp;
            kp.pt.x = (float)h_corners[i].x;
            kp.pt.y = (float)h_corners[i].y;
            keypoints.push_back(kp);
        }
    }
}

void launchOrbKernel(const cv::Mat& image, const std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors)
{
    int numKeypoints = keypoints.size();
    if (numKeypoints == 0) return;

    size_t needed_image_size = image.step * image.rows;
    if (needed_image_size > g_d_image_size) {
        if (g_d_image) cudaFree(g_d_image);
        cudaMalloc((void**)&g_d_image, needed_image_size);
        g_d_image_size = needed_image_size;
    }

    if (numKeypoints > g_d_keypoints_capacity) {
        if (g_d_keypoints) cudaFree(g_d_keypoints);
        if (g_d_descriptors) cudaFree(g_d_descriptors);
        size_t new_capacity = (size_t)(numKeypoints * 1.5);
        cudaMalloc((void**)&g_d_keypoints, new_capacity * sizeof(GpuPoint));
        cudaMalloc((void**)&g_d_descriptors, new_capacity * 32 * sizeof(unsigned char));
        g_d_keypoints_capacity = new_capacity;
        g_d_descriptors_capacity = new_capacity;
    }

    std::vector<GpuPoint> rawPoints(numKeypoints);
    for (int i = 0; i < numKeypoints; ++i) {
        rawPoints[i].x = keypoints[i].pt.x;
        rawPoints[i].y = keypoints[i].pt.y;
        rawPoints[i].angle = keypoints[i].angle;
    }

    cudaMemcpy(g_d_image, image.data, needed_image_size, cudaMemcpyHostToDevice);
    cudaMemcpy(g_d_keypoints, rawPoints.data(), numKeypoints * sizeof(GpuPoint), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (numKeypoints + threadsPerBlock - 1) / threadsPerBlock;
    computeDescriptorsKernel<<<blocksPerGrid, threadsPerBlock>>>(g_d_image, image.step, g_d_keypoints, g_d_descriptors, numKeypoints);
    cudaDeviceSynchronize();

    cudaMemcpy(descriptors.data, g_d_descriptors, numKeypoints * 32, cudaMemcpyDeviceToHost);
}