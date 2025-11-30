#include <cuda.h>
#include <iostream>
#include "ORBextractor.h"
#include "descriptor.h"

__device__ inline void comp_descr(const uchar *image, ORB_SLAM3::GpuPoint &pt, cv::Point *pattern, int imageStep) {
        const float factorPI = (float)(CV_PI/180.f);
        const float angle = (float)pt.angle*factorPI;
        const float a = (float)cos(angle);
        const float b = (float)sin(angle);

        const uchar* center = &(image[(int)pt.y*imageStep+(int)pt.x]);
        const int step = imageStep;

#define GET_VALUE(idx) \
        center[(int)round(pattern[idx].x*b + pattern[idx].y*a)*step + \
               (int)round(pattern[idx].x*a - pattern[idx].y*b)]

        #pragma unroll
        for (int i = 0; i < 32; ++i, pattern += 16)
        {
            int t0, t1, val;
            t0 = GET_VALUE(0); t1 = GET_VALUE(1);
            val = t0 < t1;
            t0 = GET_VALUE(2); t1 = GET_VALUE(3);
            val |= (t0 < t1) << 1;
            t0 = GET_VALUE(4); t1 = GET_VALUE(5);
            val |= (t0 < t1) << 2;
            t0 = GET_VALUE(6); t1 = GET_VALUE(7);
            val |= (t0 < t1) << 3;
            t0 = GET_VALUE(8); t1 = GET_VALUE(9);
            val |= (t0 < t1) << 4;
            t0 = GET_VALUE(10); t1 = GET_VALUE(11);
            val |= (t0 < t1) << 5;
            t0 = GET_VALUE(12); t1 = GET_VALUE(13);
            val |= (t0 < t1) << 6;
            t0 = GET_VALUE(14); t1 = GET_VALUE(15);
            val |= (t0 < t1) << 7;

            pt.descriptor[i] = (uchar)val;
        }

#undef GET_VALUE
    }    

__global__ void compute_descriptor_kernel(uchar *images, uchar *inputImage, ORB_SLAM3::GpuPoint *pointsTotal, 
    const uint *sizes, cv::Point* pattern, int inputImageStep, int maxLevel, const float *mvScaleFactor, int cols, int rows) {

    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int level = blockIdx.y * blockDim.y + threadIdx.y;
    if (level >= maxLevel)
        return;
    
    const uint n = sizes[level];
    if (index >= n) {
        return;
    }

    ORB_SLAM3::GpuPoint *points = &(pointsTotal[level*cols*rows]);

    const uchar* im[2] = {inputImage, &(images[level*cols*rows])};
    const int imIndex = (level == 0) * 0 + (level != 0) * 1;
    const float scale = mvScaleFactor[level];
    const int new_cols = round(cols * 1/scale);
    const int imageStep = (level == 0) * inputImageStep + (level != 0) * new_cols;
    const uchar *myImagePyrimid = im[imIndex];
    
    comp_descr(myImagePyrimid, points[index], pattern, imageStep);
}

void compute_descriptor(uchar *images, uchar *inputImage, ORB_SLAM3::GpuPoint *points, uint *sizes, int maxPointsLevel, 
    cv::Point* pattern, int inputImageStep, int maxLevel, int cols, int rows, float *mvScaleFactor, cudaStream_t cudaStream){

    dim3 dg( ceil( (float)maxPointsLevel/128 ), ceil((float)maxLevel/8) );
    dim3 db( 128, 8 );
    compute_descriptor_kernel<<<dg, db, 0, cudaStream>>>(images, inputImage, points, sizes, pattern, inputImageStep, maxLevel, mvScaleFactor, cols, rows);
}
