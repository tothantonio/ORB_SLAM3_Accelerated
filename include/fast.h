#ifndef FAST
#define FAST

#include <opencv2/core/hal/interface.h>

#include "ORBextractor.h"

void fast_extract( uchar *images, uchar *inputImage, uint8_t th, uint8_t th_low, uint8_t *d_Rs, uint8_t *d_Rs_low, int *points, 
    int n_, ORB_SLAM3::GpuPoint *buffers, uint *sizes, int cols, int rows, int inputImageStep, float* mvScaleFactor, int maxLevel, 
    cudaStream_t cudaStream, cudaEvent_t interComplete, cv::Mat pyramid);

#endif