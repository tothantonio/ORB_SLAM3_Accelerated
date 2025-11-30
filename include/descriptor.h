#ifndef DESCRIPTOR
#define DESCRIPTOR

#include "ORBextractor.h"

void compute_descriptor(uchar *images, uchar *inputImage, ORB_SLAM3::GpuPoint *points, uint *sizes, int maxPointsLevel, 
    cv::Point* pattern, int inputImageStep, int maxLevel, int cols, int rows, float *mvScaleFactor, cudaStream_t cudaStream);

#endif