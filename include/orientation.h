#ifndef ORIENTATION
#define ORIENTATION

#include "ORBextractor.h"

void compute_orientation(uchar *images, uchar *inputImage, ORB_SLAM3::GpuPoint *points, 
    uint *sizes, int maxPointsLevel, int* umax, int inputImageStep, int maxLevel, int cols, int rows, 
    float *mvScaleFactor, cudaStream_t cudaStream);

#endif