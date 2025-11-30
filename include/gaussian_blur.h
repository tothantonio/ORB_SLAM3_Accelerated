#ifndef GAUSSIAN_BLUR
#define GAUSSIAN_BLUR

#include <opencv2/core/hal/interface.h>
#include <cuda.h>

void gaussian_blur( uchar *images, uchar *inputImage, uchar *imagesBlured, uchar *inputImageBlured, 
    float *kernel, int cols, int rows, int inputImageStep, float* mvScaleFactor, int maxLevel, cudaStream_t cudaStream);

#endif