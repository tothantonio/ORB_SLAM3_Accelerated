#ifndef ORB_CUDA_H
#define ORB_CUDA_H

#include <opencv2/core/core.hpp>

struct GpuPoint {
    float x;
    float y;
    float angle;
};

void copyPatternToGpu(const int* hostPattern, int nPoints);
void launchOrbKernel(const cv::Mat& image, const std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors);

#endif // ORB_CUDA_H