#ifndef CUDAORBEXTRACTOR_H
#define CUDAORBEXTRACTOR_H

#include <vector>
#include <opencv2/core/core.hpp>
#include <cuda_runtime.h>

namespace ORB_SLAM3
{

class CudaORBextractor
{
public:
    CudaORBextractor(int nLevels, float fScaleFactor);
    ~CudaORBextractor();
    void ExtractORB(cv::InputArray _image);
    void DownloadPyramidLevel(int level, cv::Mat& output);

protected:
    void InitGPU(int width, int height);
    void FreeMemory();

    void ComputePyramid(const cv::Mat& image);
    void ApplyGaussianBlur(unsigned char* src, unsigned char* dst, int width, int height, int pitch);
    
    int mnLevels;
    float mfScaleFactor;
    std::vector<float> mvInvScaleFactor;

    std::vector<unsigned char*> mvdImagePyramid; 
    std::vector<int> mvWidths;
    std::vector<int> mvHeights;
    std::vector<int> mvPitches;

    bool mbGPUInitialized;
};

} // namespace ORB_SLAM3

#endif // CUDAORBEXTRACTOR_H