#ifndef CUDAORBEXTRACTOR_H
#define CUDAORBEXTRACTOR_H

#include <vector>
#include <opencv2/core/core.hpp>

namespace ORB_SLAM3
{
class CudaORBextractor
{
public:
    CudaORBextractor();

    void WarmupGPU();
};
} // namespace ORB_SLAM3

#endif // CUDAORBEXTRACTOR_H