#include <cuda.h>
#include <iostream>

#include "ORBextractor.h"

#include "orientation.h"

/*
    * __device__ helper that computes the orientation angle (in degrees)
    * for a keypoint located at (x, y) in the given image using the
    * intensity centroid method. This function is executed on the GPU
    * and is intended to be inlined into caller kernels for performance.
    *
    * Parameters:
    * - image: pointer to the image (single-channel, uchar) for the current
    *          pyramid level or the original image depending on caller.
    * - x, y:  coordinates of the keypoint within the image.
    * - u_max: pointer to a precomputed array that holds the maximum
    *          horizontal offsets for each vertical offset within the
    *          circular patch used for orientation computation.
    * - imageStep: number of bytes per image row (stride) for indexing
    *
    * Returns:
    * - angle in degrees [0, 360) representing the keypoint orientation.
*/
__device__ inline float ic_angle_gpu(const uchar *image, int x, int y, int *u_max, int imageStep) {
    int m_01 = 0, m_10 = 0;

    int center_index = x + y * imageStep;

    const uchar* center = &(image[center_index]);

    // Treat the center line differently, v=0
    #pragma unroll
    for (int u = -HALF_PATCH_SIZE; u <= HALF_PATCH_SIZE; ++u)
        m_10 += u * center[u];

    // Go line by line in the circuI853lar patch
    const int step = imageStep;
    #pragma unroll
    for (int v = 1; v <= HALF_PATCH_SIZE; ++v)
    {
        // Proceed over the two lines
        int v_sum = 0;
        const int d = u_max[v];
        for (int u = -d; u <= d; ++u)
        {
            int val_plus = center[u + v*step], val_minus = center[u - v*step];
            v_sum += (val_plus - val_minus);
            m_10 += u * (val_plus + val_minus);
        }
        m_01 += v * v_sum;
    }

    float kp_dir = atan2f((float)m_01, (float)m_10);
    kp_dir += (kp_dir < 0) * (2.0f * CV_PI);
    kp_dir *= 180.0f / CV_PI;

    return kp_dir;
}

__global__ void compute_orientation_kernel(const uchar *images, const uchar *inputImage, ORB_SLAM3::GpuPoint *pointsTotal, const uint *sizes, int* umax, int inputImageStep, int maxLevel, const float *mvScaleFactor, int cols, int rows) {
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

    const int scaledPatchSize = PATCH_SIZE*mvScaleFactor[level];
    
    const int x = points[index].x;
    const int y = points[index].y;
    const float angle = ic_angle_gpu(myImagePyrimid, x, y, umax, imageStep);
    points[index].angle = angle;
    points[index].octave = level;
    points[index].size = scaledPatchSize;

}

void compute_orientation(uchar *images, uchar *inputImage, ORB_SLAM3::GpuPoint *points, uint *sizes, int maxPointsLevel, int* umax, int inputImageStep, int maxLevel, int cols, int rows, float *mvScaleFactor, cudaStream_t cudaStream){
    dim3 dg( ceil( (float)maxPointsLevel/128 ), ceil((float)maxLevel/8) );
    dim3 db( 128, 8 );

    compute_orientation_kernel<<<dg, db, 0, cudaStream>>>(images, inputImage, points, sizes, umax, inputImageStep, maxLevel, mvScaleFactor, cols, rows);
}
