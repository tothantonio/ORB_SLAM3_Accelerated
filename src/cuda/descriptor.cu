#include <cuda.h>
#include <iostream>
#include "ORBextractor.h"
#include "descriptor.h"

/*
 * comp_descr:
 * __device__ helper that computes the 256-bit ORB descriptor (stored
 * as 32 bytes) for a single keypoint `pt` using a precomputed sampling
 * `pattern`. This function is executed on the GPU and is intended to be
 * inlined into caller kernels for performance.
 *
 * Parameters:
 * - image: pointer to the image (single-channel, uchar) for the current
 *          pyramid level or the original image depending on caller.
 * - pt:    reference to an `ORB_SLAM3::GpuPoint` that holds the keypoint
 *          position (`x`, `y`), orientation (`angle`) and the output
 *          `descriptor` byte array where the result is stored.
 * - pattern: pointer to an array of `cv::Point` pairs describing the
 *            sampling pattern used by ORB (each loop iteration consumes
 *            16 pairs to compute one descriptor byte across 32 bytes).
 * - imageStep: number of bytes per image row (stride) for indexing
 *
 * Behavior/notes:
 * - Rotates the sampling pattern by the keypoint orientation and reads
 *   intensity values from the rotated coordinates.
 * - Compares intensity pairs to form bits of the descriptor and writes
 *   32 descriptor bytes into `pt.descriptor`.
 */
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

/*
 * compute_descriptor_kernel:
 * CUDA kernel that computes ORB descriptors for a set of keypoints across
 * pyramid levels. The kernel is launched with a 2D grid where the X
 * dimension indexes keypoint slots (threads per level) and the Y dimension
 * indexes the pyramid `level`.
 *
 * Parameters:
 * - images: device buffer containing image pyramid levels concatenated
 *           (except level 0 which may be in `inputImage`). Layout is
 *           assumed to match indexing used here: `level*cols*rows`.
 * - inputImage: device pointer to the original level-0 image data.
 * - pointsTotal: array of `ORB_SLAM3::GpuPoint` storing all keypoints for
 *                every level; points for level L are at offset
 *                `level * cols * rows`.
 * - sizes: array with `sizes[level]` = number of keypoints at that level.
 * - pattern: sampling pattern used to compute descriptors (shared across)
 * - inputImageStep: stride (bytes per row) for the level-0 `inputImage`
 * - maxLevel: maximum pyramid levels to process (height of `sizes`)
 * - mvScaleFactor: per-level scale factors (used to compute sampling size)
 * - cols, rows: base image width/height used to compute offsets into the
 *               packed pyramid buffer for higher levels.
 *
 * Kernel mapping notes:
 * - `index` is the point index within a level computed from block/thread
 *   X indices. `level` is computed from block/thread Y indices.
 * - If `index >= sizes[level]` the thread exits; similarly threads with
 *   `level >= maxLevel` return early.
 * - The kernel selects either `inputImage` (level 0) or the packed
 *   `images[level]` buffer for non-zero levels. It computes a per-level
 *   `imageStep` (stride) used by `comp_descr`.
 *
 * Thread-safety/assumptions:
 * - Each thread writes only to its own `points[index].descriptor` so no
 *   synchronization is required here.
 */
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

/*
 * compute_descriptor (host):
 * Host function that prepares and launches the CUDA kernel to compute
 * ORB descriptors for a batch of keypoints across pyramid levels.
 *
 * Parameters mirror those of the kernel plus a `cudaStream` to run on.
 * - images, inputImage, points, sizes, pattern, inputImageStep, maxLevel,
 *   cols, rows, mvScaleFactor
 * - maxPointsLevel: maximum number of points across all levels used only
 *   to size the grid's X dimension (total threads per Y-level).
 * - cudaStream: stream to launch the kernel on (use 0 for default stream).
 *
 * Launch configuration:
 * - block size `db` is (128, 8): 128 threads in X (points per block) and
 *   8 threads in Y (levels per block). Grid dims `dg` are computed to
 *   cover `maxPointsLevel` in X and `maxLevel` in Y.
 */
void compute_descriptor(uchar *images, uchar *inputImage, ORB_SLAM3::GpuPoint *points, uint *sizes, int maxPointsLevel, 
    cv::Point* pattern, int inputImageStep, int maxLevel, int cols, int rows, float *mvScaleFactor, cudaStream_t cudaStream){

    dim3 dg( ceil( (float)maxPointsLevel/128 ), ceil((float)maxLevel/8) );
    dim3 db( 128, 8 );
    compute_descriptor_kernel<<<dg, db, 0, cudaStream>>>(images, inputImage, points, sizes, pattern, inputImageStep, maxLevel, mvScaleFactor, cols, rows);
}
