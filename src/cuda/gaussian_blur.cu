#include <cuda.h>
#include <opencv2/core/hal/interface.h>

#include "gaussian_blur.h"
#include "ORBextractor.h"

/*
    * CUDA kernel that applies a Gaussian blur to images at multiple pyramid levels.
    * Parameters:
    * - old_h, old_w: dimensions of the original image
    * - _scaleFactor: array of scale factors for each pyramid level
    * - original_img: pointer to the original image data (single-channel uchar)
    * - images: pointer to the packed pyramid images buffer
    * - original_img_blurred: pointer to the output blurred original image
    * - images_blurred: pointer to the output blurred pyramid images buffer
    * - kernel: pointer to the Gaussian kernel used for blurring
    * - maxLevel: number of pyramid levels
    * - inputImageStep: stride (bytes per row) of the original image
    *
    * Behavior/notes:
    * - Each thread computes one pixel in one pyramid level.
    * - Selects either `original_img` (level 0) or `images[level]` for non-zero levels.
    * - Writes the blurred result to `original_img_blurred` or `images_blurred[level]`.
*/

__global__ void gaussian_blur_kernel(uint old_h, uint old_w, float *_scaleFactor, const uchar *original_img, 
    const uchar *images, uchar *original_img_blurred, uchar *images_blurred, float *kernel, uint maxLevel, uint inputImageStep) {

    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int level = blockIdx.z * blockDim.z + threadIdx.z;

    if (level >= maxLevel){
        return;
    } 

    const float scaleFactor = _scaleFactor[level];
    const uint new_rows = round(old_h * 1/scaleFactor);
    const uint new_cols = round(old_w * 1/scaleFactor);

    if (x >= new_cols || y >= new_rows){
        return;
    }

    const int imageStep = (level == 0) * inputImageStep + (level != 0) * new_cols;
    const int image_index = x + y * imageStep;

    const uchar* im[2] = {original_img, &(images[(level*old_w*old_h)])};
    const int imIndex = (level != 0);

    const uchar *image = im[imIndex];

    uchar* imBlured[2] = {original_img_blurred, &(images_blurred[(level*old_w*old_h)])};
    uchar *imageBlured = imBlured[imIndex];

    float acc = 0;
    for (int w = -KW/2; w<=KW/2; w++)
        for (int h = -KH/2; h<=KH/2; h++) {
            const int index = min(max(image_index+(h*imageStep)+w, 0), new_cols*new_rows); // vecinul de la (h, w)
            acc += image[index] * kernel[(h + KH/2) * KW + (w + KW/2)];
        }
    
    imageBlured[image_index] = round(acc);
}

// lansez kernel
void gaussian_blur( uchar *images, uchar *inputImage, uchar *imagesBlured, uchar *inputImageBlured, 
    float *kernel, int cols, int rows, int inputImageStep, float* mvScaleFactor, int maxLevel, cudaStream_t cudaStream) {

    dim3 dg( ceil( (float)cols/64 ), ceil( (float)rows/8 ), ceil( (float)maxLevel/1 ) );
    dim3 db( 64, 8, 1 );

    gaussian_blur_kernel<<<dg, db, 0, cudaStream>>>(rows, cols, mvScaleFactor, inputImage, images, inputImageBlured, imagesBlured, kernel, maxLevel, inputImageStep);
}