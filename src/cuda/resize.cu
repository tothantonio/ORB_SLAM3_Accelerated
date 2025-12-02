#include <cuda.h>
#include <opencv2/core/hal/interface.h>
#include <stdio.h>

#include "resize.h"

#define index(x, y, step) (y * step + x)

/*
    * CUDA kernel that resizes the original image into multiple pyramid levels.
    * Each pyramid level is scaled down by a precomputed scale factor.
    * Parameters:
    * - old_h, old_w: dimensions of the original image
    * - _scaleFactor: array of scale factors for each pyramid level
    * - original_img: pointer to the original image data (single-channel uchar)
    * - new_images: pointer to the packed pyramid images buffer
    * - maxLevel: number of pyramid levels to generate
    * - imageStep: stride (bytes per row) of the original image
    *
    * note:
    * - Each thread computes one pixel in one pyramid level.
    * - returns nothing; writes directly to new_images.
*/

__global__ void resize_kernel(uint old_h, uint old_w, float *_scaleFactor, const uchar *original_img,
     uchar *new_images, uint maxLevel, uint imageStep) {

    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    const int level = (blockIdx.z * blockDim.z + threadIdx.z) + 1; // nivelul piramidei

    if (level >= maxLevel){
        return;
    }
        
    const float scaleFactor = _scaleFactor[level];
    const uint new_h = round(old_h * 1/scaleFactor);
    const uint new_w = round(old_w * 1/scaleFactor);

    if (x >= new_w || y >= new_h){
        return;
    }
    
    uchar *new_image = &(new_images[level*old_h*old_w]);
    const uint newImageStep = new_w;
    uchar *newPixel = &(new_image[index(x, y, newImageStep)]);

    const float old_x = x * scaleFactor;
    const float old_y = y * scaleFactor;
    const int x_floor = floor(old_x); // stange
    const int x_ceil = min(old_w - 1, (int)ceil(old_x)); // dreapta
    const int y_floor = floor(old_y); // sus
    const int y_ceil = min(old_h - 1, (int)ceil(old_y)); // jos

    // intensitatea celor 4 pixeli vecini
    const uchar v1 = original_img[index(x_floor, y_floor, imageStep)];
    const uchar v2 = original_img[index(x_ceil, y_floor, imageStep)];
    const uchar v3 = original_img[index(x_floor, y_ceil, imageStep)];
    const uchar v4 = original_img[index(x_ceil, y_ceil, imageStep)];

    const float q1 = (x_ceil != x_floor) ? (v1 * ((x_ceil - old_x)/(x_ceil-x_floor)) + v2 * ((old_x - x_floor)/(x_ceil-x_floor))) : 
    (x_ceil == x_floor) * v1;

    const float q2 = (x_ceil != x_floor) ? (v3 * ((x_ceil - old_x)/(x_ceil-x_floor)) + v4 * ((old_x - x_floor)/(x_ceil-x_floor))) : 
    (x_ceil == x_floor) * v4;

    const float q = (y_ceil != y_floor) ? (q1 * ((y_ceil - old_y)/(y_ceil-y_floor)) + q2 * ((old_y - y_floor)/(y_ceil-y_floor))) : 
    (y_ceil == y_floor) * q1;

    *newPixel = q;
}


void resize(uint old_h, uint old_w, float *_scaleFactor, uchar *original_img, uchar *new_images, uint maxLevel, uint imageStep, cudaStream_t stream) {
    dim3 dg( ceil( (float)old_w/128 ), ceil( (float)old_h/8 ), ceil( (float)maxLevel/1 ) );
    dim3 db( 128, 8, 1 );

    resize_kernel<<<dg, db, 0, stream>>>(old_h, old_w, _scaleFactor, original_img, new_images, maxLevel, imageStep);
}