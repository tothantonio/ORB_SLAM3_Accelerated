#ifndef RESIZE
#define RESIZE

#include <opencv2/core/hal/interface.h>

void resize(uint old_h, uint old_w, float *_scaleFactor, uchar *original_img, 
    uchar *new_images, uint maxLevel, uint imageStep, cudaStream_t stream);

#endif