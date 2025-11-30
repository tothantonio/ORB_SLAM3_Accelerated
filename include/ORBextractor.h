/**
* This file is part of ORB-SLAM3
*
* Copyright (C) 2017-2021 Carlos Campos, Richard Elvira, Juan J. Gómez Rodríguez, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
* Copyright (C) 2014-2016 Raúl Mur-Artal, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
*
* ORB-SLAM3 is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
* License as published by the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM3 is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
* the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License along with ORB-SLAM3.
* If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef ORBEXTRACTOR_H
#define ORBEXTRACTOR_H

#include <vector>
#include <list>
#include <opencv2/opencv.hpp>

#include <cuda.h>
#include <cuda_runtime.h>

#define INIT_IMAGE_W 752
#define INIT_IMAGE_H 480

#define PATCH_SIZE 31
#define HALF_PATCH_SIZE 15
#define EDGE_THRESHOLD 19

#define KW 7
#define KH 7
#define SIGMA 2


namespace ORB_SLAM3
{

    struct GpuPoint {
        uint x;
        uint y;
        uint score;
        int octave;
        float angle;
        float size;
        int clust_assn;
        uchar descriptor[32];
    };

    struct copyPyrimid_t
    {
        int nlevels;
        uchar *outputImages;
        int cols;
        int rows;
        float *mvScaleFactor;
        cv::Mat *mvImagePyramid;
    };

    struct OrbKeyPoint {
        cv::KeyPoint point;
        uchar *descriptor;
    };

    class ORBextractor
    {
    public:

        enum {HARRIS_SCORE=0, FAST_SCORE=1 };

        ORBextractor(int nfeatures, float scaleFactor, int nlevels,
                     int iniThFAST, int minThFAST);

        ~ORBextractor();

        // Compute the ORB features and descriptors on an image.
        // ORB are dispersed on the image using an octree.
        // Mask is ignored in the current implementation.
        int operator()( cv::InputArray _image, cv::InputArray _mask,
                        std::vector<cv::KeyPoint>& _keypoints,
                        cv::OutputArray _descriptors, std::vector<int> &vLappingArea);

        int inline GetLevels(){
            return nlevels;}

        float inline GetScaleFactor(){
            return scaleFactor;}

        std::vector<float> inline GetScaleFactors(){
            return mvScaleFactor;
        }

        std::vector<float> inline GetInverseScaleFactors(){
            return mvInvScaleFactor;
        }

        std::vector<float> inline GetScaleSigmaSquares(){
            return mvLevelSigma2;
        }

        std::vector<float> inline GetInverseScaleSigmaSquares(){
            return mvInvLevelSigma2;
        }

        std::vector<cv::Mat> mvImagePyramid;

    protected:

        void ComputePyramid(cv::Mat image);
        void ComputeKeyPointsOctTree(std::vector<std::vector<OrbKeyPoint> >& allKeypoints);
        void ComputeKeyPointsOld(std::vector<std::vector<cv::KeyPoint> >& allKeypoints);
        std::vector<cv::Point> pattern;

        int nfeatures;
        double scaleFactor;
        int nlevels;
        int iniThFAST;
        int minThFAST;

        std::vector<int> mnFeaturesPerLevel;

        std::vector<int> umax;

        std::vector<float> mvScaleFactor;
        std::vector<float> mvInvScaleFactor;
        std::vector<float> mvLevelSigma2;
        std::vector<float> mvInvLevelSigma2;


        uint8_t n_ = 12;

        float maxScaleFactor;
        int allocatedSize;
        int allocatedInputSize;

        int rows;
        int cols;
        int imageStep;

        cudaStream_t cudaStream;
        cudaStream_t cudaStreamCpy;
        cudaStream_t cudaStreamBlur;
        cudaEvent_t resizeComplete;
        cudaEvent_t blurComplete;
        cudaEvent_t interComplete;
        cudaEvent_t filterKernelComplete;

        copyPyrimid_t copyPyrimidData;

        int *umax_gpu;

        uint8_t *d_R;
        uint8_t *d_R_low;

        GpuPoint *corner_buffer;
        uint *corner_size;

        GpuPoint *d_corner_buffer;
        GpuPoint *d_corner_buffer2;
        uint *d_corner_size;

        int *d_score;
        // int *score;

        int *features;

        int2 *d_centroids;
        int *d_clust_sizes;

        int2 *initial_centroids;
        int2 *centroids;

        int *d_points;
        cv::Point *d_pattern;

        float *kernel;

        //piramida
        uchar *d_images;
        uchar *d_inputImage;
        uchar *d_imagesBlured;
        uchar *d_inputImageBlured;
        uchar *outputImages;
        float *d_scaleFactor;

    private:
        void freeMemory();
        void freeInputMemory();
        void checkAndReallocMemory(cv::Mat);
        void allocMemory(int, int, int);
        void allocInputMemory(int, int, int);


    };

} //namespace ORB_SLAM

#endif

