# ORB-SLAM3 Accelerated on NVIDIA Jetson Orin Nano 🚀

![C++](https://img.shields.io/badge/C++-11%2F14%2F17-blue.svg)
![CUDA](https://img.shields.io/badge/CUDA-11.4+-green.svg)
![Platform](https://img.shields.io/badge/Platform-NVIDIA%20Jetson-76B900.svg)
![License](https://img.shields.io/badge/License-GPLv3-red.svg)

**An optimized implementation of the ORB-SLAM3 visual SLAM system, designed for embedded edge computing platforms.**

This project addresses the high computational latency of visual feature extraction on mobile processors by implementing a **Hybrid CPU-GPU Architecture**.

---

## ⚡ Key Features & Optimizations

### 1. CUDA Accelerated ORB Extraction (GPU)
* Replaced the sequential descriptor extraction with a massive parallel **CUDA Kernel**.
* Utilizes **Constant Memory** for ORB patterns to minimize global memory latency.
* Achieves **~5x speedup** in the descriptor calculation stage (from 23ms to 4ms).

### 2. Grid-Based Feature Distribution (CPU)
* Replaced the recursive QuadTree algorithm (standard in ORB-SLAM3) with a linear **Grid-based filtering** approach.
* Ensures uniform feature distribution with **O(N)** complexity.
* Reduces CPU overhead and branch mispredictions.

### 3. Optimized Memory Management
* Implemented **Static Memory Pooling** on the GPU to avoid `cudaMalloc` overhead per frame.
* Zero-copy data transfer where applicable (Unified Memory architecture).

---

## 📊 Performance Results

Tested on **NVIDIA Jetson Orin Nano** using the **KITTI Odometry Benchmark (Sequence 00)**.

| Processing Stage | Original (CPU) | **Accelerated (CPU+GPU)** | Speedup |
| :--- | :---: | :---: | :---: |
| Pyramid Building | 1.52 ms | 2.45 ms | - |
| **Feature Distribution** | **20.85 ms** | **11.20 ms** | **1.86x** |
| Gaussian Blur | 2.65 ms | 4.10 ms | - |
| **ORB Descriptor** | **21.40 ms** | **4.35 ms** | **4.92x** |
| Data Transfer | 2.45 ms | 0.45 ms | 5.44x |
| **TOTAL FRAME TIME** | **~49 ms** | **~22 ms** | **~2.2x** |
| **FPS** | **20 FPS** | **45 FPS** | **Real-Time** |

> *Note: While the GPU kernel is 5x faster, the total system speedup is governed by Amdahl's Law, resulting in a 2.2x overall improvement.*

---

## 🛠️ Hardware & Software Requirements

### Hardware
* **Device:** NVIDIA Jetson Orin Nano / Xavier NX / AGX Orin.
* **Storage:** NVMe SSD (Recommended for high-bandwidth dataset reading).

### Software
* **OS:** Ubuntu 20.04 (JetPack 5.x) or Ubuntu 22.04/24.04.
* **Compilers:** GCC 9+, NVCC 11.4+.
* **Libraries:**
    * OpenCV 4.4+ (Must be compiled with CUDA & Eigen support).
    * Eigen3.
    * Pangolin (for visualization).

---

## 📦 Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/tothantonio/ORB_SLAM3_Accelerated.git
   cd ORB_SLAM3_Accelerated
2. **Build the project: We provide a script to handle the mixed C++/CUDA compilation.**
   ```bash
   chmod +x build.sh
   ./build.sh
3. **Download KITTI Dataset: Download the grayscale odometry sequences from the KITTI Website.**

## 🚀 Usage

To run the Monocular SLAM on KITTI Sequence 00:
```bash
# Using the helper script
chmod +x run.sh
./run.sh
```
**Manual execution:**
```bash
./Examples/Monocular/mono_kitti \
    Vocabulary/ORBvoc.txt \
    Examples/Monocular/KITTI00-02.yaml \
    /path/to/dataset/sequences/00
```

**Controls**

    Map Viewer: Use the mouse to rotate/zoom the 3D map.

    Terminal: Real-time profiling logs are printed to stdout.

    Exit: Press Ctrl+C.
    
## 📂 Project Structure

    src/ORBextractor.cc: Modified class with Grid Distribution logic.

    src/OrbCuda.cu: CUDA Kernels and device memory management.

    include/OrbCuda.h: Header for C++/CUDA interfacing.

    Examples/: Executables for Monocular mode.

# 🤝 Acknowledgements

This project is based on ORB-SLAM3 by Carlos Campos, Richard Elvira, Juan J. Gómez Rodríguez, José M. M. Montiel and Juan D. Tardós.

Modifications by: Toth Antonio-Roberto

Technical University of Cluj-Napoca
