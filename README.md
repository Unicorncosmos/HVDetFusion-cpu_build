# HVDetFusion-cpu_build

Segmentation_model: Internimage
Ops: DCNv3
Transformer: LSSviewTransformer + bevpoolv2_cpu

This repository will generate result.pkl fusion data

## Overview
The InternImage backbone with dcnv3 has cpu build capabilities 
also bevpool and lssview transformer has cpu build capabilities 
This readme provides instructions for setting up and using the CPU version of the SFusion model with DCNV3, LSSViewTransformer, and DepthNet on the NuScenes dataset. SFusion is a sensor fusion model designed to work with radar and camera data for 3D Segmentation and depth estimation.

## Table of Contents

- [Prerequisites](#prerequisites)

## Prerequisites

```shell
mmcv=1.4.0=pypi_0
mmcv-full=1.4.0=pypi_0
mmdet=2.28.1=dev_0
mmengine=0.8.2=pypi_0
mmsegmentation=0.30.0=pypi_0  
onnx=1.14.1=pypi_0
onnxruntime=1.16.0=pypi_0
pytorch=1.9.0=py3.9_cpu_0
```

## To run this model on CPU

To run this model on CPU, you will need
Either 

1. A machine with a CPU that supports AVX2 instructions
2. A machine with a CPU that has minimum 16GB RAM.


## Prepare Datasets
- Prepare nuScenes dataset(v1.0trainval)
Download nuScenes 3D detection [data](https://www.nuscenes.org/download) and unzip all zip files.
The folder structure should be organized as follows before our processing.

```
HVDetFusion-cpu_build
├── mmdet3d
├── tools
├── configs
├── data
│   ├── nuscenes
│   │   ├── maps
│   │   ├── samples
│   │   ├── sweeps
│   │   ├── v1.0-test
|   |   ├── v1.0-trainval
```

# Setup and Inference(Before running the docker make sure the Dataset is prepared based on the above dir structure) 
```
docker build -t sfusion_cpu .
```

# Inference
```angular2html
python tools/HVDet_infer.py configs/hvdet/HVDetInfer_sim.py tools/convter2onnx/onnx_output --fuse-conv-bn --eval bbox  # --offline_eval --out ./res_pkl/test.pkl
```

## Acknowledgement

This work is built on the open-sourced [HVDetFusion](https://github.com/HVXLab/HVDetFusion/), [BevDet](https://github.com/HuangJunJie2017/BEVDet),[BevDepth](https://github.com/Megvii-BaseDetection/BEVDepth) and the published code of [CenterFusion](https://github.com/mrnabati/CenterFusion).

## License
This project is released under the Apache 2.0 license.

 
