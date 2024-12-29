# SHARPE: State-space HumAn Pose Estimation

## Summary

* We propose a new dual-stream architecture calledSHARPE, based on state-space models (SSMs), designed for 3D Human Pose Estimation. We consider it an alternative to hybrid and Transformer based architectures, addressing their issues with processing long input sequences.
* The network is designed to be scalable. It can be easily adapted to meet different needs: having better accuracy but with the disadvantage being slower and having more parameters, or having lower accuracy but with the advantage of being lightweight (fast and fewer parameters).
* The model achieves state-of-the-art results on MPI-INF3DHP and Human36M, popular datasets for 3D Human Pose Estimation. It does this while having fewerparameters than it’s predecessors

## Architecture

![Artboard 1](https://github.com/user-attachments/assets/aa2f52ea-7b4c-486b-9e70-2082bdf11b0c)


## Setup

1. Clone this repository

```
git clone https://github.com/isacciobota/SHARPE
cd SHARPE
```

2. Run this commands for instalation of dependencies:

```
pip install -r requirements.txt
cd kernels/selective_scan
pip install .
```

Download and preprocess Human3.6M
Download and preprocess MPI-INF-3DHP

## Training

Command for Human3.6M
Command for MPI-INF-3DHP

## Evaluation

Command for Human3.6M
Command for MPI-INF-3DHP

## Inference

Details
Command
Example

## Thank you

We want to say a big thank you to the authors of the following projects. Their code served as a starting point for our model:
* [MotionBERT](https://github.com/Walter0807/MotionBERT)
* [MotionAGFormer](https://github.com/TaatiTeam/MotionAGFormer)
* [VMamba](https://github.com/MzeroMiko/VMamba/tree/main/kernels/selective_scan)
