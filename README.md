# SHARPE: State-space HumAn Pose Estimation

## Summary

* We propose a new dual-stream architecture calledSHARPE, based on state-space models (SSMs), designed for 3D Human Pose Estimation. We consider it an alternative to hybrid and Transformer based architectures, addressing their issues with processing long input sequences.
* The network is designed to be scalable. It can be easily adapted to meet different needs: having better accuracy but with the disadvantage being slower and having more parameters, or having lower accuracy but with the advantage of being lightweight (fast and fewer parameters).
* The model achieves state-of-the-art results on MPI-INF3DHP and Human36M, popular datasets for 3D Human Pose Estimation. It does this while having fewer parameters than it’s predecessors

## Architecture

![Artboard 1](https://github.com/user-attachments/assets/aa2f52ea-7b4c-486b-9e70-2082bdf11b0c)

## Qualitative results

![qualitative results](https://github.com/user-attachments/assets/d55c4f1c-bf4a-4531-953f-30b9cd9eefdb)

## Setup

1. Clone this repository:

```
git clone https://github.com/isacciobota/SHARPE
cd SHARPE
```

2. Run this commands for installation of dependencies:

```
pip install -r requirements.txt
cd kernels/selective_scan
pip install .
```

3. Download and preprocess Human3.6M following [MotionAGFormer](https://github.com/TaatiTeam/MotionAGFormer):

   Download the fine-tuned Stacked Hourglass detections of [MotionBERT](https://github.com/Walter0807/MotionBERT)'s preprocessed H3.6M data [here](https://onedrive.live.com/?authkey=%21AMG5RlzJp%2D7yTNw&id=A5438CD242871DF0%21206&cid=A5438CD242871DF0&parId=root&parQt=sharedby&o=OneUp) and unzip it to `data/motion3d`.

   Slice the motion clips by running the following python code in `data/preprocess` directory:
    ```
    python h36m.py --n-frames 27
    ```

5. Download and preprocess MPI-INF-3DHP following [MotionAGFormer](https://github.com/TaatiTeam/MotionAGFormer):

   Please refer [P-STMO](https://github.com/paTRICK-swk/P-STMO#mpi-inf-3dhp) for dataset setup.

   After preprocessing, the generated .npz files (`data_train_3dhp.npz` and `data_test_3dhp.npz`) should be located at `data/motion3d` directory.

## Training

Command for training on Human3.6M:
```
python train.py --config configs/h36m/config_file_name.yaml
```
Command for training on MPI-INF-3DHP:
```
python train_3dhp.py --config configs/mpi/config_file_name.yaml
```
Extra arguments:
```
--use-wandb --wandb-name name_for_wandb
--checkpoint path_to_checkpoint
```

## Evaluation

Command for evaluation on Human3.6M:
```
python train.py --eval-only --checkpoint checkpoints/h36m/path_to_checkpoint_dir --checkpoint_file checkpoint_file_name --config --configs/h36m/config_file_name.yaml
```
Command for evaluation on MPI-INF-3DHP:
```
python train_3dhp.py --eval-only --checkpoint checkpoints/mpi/path_to_checkpoint_dir --checkpoint_file checkpoint_file_name --config --configs/mpi/config_file_name.yaml
```

> [!NOTE]
> For PCK and AUC metrics on MPI-INF-3DHP clone this repository: [STCFormer](https://github.com/zhenhuat/STCFormer/tree/main/3dhp_test).
> 
> Use the generated `inference_data_best.mat` and place it in `STCFormer/checkpoint` folder.
> 
> Download the original MPI-INF-3DHP dataset from [here](https://vcai.mpi-inf.mpg.de/3dhp-dataset/). Modify the config to download all subjects and run the script. Place TS1-TS6 in the `STCFormer/3dhp_test` folder.
> 
> Run `STCFormer/3dhp_test/test_util/mpii_test_predictions_py.m` script usint Matlab.

## Inference

This is an adaption from [MotionAGFormer](https://github.com/TaatiTeam/MotionAGFormer)

1. Place your own video in the `./demo/video` folder.

2. Download YOLOv3 and HRNet pretrained models for 2D detecton from [here](https://drive.google.com/drive/folders/1_ENAMOsPM7FXmdYRbkwbFHgzQq_B_NQA). Place them in the `./demo/lib/checkpoint` folder.

3. Currently, we are using the SHARPE-tiny model, pretrained on Human3.6M. If you want to change that, make sure to modify `inference.py` to update the model and the config file.

Run this command in the base folder:
```
python inference.py --video video_file_name.mp4
```

## Running in Jupyter Notebook or Google Colab

We recommend doing the following steps:

1. Make sure you have done all the steps presented above.

2. Zip the folder that you have locally and upload it to your Google Drive account.

3. You can use the `SHARPE.ipynb` file from our repository.

## Thank you

We want to say a big thank you to the authors of the following projects. Their code served as a starting point for our model:
* [MotionBERT](https://github.com/Walter0807/MotionBERT)
* [MotionAGFormer](https://github.com/TaatiTeam/MotionAGFormer)
* [VMamba](https://github.com/MzeroMiko/VMamba/tree/main/kernels/selective_scan)
* [P-STMO](https://github.com/paTRICK-swk/P-STMO#mpi-inf-3dhp)
