# Official implementation of ENARF-GAN

**Unsupervised Learning of Efficient Geometry-Aware Neural Articulated Representations**
Atsuhiro Noguchi, Xiao Sun, Stephen Lin, Tatsuya Harada

[Project page](https://nogu-atsu.github.io/ENARF-GAN/) / [Paper](https://nogu-atsu.github.io/ENARF-GAN/)

## Requirements
- Python 3.9
- PyTorch 1.10.1, torchvision 0.11.2

We have only tested the code on NVIDIA A100 and A6000 GPUs.
## Installation
```angular2html
git clone --recursive git@github.com:nogu-atsu/ENARF-GAN.git
cd ENARF-GAN
pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
```

## Dataset Preprocessing
### Prepare SMPL models
- Install smplx: `pip install smplx`
- Download the SMPL models
  following [EasyMocap installation](https://github.com/zju3dv/EasyMocap/blob/master/doc/installation.md). You only need to download smplx models.
  ```
  smpl_data
        └── smplx
            ├── J_regressor_body25.npy
            ├── J_regressor_body25_smplh.txt
            ├── J_regressor_body25_smplx.txt 
            ├── J_regressor_mano_LEFT.txt 
            ├── J_regressor_mano_RIGHT.txt
            ├── SMPLX_FEMALE.pkl
            ├── SMPLX_MALE.pkl
            └── SMPLX_NEUTRAL.pkl
  ```
- Run
  ```
  cd data_preprocess
  python prepare_canonical_pose.py
  ```

### DeepCap Dataset
- Download the DeepCap dataset used for training Neural Actor from [here](https://vcai.mpi-inf.mpg.de/projects/NeuralActor/) (S1_marc.zip, S2_lan.zip)
- Unzip them and `transform.zip` inside them
  ```
  <path_to_data>
        ├── lan
        │   ├── intrinsic
        │   └── ...
        └── marc
            ├── intrinsic
            └── ...
  ```
- Run
  ```
  cd data_preprocess
  python NeuralActor/prepare_sample_data.py --data_path <path_to_NeuralActor> --person_name lan
  python NeuralActor/prepare_sample_data.py --data_path <path_to_NeuralActor> --person_name marc
  ```

### ZJU MOCAP

- Requirements: [EasyMocap](https://github.com/zju3dv/EasyMocap)

- Download the [ZJU MOCAP dataset](https://github.com/zju3dv/animatable_nerf/blob/master/INSTALL.md#zju-mocap-dataset) used for training AnimatableNeRF
  ```
  <path_to_data>
        ├── CoreView_313
        ├── CoreView_315
        └── CoreView_386
  ```
- Run
  ```angular2html
  cd data_preprocess
  python ZJU_SSO/prepare_sample_data.py --data_path /data/unagi0/noguchi/dataset/animatable_nerf_zju --person_id 313 
  ```


### SURREAL Dataset
- Download the [SURREAL dataset](https://github.com/gulvarol/surreal) as
  ```
  <path_to_surreal>
      ├── test
      ├── train
      └── val
  ```
- Run
  ```
  cd data_preprocess
  python surreal/prepare_sample_data.py --data_path <path_to_surreal>
  ```

### AIST++ Dataset
- Install [aist_plusplus](https://github.com/google/aistplusplus_api/)
- Download the [AIST++ dataset](https://google.github.io/aistplusplus_dataset/) as
  ```
  <path_to_aist++>
      ├── gLO_sBM_c07_d15_mLO5_ch03.mp4
      └── ...
  <path_to_annotation>
      ├── camearas
      ├── ingore_list.txt
      ├── keypoints2d
      ├── keypoints3d
      ├── motions
      └── splits
  ```
- Run
  ```
    cd data_preprocess
  python AIST/prepare_sample_data.py --data_path <path_to_aist++> --annotation_path <path_to_annotatin>
  ```
## Demo
### Dynamic Scene Overfitting (DSO)
- Example command for running the DSO model
  ```
  python DSO_demo.py --config configs/DSO/NeuralActor/lan_denarf.yml 
  ```
- Synthesized images are saved in `data/result/DSO/NeuralActor/lan_denarf/samples`

### GAN
- Example command for running the GAN model
  ```
  python ENARF_GAN_demo.py --config configs/enarfgan/AIST/enarfgan.yml
  ```