# Official implementation of ENARF-GAN

**Unsupervised Learning of Efficient Geometry-Aware Neural Articulated Representations** \
Atsuhiro Noguchi, Xiao Sun, Stephen Lin, Tatsuya Harada

[Project page](https://nogu-atsu.github.io/ENARF-GAN/) / [Paper](https://nogu-atsu.github.io/ENARF-GAN/)

## Installation

```angular2html
git clone --recursive git@github.com:nogu-atsu/ENARF-GAN.git
cd ENARF-GAN
conda create -n enarfgan python=3.9
conda activate enarfgan
pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
cd cuda_extention
python setup.py install

# For dataset preprocessing
cd ../
git clone git@github.com:google/aistplusplus_api.git
cd aistplusplus_api
pip install -r requirements.txt
python setup.py install
```

We have only tested the code on NVIDIA A100, A6000, and RTX3080Ti GPUs.

If you get `RuntimeError: Ninja is required to load C++ extension`, [this](https://github.com/zhanghang1989/PyTorch-Encoding/issues/167) may be helpful.

## Dataset Preprocessing

You only need to generate sample data for the demo.

### Training data format

Dictionary of all data is stored in a single pickle file.

```python
{
  "img": numpy array of all images. each image is compressed by blosc. [N],
  "camera_intrinsic": camera intrinsic matrix [N, 3, 3],
  "camera_rotation": camera rotation matrix (optional) [N, 3, 3],
  "camera_translation": camera translation matrix (optional)[N, 3, 1],
  "smpl_pose": pose of SMPL. pose is in world coordinate if camera rotation and translation are provided, otherwise in camera coordinate.[N, 24, 4, 4],
  "frame_id": frame index of video (optional) [N]
}
```

### Prepare SMPL models

- Download the SMPL models
  following [EasyMocap installation](https://github.com/zju3dv/EasyMocap/blob/master/doc/installation.md). You only need
  to download smplx models.
  ```
  smpl_data
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

- Download the DeepCap dataset used for training Neural Actor
  from [here](https://vcai.mpi-inf.mpg.de/projects/NeuralActor/) (S1_marc.zip, S2_lan.zip)
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
- Sample data generation
  ```
  cd data_preprocess
  python NeuralActor/prepare_sample_data.py --data_path <path_to_NeuralActor> --person_name lan
  python NeuralActor/prepare_sample_data.py --data_path <path_to_NeuralActor> --person_name marc
  ```
- Training data generation
  ```
  cd data_preprocess
  python NeuralActor/preprocess.py --data_path <path_to_NeuralActor>
  ```

### ZJU MOCAP

- Download the [ZJU MOCAP dataset](https://github.com/zju3dv/animatable_nerf/blob/master/INSTALL.md#zju-mocap-dataset)
  used for training AnimatableNeRF
  ```
  <path_to_zju>
        ├── CoreView_313
        ├── CoreView_315
        └── CoreView_386
  ```
- Sample data generation
  ```
  cd data_preprocess
  python ZJU/prepare_sample_data.py --data_path <path_to_zju> --person_id 313 
  ```
- Training data generation
  ```
  cd data_preprocess
  python ZJU/preprocess.py --data_path <path_to_zju>
  ```

### SURREAL Dataset

- Download the [SURREAL dataset](https://github.com/gulvarol/surreal) as
  ```
  <path_to_surreal>
      ├── test
      ├── train
      └── val
  ```
- Sample data generation
  ```
  cd data_preprocess
  python surreal/prepare_sample_data.py --data_path <path_to_surreal>
  ```
- Training data generation
  ```
  cd data_preprocess
  python surreal/preprocess.py --data_path
  ```

### AIST++ Dataset

- Download the [AIST++ dataset](https://google.github.io/aistplusplus_dataset/download.html) as
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
- Sample data generation
  ```
  cd data_preprocess
  python AIST/prepare_sample_data.py --data_path <path_to_aist++> --annotation_path <path_to_annotatin>
  ```
- Training data generation
  ```
  cd data_preprocess
  python AIST/preprocess.py --data_path <path_to_aist++> --annotation_path <path_to_annotatin>
  ```

## Demo
Please run sample data generation before running the demo.

### Pretrained models

- Download the pretrained models from [here](https://drive.google.com/drive/folders/18ztt3_VKUX7P3IrKSbNedH6xNTUPiMwW?usp=sharing)
  ```
  data 
    └── result
        ├── DSO
        │   ├── NeuralActor
        │   │   ├── lan_denarf
        │   │   │   └── snapshot_latest.pth
        │   │   └── ...
        │   └── ZJU
        │       ├── 313_denarf
        │       │   └── snapshot_latest.pth
        │       └── ...
        └── GAN
            ├── AIST
            │   └── enarfgan
            │       └── snapshot_latest.pth
            └── SURREAL
                └── enarfgan
                    └── snapshot_latest.pth
  ```

### Dynamic Scene Overfitting (DSO)

- Example command for running the DSO model
  ```
  python DSO_demo.py --config configs/DSO_demo/NeuralActor/lan_denarf.yml 
  ```
- Synthesized images are saved in `data/result/DSO/NeuralActor/lan_denarf/samples`

### GAN

- Example command for running the GAN model
  ```
  python ENARF_GAN_demo.py --config configs/enarfgan_demo/AIST/enarfgan.yml
  ```
- Synthesized images are saved in `data/result/GAN/AIST/enarfgan/samples`
- If you get Out of Memory error, please try multiple times or reduce `ray_batchsize` in `libraries/NARF/mesh_rendering.py`

## Training
We tested training on a single A100 GPU.

- Example command for training the DSO model
  ```
  python train_DSO.py --config configs/DSO_train/ZJU/313_denarf.yml --default_config configs/SSO/default.yml
  ```
- Example command for training the GAN model
  ```
  python train_ENARF_GAN.py --config configs/enarfgan_train/AIST/relu.yml --default_config configs/NARF_GAN/default.yml
  ```
- If there is not enough memory, try reducing bs (batchsize) or increasing n_accum_step in config.
## Evaluation
### DSO
```
python train_DSO.py --validation --config configs/DSO_train/NeuralActor/lan_denarf.yml --num_workers 2 --resume_latest
```
### GAN (only for SURREAL)

Please install [mmpose](https://github.com/open-mmlab/mmpose) before running `compute_fid.py` 
```
python evaluation/compute_depth.py --config configs/enarfgan_train/SURREAL/config.yml --num_workers 2 --iteration -1 --truncation 0.4
python evaluation/compute_PCK.py --config configs/enarfgan_train/SURREAL/config.yml --num_workers 2 --iteration -1 --truncation 0.4
python evaluation/compute_fid.py --config configs/enarfgan_train/SURREAL/config.yml --num_workers 2 --iteration -1
```