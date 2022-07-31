# NARF GAN

Train NARF without 3D supervision

## install cuda extension
```angular2html
cd cuda_extention
python setup.py install
```

## preparation
I'm planning to make the NARF pip installable when I publish it.

Tentatively put a symbolic link so that NARF can be imported.

```angular2html
ln -s path_to_NARF_repo NARF
```

Or simply git clone NARF here.

```angular2html
git clone https://github.com/nogu-atsu/unsupervised_pose_disentanglement.git NARF
```

## training

3d pose prior

```angular2html
CUDA_VISIBLE_DEVICES=1 python train_NARF_GAN.py --config configs/NARF_GAN/THUman/20210914_NoPoseAndRayDirection.yml --num_workers 2
```

2d annotation

```angular2html
CUDA_VISIBLE_DEVICES=1 python train_NARF_GAN_from_2d.py --config configs/NARF_GAN/THUman/20210914_NoPoseAndRayDirection.yml --num_workers 2
```

train encoder with 3D pose suprevision
```
CUDA_VISIBLE_DEVICES=1 python train_encoder_supervised.py --config configs/NARF_GAN/THUman/20210914_NoPoseAndRayDirection.yml --num_workers 2
```

## 7/1
NeRF_baseにほぼ全てのレンダリング関数などを移植して，NARFのinferenceが前と同じように走ることを確認したい．
子クラスのメソッドで必要な入力変数を準備してしまってから，共通の関数で処理する．

## 6/18
nerfのレンダリングで使うextrinsicsと
narfのレンダリングで使うpose_to_cameraは実は同じものなので，
実装を完全に共通化できる．

## 6/1

```angular2html
root
├── libraries
│    ├── stylegan2
│    ├── triplanes
│    ├── GAN_training
│    ├── just clean NeRF
│    ├── NARF
```
- merge training codes
