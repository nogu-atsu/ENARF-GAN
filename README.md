# NARF GAN

Train NARF without 3D supervision

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