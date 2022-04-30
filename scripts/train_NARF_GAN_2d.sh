source ~/.zshrc_cuda110
pyenv shell anaconda3-4.3.1/envs/cuda110

#CUDA_VISIBLE_DEVICES=3 python train_NARF_GAN_from_2d.py --config configs/NARF_GAN_from_2d/THUman/20211208_2d.yml --num_workers 2
CUDA_VISIBLE_DEVICES=1 python train_NARF_GAN_from_2d.py --config configs/NARF_GAN_from_2d/ZJU/20211226_zju_aligned_2d.yml --num_workers 2 --resume_latest --disable_checkpoint
