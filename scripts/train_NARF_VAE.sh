source ~/.zshrc_cuda111
pyenv shell anaconda3-4.3.1/envs/cuda111

CUDA_VISIBLE_DEVICES=0 python train_NARF_VAE.py --config configs/NARF_VAE/SURREAL/20220228_surreal_triplane_centerFixed_mask.yml --num_workers 2 --resume_latest
