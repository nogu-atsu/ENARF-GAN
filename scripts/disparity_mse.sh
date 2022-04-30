source ~/.zshrc_cuda111
pyenv shell anaconda3-4.3.1/envs/cuda111

#CUDA_VISIBLE_DEVICES=6 python evaluation/compute_depth.py --config configs/NARF_GAN/SURREAL/20220225_surreal_triplane_centerFixed.yml --num_workers 2 --iteration -1
#CUDA_VISIBLE_DEVICES=7 python evaluation/compute_depth.py --config configs/NARF_GAN/SURREAL/20220227_surreal_triplane_centerFixed_cmu.yml --num_workers 2 --iteration -1
#CUDA_VISIBLE_DEVICES=7 python evaluation/compute_depth.py --config configs/NARF_GAN/SURREAL/20220228_surreal_triplane_centerFixed_gaussian.yml --num_workers 2 --iteration -1
#CUDA_VISIBLE_DEVICES=7 python evaluation/compute_depth.py --config configs/NARF_GAN/SURREAL/20220226_surreal_stylenerf_centerFixed.yml --num_workers 2 --iteration -1
#CUDA_VISIBLE_DEVICES=7 python evaluation/compute_depth.py --config configs/NARF_GAN/SURREAL/20220228_surreal_triplane_centerFixed_no_reg.yml --num_workers 2 --iteration -1
#CUDA_VISIBLE_DEVICES=7 python evaluation/compute_depth.py --config configs/NARF_VAE/SURREAL/20220228_surreal_triplane_centerFixed_mask.yml --num_workers 2 --iteration -1
#CUDA_VISIBLE_DEVICES=7 python evaluation/compute_depth.py --config configs/NARF_GAN/SURREAL/20220302_surreal_triplane_centerFixed_gaussian_squareReg.yml --num_workers 2 --iteration -1
#CUDA_VISIBLE_DEVICES=7 python evaluation/compute_depth.py --config configs/NARF_GAN/SURREAL/20220303_surreal_triplane_centerFixed_gaussian_squareReg.yml --num_workers 2 --iteration -1


#CUDA_VISIBLE_DEVICES=7 python evaluation/compute_depth.py --config configs/NARF_GAN/SURREAL/20220302_surreal_triplane_centerFixed_gaussian_squareReg.yml --num_workers 2 --iteration -1 --truncation 0.4

CUDA_VISIBLE_DEVICES=7 python evaluation/compute_depth.py --config configs/NARF_GAN/SURREAL/20220225_surreal_triplane_centerFixed.yml --num_workers 2 --iteration -1 --truncation 0.4
CUDA_VISIBLE_DEVICES=7 python evaluation/compute_depth.py --config configs/NARF_GAN/SURREAL/20220227_surreal_triplane_centerFixed_cmu.yml --num_workers 2 --iteration -1 --truncation 0.4
CUDA_VISIBLE_DEVICES=7 python evaluation/compute_depth.py --config configs/NARF_GAN/SURREAL/20220228_surreal_triplane_centerFixed_gaussian.yml --num_workers 2 --iteration -1 --truncation 0.4
CUDA_VISIBLE_DEVICES=7 python evaluation/compute_depth.py --config configs/NARF_GAN/SURREAL/20220228_surreal_triplane_centerFixed_no_reg.yml --num_workers 2 --iteration -1 --truncation 0.4
CUDA_VISIBLE_DEVICES=7 python evaluation/compute_depth.py --config configs/NARF_VAE/SURREAL/20220228_surreal_triplane_centerFixed_mask.yml --num_workers 2 --iteration -1 --truncation 0.4