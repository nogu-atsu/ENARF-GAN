source ~/.zshrc_cuda111
pyenv shell anaconda3-4.3.1/envs/cuda111

#CUDA_VISIBLE_DEVICES=1 python evaluation/compute_fid.py --config configs/NARF_GAN/ZJU/20220101_zju_aligned_l2_stylenerf_renderer_cropbg.yml --num_workers 2 --iteration 300000

#CUDA_VISIBLE_DEVICES=1 python evaluation/compute_fid.py --config configs/NARF_GAN/THUman/20211227_THUmanPrior.yml --num_workers 2 --iteration 400000

#CUDA_VISIBLE_DEVICES=2 python evaluation/compute_fid.py --config configs/NARF_GAN/THUman/20211227_THUman100CMU100.yml --num_workers 2 --iteration 300000




#CUDA_VISIBLE_DEVICES=6 python evaluation/compute_PCK.py --config configs/NARF_GAN/SURREAL/20220225_surreal_triplane_centerFixed.yml --num_workers 2 --iteration -1
#CUDA_VISIBLE_DEVICES=6 python evaluation/compute_PCK.py --config configs/NARF_GAN/SURREAL/20220227_surreal_triplane_centerFixed_cmu.yml --num_workers 2 --iteration -1
#CUDA_VISIBLE_DEVICES=6 python evaluation/compute_PCK.py --config configs/NARF_GAN/SURREAL/20220228_surreal_triplane_centerFixed_gaussian.yml --num_workers 2 --iteration -1
#CUDA_VISIBLE_DEVICES=6 python evaluation/compute_PCK.py --config configs/NARF_GAN/SURREAL/20220226_surreal_stylenerf_centerFixed.yml --num_workers 2 --iteration -1
#CUDA_VISIBLE_DEVICES=6 python evaluation/compute_PCK.py --config configs/NARF_GAN/SURREAL/20220228_surreal_triplane_centerFixed_no_reg.yml --num_workers 2 --iteration -1
#CUDA_VISIBLE_DEVICES=6 python evaluation/compute_PCK.py --config configs/NARF_VAE/SURREAL/20220228_surreal_triplane_centerFixed_mask.yml --num_workers 2 --iteration -1
#CUDA_VISIBLE_DEVICES=6 python evaluation/compute_PCK.py --config configs/NARF_GAN/SURREAL/20220302_surreal_triplane_centerFixed_gaussian_squareReg.yml --num_workers 2 --iteration -1
#CUDA_VISIBLE_DEVICES=6 python evaluation/compute_PCK.py --config configs/NARF_GAN/SURREAL/20220303_surreal_triplane_centerFixed_gaussian_squareReg.yml --num_workers 2 --iteration -1

#CUDA_VISIBLE_DEVICES=4 python evaluation/compute_PCK.py --config configs/NARF_GAN/SURREAL/20220228_surreal_triplane_centerFixed_no_reg.yml --num_workers 2 --iteration -1  --truncation 0.4
#CUDA_VISIBLE_DEVICES=4 python evaluation/compute_PCK.py --config configs/NARF_GAN/SURREAL/20220225_surreal_triplane_centerFixed.yml --num_workers 2 --iteration -1 --truncation 0.4
#CUDA_VISIBLE_DEVICES=4 python evaluation/compute_PCK.py --config configs/NARF_GAN/SURREAL/20220227_surreal_triplane_centerFixed_cmu.yml --num_workers 2 --iteration -1 --truncation 0.4
#CUDA_VISIBLE_DEVICES=4 python evaluation/compute_PCK.py --config configs/NARF_GAN/SURREAL/20220228_surreal_triplane_centerFixed_gaussian.yml --num_workers 2 --iteration -1 --truncation 0.4
#CUDA_VISIBLE_DEVICES=4 python evaluation/compute_PCK.py --config configs/NARF_GAN/SURREAL/20220302_surreal_triplane_centerFixed_gaussian_squareReg.yml --num_workers 2 --iteration -1  --truncation 0.4


#CUDA_VISIBLE_DEVICES=3 python evaluation/compute_PCK.py --config configs/NARF_GAN/SURREAL/20220228_surreal_triplane_centerFixed_no_reg.yml --num_workers 2 --iteration -1  --truncation 0.4 &
#CUDA_VISIBLE_DEVICES=4 python evaluation/compute_PCK_fixed.py --config configs/NARF_GAN/SURREAL/20220228_surreal_triplane_centerFixed_no_reg.yml --num_workers 2 --iteration -1  --truncation 1.0
#CUDA_VISIBLE_DEVICES=4 python evaluation/compute_PCK_fixed.py --config configs/NARF_GAN/SURREAL/20220225_surreal_triplane_centerFixed.yml --num_workers 2 --iteration -1 --truncation 1.0
#CUDA_VISIBLE_DEVICES=4 python evaluation/compute_PCK_fixed.py --config configs/NARF_GAN/SURREAL/20220227_surreal_triplane_centerFixed_cmu.yml --num_workers 2 --iteration -1 --truncation 1.0


#CUDA_VISIBLE_DEVICES=2 python evaluation/pck_by_stanford.py --config configs/NARF_GAN/SURREAL/20220227_surreal_triplane_centerFixed_cmu.yml --num_workers 2 --iteration 200000 --truncation 1.0  # 0.9629328857849244
#CUDA_VISIBLE_DEVICES=2 python evaluation/pck_by_stanford.py --config configs/NARF_GAN/SURREAL/20220225_surreal_triplane_centerFixed.yml --num_workers 2 --iteration 200000 --truncation 1.0  # 0.9501936511383792
#CUDA_VISIBLE_DEVICES=2 python evaluation/pck_by_stanford.py --config configs/NARF_GAN/SURREAL/20220228_surreal_triplane_centerFixed_no_reg.yml --num_workers 2 --iteration 200000 --truncation 1.0 #
#CUDA_VISIBLE_DEVICES=2 python evaluation/pck_by_stanford.py --config configs/NARF_VAE/SURREAL/20220228_surreal_triplane_centerFixed_mask.yml --num_workers 2 --iteration 200000 --truncation 1.0
CUDA_VISIBLE_DEVICES=3 python evaluation/pck_by_stanford.py --config configs/NARF_GAN/SURREAL/20220228_surreal_triplane_centerFixed_gaussian.yml --num_workers 2 --iteration 300000 --truncation 1.0
CUDA_VISIBLE_DEVICES=3 python evaluation/pck_by_stanford.py --config configs/NARF_GAN/SURREAL/20220228_surreal_triplane_centerFixed_gaussian.yml --num_workers 2 --iteration 300000 --truncation 0.4
#CUDA_VISIBLE_DEVICES=2 python evaluation/pck_by_stanford.py --config configs/NARF_GAN/SURREAL/20220226_surreal_stylenerf_centerFixed.yml --num_workers 2 --iteration 200000 --truncation 1.0