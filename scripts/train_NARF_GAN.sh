source ~/.zshrc_cuda111
pyenv shell anaconda3-4.3.1/envs/cuda111

#CUDA_VISIBLE_DEVICES=2 python train_NARF_GAN.py --config configs/NARF_GAN/THUman/20220204_thuman_triplane.yml --num_workers 2 --resume_latest
#python scripts/endless_run.py --command "CUDA_VISIBLE_DEVICES=1 python train_NARF_GAN.py --config configs/NARF_GAN/SURREAL/20220227_surreal_triplane_centerFixed_cmu.yml --num_workers 2 --resume_latest"
#python scripts/endless_run.py --command "CUDA_VISIBLE_DEVICES=2 python train_NARF_GAN.py --config configs/NARF_GAN/SURREAL/20220228_surreal_triplane_centerFixed_no_reg.yml --num_workers 2 --resume_latest"
#python scripts/endless_run.py --command "CUDA_VISIBLE_DEVICES=3 python train_NARF_GAN.py --config configs/NARF_GAN/AIST/20220301_aist_stylenerf_centerFixed.yml --num_workers 2 --resume_latest"
#python scripts/endless_run.py --command "CUDA_VISIBLE_DEVICES=4 python train_NARF_GAN.py --config configs/NARF_GAN/Wild/20220301_wild_stylenerf_centerFixed.yml --num_workers 2 --resume_latest"
#CUDA_VISIBLE_DEVICES=7 python train_NARF_GAN.py --config configs/NARF_GAN/SURREAL/20220228_surreal_triplane_centerFixed_no_bgLoss.yml --num_workers 2 --resume_latest
#CUDA_VISIBLE_DEVICES=1 python train_NARF_GAN.py --config configs/NARF_GAN/Wild/20220225_wild_triplane_pretrained_centerFixed.yml --num_workers 2 #--resume_latest
#CUDA_VISIBLE_DEVICES=5 python train_NARF_GAN.py --config configs/NARF_GAN/SURREAL/20220228_surreal_triplane_centerFixed_gaussian.yml --num_workers 2 --resume_latest
#python scripts/endless_run.py --command "CUDA_VISIBLE_DEVICES=0 python train_NARF_GAN.py --config configs/NARF_GAN/SURREAL/20220228_surreal_triplane_centerFixed_gaussian.yml --num_workers 2 --resume_latest"

python scripts/endless_run.py --command "CUDA_VISIBLE_DEVICES=0 python train_NARF_GAN.py --config configs/NARF_GAN/SURREAL/20220302_surreal_triplane_centerFixed_gaussian_squareReg.yml --num_workers 2 --resume_latest"

#python scripts/endless_run.py --command "CUDA_VISIBLE_DEVICES=1 python train_NARF_GAN.py --config configs/NARF_GAN/SURREAL/20220303_surreal_triplane_centerFixed_gaussian_squareReg.yml --num_workers 2 --resume_latest"

#CUDA_VISIBLE_DEVICES=2 python train_NARF_GAN.py --config configs/NARF_GAN/Wild/20220303_wild_triplane_pretrained_centerFixed_squareReg.yml --num_workers 2 #--resume_latest
