source ~/.bash_profile
module load cuda/11.1/11.1.1
module load gcc/9.3.0
pyenv local miniconda3-4.7.12/envs/NARFGAN

CUDA_VISIBLE_DEVICES=0 python train_SSO.py --abci --config configs/SSO/ZJU/20220224_zju386_narf.yml --num_workers 2 --resume_latest
#CUDA_VISIBLE_DEVICES=3 python train_SSO.py --config configs/SSO/ZJU/20220215_zju313_triplane_const_tri.yml --num_workers 2 --resume_latest

#CUDA_VISIBLE_DEVICES=4 python train_SSO.py --config configs/SSO/ZJU/20220215_zju315_const_triplane.yml --num_workers 2 --resume_latest


#CUDA_VISIBLE_DEVICES=0 python train_SSO.py --config configs/SSO/ZJU/20220220_zju315_narf.yml --num_workers 2 --resume_latest
#CUDA_VISIBLE_DEVICES=1 python train_SSO.py --config configs/SSO/ZJU/20220220_zju315_tnarf.yml --num_workers 2 --resume_latest
#CUDA_VISIBLE_DEVICES=2 python train_SSO.py --config configs/SSO/ZJU/20220220_zju315_dnarf.yml --num_workers 2 --resume_latest

