source ~/.bash_profile
export CXX=g++
module load cuda/11.1
module load gcc/8.3.1
#pyenv local miniconda3-4.7.12/envs/NARFGAN

CUDA_VISIBLE_DEVICES=0 python train_SSO.py --wisteria --config configs/SSO/ZJU/20220221_zju315_deform_triplane.yml --num_workers 2 --resume_latest
#CUDA_VISIBLE_DEVICES=3 python train_SSO.py --config configs/SSO/ZJU/20220215_zju313_triplane_const_tri.yml --num_workers 2 --resume_latest

#CUDA_VISIBLE_DEVICES=4 python train_SSO.py --config configs/SSO/ZJU/20220215_zju315_const_triplane.yml --num_workers 2 --resume_latest


#CUDA_VISIBLE_DEVICES=0 python train_SSO.py --config configs/SSO/ZJU/20220220_zju315_narf.yml --num_workers 2 --resume_latest
#CUDA_VISIBLE_DEVICES=1 python train_SSO.py --config configs/SSO/ZJU/20220220_zju315_tnarf.yml --num_workers 2 --resume_latest
#CUDA_VISIBLE_DEVICES=2 python train_SSO.py --config configs/SSO/ZJU/20220220_zju315_dnarf.yml --num_workers 2 --resume_latest

