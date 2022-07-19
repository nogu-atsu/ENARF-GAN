source ~/.zshrc_cuda111
pyenv shell anaconda3-4.3.1/envs/cuda111

#CUDA_VISIBLE_DEVICES=5 python train_SSO.py --config configs/SSO/ZJU/cvpr_exp/20220303_zju386_senarf_centerFixed_noRay.yml --num_workers 2 --resume_latest
#CUDA_VISIBLE_DEVICES=6 python train_SSO.py --config configs/SSO/ZJU/cvpr_exp/20220301_zju315_enarf_centerFixed_scale1.yml --num_workers 2 --resume_latest &
#CUDA_VISIBLE_DEVICES=7 python train_SSO.py --config configs/SSO/ZJU/cvpr_exp/20220301_zju386_enarf_centerFixed_scale1.yml --num_workers 2 --resume_latest


#CUDA_VISIBLE_DEVICES=0 python train_SSO.py --config configs/SSO/ZJU/20220220_zju315_narf.yml --num_workers 2 --resume_latest
#CUDA_VISIBLE_DEVICES=1 python train_SSO.py --config configs/SSO/ZJU/20220220_zju315_tnarf.yml --num_workers 2 --resume_latest
#CUDA_VISIBLE_DEVICES=2 python train_SSO.py --config configs/SSO/ZJU/20220220_zju315_dnarf.yml --num_workers 2 --resume_latest

#CUDA_VISIBLE_DEVICES=3 python train_SSO.py --config configs/SSO/NeuralActor/20220714_NeuralActor_marc_enarf.yml --num_workers 2 --resume_latest
#CUDA_VISIBLE_DEVICES=3 python train_SSO.py --config configs/SSO/NeuralActor/20220714_NeuralActor_marc_tpenarf.yml --num_workers 2 --resume_latest

CUDA_VISIBLE_DEVICES=0 python train_SSO.py --config configs/SSO/NeuralActor/20220715_NeuralActor_lan_tpenarf.yml --num_workers 2 --resume_latest