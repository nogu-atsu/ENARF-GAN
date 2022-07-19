source ~/.zshrc_cuda111
pyenv shell anaconda3-4.3.1/envs/cuda111


CUDA_VISIBLE_DEVICES=5 python train_SSO.py --validation --config configs/SSO/NeuralActor/eval/20220714_NeuralActor_lan_enarf.yml --num_workers 2 --resume_latest
CUDA_VISIBLE_DEVICES=5 python train_SSO.py --validation --config configs/SSO/NeuralActor/eval/20220714_NeuralActor_lan_tpenarf.yml --num_workers 2 --resume_latest
#CUDA_VISIBLE_DEVICES=2 python train_SSO.py --validation --config configs/SSO/NeuralActor/eval/20220714_NeuralActor_lan_enarf.yml --num_workers 2 --resume_latest
#CUDA_VISIBLE_DEVICES=2 python train_SSO.py --validation --config configs/SSO/NeuralActor/eval/20220714_NeuralActor_lan_enarf.yml --num_workers 2 --resume_latest


#20220225_zju313_denarf_centerFixed
#20220225_zju313_enarf_centerFixed
#20220225_zju313_narf_centerFixed
#20220225_zju313_senarf_centerFixed
#20220225_zju315_denarf_centerFixed
#20220225_zju315_enarf_centerFixed
#20220225_zju315_narf_centerFixed
#20220225_zju315_senarf_centerFixed
#20220225_zju386_denarf_centerFixed
#20220225_zju386_enarf_centerFixed
#20220225_zju386_narf_centerFixed
#20220225_zju386_senarf_centerFixed
#20220226_zju313_enarf_wo_selector_centerFixed
#20220226_zju313_penarf_centerFixed
#20220226_zju313_tpenarf_centerFixed
#20220226_zju315_enarf_wo_selector_centerFixed
#20220226_zju315_penarf_centerFixed
#20220226_zju315_tpenarf_centerFixed
#20220226_zju386_enarf_wo_selector_centerFixed
#20220226_zju386_penarf_centerFixed
#20220226_zju386_tpenarf_centerFixed