import os
import subprocess
import sys

sys.path.append(".")
from libraries.config import yaml_config

config_paths = [
    # "configs/SSO/ZJU/cvpr_exp/20220224_zju313_narf.yml",
    # "configs/SSO/ZJU/cvpr_exp/20220224_zju315_narf.yml",
    # "configs/SSO/ZJU/cvpr_exp/20220224_zju386_narf.yml",
    # "configs/SSO/ZJU/cvpr_exp/20220224_zju315_senarf.yml",
    # "configs/SSO/ZJU/cvpr_exp/20220224_zju386_senarf.yml",

    # "configs/SSO/ZJU/cvpr_exp/20220225_zju313_narf_centerFixed.yml",
    # "configs/SSO/ZJU/cvpr_exp/20220225_zju315_narf_centerFixed.yml",
    # "configs/SSO/ZJU/cvpr_exp/20220225_zju386_narf_centerFixed.yml",
    # "configs/SSO/ZJU/cvpr_exp/20220225_zju386_senarf_centerFixed.yml",

    # "configs/SSO/ZJU/cvpr_exp/20220226_zju386_enarf_wo_selector_centerFixed.yml",

    # "configs/SSO/ZJU/cvpr_exp/20220310_zju313_enarf_wo_selector_centerFixed_noRay.yml",
    # "configs/SSO/ZJU/cvpr_exp/20220310_zju315_enarf_wo_selector_centerFixed_noRay.yml",
    # "configs/SSO/ZJU/cvpr_exp/20220310_zju386_enarf_wo_selector_centerFixed_noRay.yml",

    # "configs/SSO/NeuralActor/20220714_NeuralActor_lan_enarf.yml",
    "configs/SSO/NeuralActor/20220714_NeuralActor_lan_tpenarf.yml",
]

for config in config_paths:
    exp_name = os.path.splitext(os.path.basename(config))[0]

    conf = yaml_config(config, "configs/SSO/default.yml")
    commands = f"""!/bin/bash
#$-l rt_AG.small=1
#$-l h_rt=48:00:00
#$-j y
#$-cwd

source /etc/profile.d/modules.sh
module load cuda/11.1/11.1.1
module load gcc/9.3.0

PATH=$PATH:$HOME/.local/bin:$HOME/bin

export PATH

source ~/.bash_profile

cd /home/acc12675ut/D1/NARF-GAN-dev
pyenv local miniconda3-4.7.12/envs/NARFGAN

export TORCH_EXTENSIONS_DIR=/home/acc12675ut/data2/results/D1/NARF_GAN/result/{conf.out}

# while :
# do
CUDA_VISIBLE_DEVICES=0 python train_SSO.py --abci --config {config} --num_workers 2 --resume_latest
# done
"""
    os.makedirs("submit_commands", exist_ok=True)
    with open(f"submit_commands/exp_{exp_name}.sh", "w") as f:
        f.write(commands)

    subprocess.run(f"qsub -g gcc50556 submit_commands/exp_{exp_name}.sh", shell=True)
    print(exp_name, "submitted")
