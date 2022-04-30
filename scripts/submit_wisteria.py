import subprocess
import os

config_paths = [
    # "configs/SSO/ZJU/20220222_zju313_enarf.yml",
    # "configs/SSO/ZJU/20220222_zju313_tenarf.yml",
    # "configs/SSO/ZJU/20220222_zju313_denarf.yml",
    # "configs/SSO/ZJU/20220222_zju315_enarf.yml",
    # "configs/SSO/ZJU/20220222_zju315_tenarf.yml",
    # "configs/SSO/ZJU/20220222_zju315_denarf.yml",
    # "configs/SSO/ZJU/20220222_zju386_enarf.yml",
    # "configs/SSO/ZJU/20220222_zju386_tenarf.yml",
    # "configs/SSO/ZJU/20220222_zju386_denarf.yml",

    # "configs/SSO/ZJU/cvpr_exp/20220225_zju313_enarf_centerFixed.yml",
    # "configs/SSO/ZJU/cvpr_exp/20220225_zju315_enarf_centerFixed.yml",
    # "configs/SSO/ZJU/cvpr_exp/20220225_zju386_enarf_centerFixed.yml",
    # "configs/SSO/ZJU/cvpr_exp/20220225_zju313_denarf_centerFixed.yml",
    # "configs/SSO/ZJU/cvpr_exp/20220225_zju315_denarf_centerFixed.yml",
    # "configs/SSO/ZJU/cvpr_exp/20220225_zju386_denarf_centerFixed.yml",
    # "configs/SSO/ZJU/cvpr_exp/20220225_zju313_senarf_centerFixed.yml",
    # "configs/SSO/ZJU/cvpr_exp/20220225_zju315_senarf_centerFixed.yml",

    # "configs/SSO/ZJU/cvpr_exp/20220226_zju313_penarf_centerFixed.yml",
    # "configs/SSO/ZJU/cvpr_exp/20220226_zju315_penarf_centerFixed.yml",
    # "configs/SSO/ZJU/cvpr_exp/20220226_zju386_penarf_centerFixed.yml",
    # "configs/SSO/ZJU/cvpr_exp/20220226_zju313_tpenarf_centerFixed.yml",
    # "configs/SSO/ZJU/cvpr_exp/20220226_zju315_tpenarf_centerFixed.yml",
    # "configs/SSO/ZJU/cvpr_exp/20220226_zju386_tpenarf_centerFixed.yml",
    # "configs/SSO/ZJU/cvpr_exp/20220226_zju313_enarf_wo_selector_centerFixed.yml",
    # "configs/SSO/ZJU/cvpr_exp/20220226_zju315_enarf_wo_selector_centerFixed.yml",

    "configs/SSO/ZJU/cvpr_exp/20220303_zju313_enarf_centerFixed_noRay.yml",
    "configs/SSO/ZJU/cvpr_exp/20220303_zju315_enarf_centerFixed_noRay.yml",
    "configs/SSO/ZJU/cvpr_exp/20220303_zju386_enarf_centerFixed_noRay.yml",
    "configs/SSO/ZJU/cvpr_exp/20220303_zju313_tpenarf_centerFixed_noRay.yml",
    "configs/SSO/ZJU/cvpr_exp/20220303_zju315_tpenarf_centerFixed_noRay.yml",
    "configs/SSO/ZJU/cvpr_exp/20220303_zju386_tpenarf_centerFixed_noRay.yml",
    "configs/SSO/ZJU/cvpr_exp/20220303_zju313_senarf_centerFixed_noRay.yml",
    "configs/SSO/ZJU/cvpr_exp/20220303_zju315_senarf_centerFixed_noRay.yml",
]


occupy_node = True

if occupy_node:
    commands = f"""
#!/bin/sh
#------ pjsub option --------#
#PJM -L rscgrp=regular-a
#PJM -L node=1
#PJM -L elapse=24:00:00
#PJM -g gn53
#PJM -j
#------- Program execution -------#
source ~/.bash_profile
export CXX=g++
module load cuda/11.1
module load gcc/8.3.1

cd /work/gn53/k75008/D1/NARF-GAN-dev
CUDA_VISIBLE_DEVICES=0 python train_SSO.py --wisteria --config {config_paths[0]} --num_workers 2 &
CUDA_VISIBLE_DEVICES=1 python train_SSO.py --wisteria --config {config_paths[1]} --num_workers 2 &
CUDA_VISIBLE_DEVICES=2 python train_SSO.py --wisteria --config {config_paths[2]} --num_workers 2 &
CUDA_VISIBLE_DEVICES=3 python train_SSO.py --wisteria --config {config_paths[3]} --num_workers 2 &
CUDA_VISIBLE_DEVICES=4 python train_SSO.py --wisteria --config {config_paths[4]} --num_workers 2 &
CUDA_VISIBLE_DEVICES=5 python train_SSO.py --wisteria --config {config_paths[5]} --num_workers 2 &
CUDA_VISIBLE_DEVICES=6 python train_SSO.py --wisteria --config {config_paths[6]} --num_workers 2 &
CUDA_VISIBLE_DEVICES=7 python train_SSO.py --wisteria --config {config_paths[7]} --num_workers 2 &
sleep 86400
"""
    with open(f"submit_commands/node_job.sh", "w") as f:
        f.write(commands)

    subprocess.run(f"pjsub submit_commands/node_job.sh", shell=True)
    print("submitted")

else:
    for config in config_paths:
        exp_name = os.path.splitext(os.path.basename(config))[0]

        commands = f"""
#!/bin/sh
#------ pjsub option --------#
#PJM -L rscgrp=share
#PJM -L gpu=1
#PJM -L elapse=24:00:00
#PJM -g gn53
#PJM -j
#------- Program execution -------#
source ~/.bash_profile
export CXX=g++
module load cuda/11.1
module load gcc/8.3.1

cd /work/gn53/k75008/D1/NARF-GAN-dev
CUDA_VISIBLE_DEVICES=0 python train_SSO.py --wisteria --config {config} --num_workers 2 --resume_latest
"""
        with open(f"submit_commands/{exp_name}.sh", "w") as f:
            f.write(commands)

        subprocess.run(f"pjsub submit_commands/{exp_name}.sh", shell=True)
        print(exp_name, "submitted")
