import subprocess

config_paths = [
    # "20220225_zju313_denarf_centerFixed",
    # "20220225_zju313_enarf_centerFixed",
    # "20220225_zju313_narf_centerFixed",
    # "20220225_zju313_senarf_centerFixed",
    # "20220225_zju315_denarf_centerFixed",
    # "20220225_zju315_enarf_centerFixed",
    # "20220225_zju315_narf_centerFixed",
    # "20220225_zju315_senarf_centerFixed",
    # "20220225_zju386_denarf_centerFixed",
    # "20220225_zju386_enarf_centerFixed",
    # "20220225_zju386_narf_centerFixed",
    # "20220225_zju386_senarf_centerFixed",
    # "20220226_zju313_enarf_wo_selector_centerFixed",
    # "20220226_zju313_penarf_centerFixed",
    # "20220226_zju313_tpenarf_centerFixed",
    # "20220226_zju315_enarf_wo_selector_centerFixed",
    # "20220226_zju315_penarf_centerFixed",
    # "20220226_zju315_tpenarf_centerFixed",
    # "20220226_zju386_enarf_wo_selector_centerFixed",
    # "20220226_zju386_penarf_centerFixed",
    # "20220226_zju386_tpenarf_centerFixed",
    "20220303_zju313_enarf_centerFixed_noRay",
    "20220303_zju313_tpenarf_centerFixed_noRay",
    "20220303_zju313_senarf_centerFixed_noRay",
    "20220303_zju315_enarf_centerFixed_noRay",
    "20220303_zju315_tpenarf_centerFixed_noRay",
    "20220303_zju315_senarf_centerFixed_noRay",
    "20220303_zju386_enarf_centerFixed_noRay",
    "20220303_zju386_tpenarf_centerFixed_noRay",
    "20220303_zju386_senarf_centerFixed_noRay",
]


available_gpus = [5, 6, 7]
num_gpu = len(available_gpus)
for i, gpu_id in enumerate(available_gpus):
    configs = config_paths[i::num_gpu]
    command = ""
    for config in configs:
        command += f"CUDA_VISIBLE_DEVICES={gpu_id} python train_SSO.py --validation --config configs/SSO/ZJU/cvpr_exp/{config}.yml --num_workers 2 --resume_latest \n"
    # subprocess.run(command, shell=True)
    subprocess.Popen(command, shell=True)
