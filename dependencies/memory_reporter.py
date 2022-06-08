from typing import Optional
import torch
import subprocess


def print_stats(self, verbose: bool = False, target_device: Optional[torch.device] = None) -> int:
    self.collect_tensor()
    self.get_stats()
    # header
    show_reuse = verbose
    for device, tensor_stats in self.device_tensor_stat.items():
        # By default, if the target_device is not specified,
        # print tensors on all devices
        if target_device is not None and device != target_device:
            continue
        total_mem = 0
        total_numel = 0
        for stat in tensor_stats:
            name, size, numel, mem = stat
            total_mem += mem
            total_numel += numel

    return total_mem


default_properies = (
    "timestamp",
    "gpu_name",
    # "gpu_uuid",
    "index",
    "memory.total",
    "memory.used",
    "memory.free",
    "utilization.gpu",
    "utilization.memory",
)


def get_gpu_properties(
        cmd_path="nvidia-smi",
        target_properties=default_properies,
        noheader=True,
        nounits=True
):
    """
    CUDA GPUのプロパティ情報取得

    Parameters
    ----------
    cmd_path : str
      コマンドラインから"nvidia-smi"を実行する際のパス
    target_properties : obj
      取得するプロパティ情報
      プロパティ情報の詳細は"nvidia-smi --help-query-gpu"で取得可能
    noheader : bool
      skip the first line with column headers
    nounits : bool
      don't print units for numerical values

    Returns
    -------
    gpu_properties : list
      gpuごとのproperty情報
    """

    # formatオプション定義
    format_option = "--format=csv"
    if noheader:
        format_option += ",noheader"
    if nounits:
        format_option += ",nounits"

    # コマンド生成
    cmd = '%s --query-gpu=%s %s' % (cmd_path, ','.join(target_properties), format_option)

    # サブプロセスでコマンド実行
    cmd_res = subprocess.check_output(cmd, shell=True)

    # コマンド実行結果をオブジェクトに変換
    gpu_lines = cmd_res.decode().split('\n')
    # リストの最後の要素に空行が入るため除去
    gpu_lines = [line.strip() for line in gpu_lines if line.strip() != '']

    # ", "ごとにプロパティ情報が入っているのでdictにして格納
    gpu_properties = [{k: v for k, v in zip(target_properties, line.split(', '))} for line in gpu_lines]

    return gpu_properties
