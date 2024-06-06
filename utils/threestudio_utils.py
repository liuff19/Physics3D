## This implementation includes some utils from threestudio: https://github.com/threestudio-project/threestudio?tab=readme-ov-file

import gc
import os
from typing import Any

import tinycudann as tcnn
import torch
from packaging import version


def parse_version(ver: str):
    return version.parse(ver)


def cleanup():
    gc.collect()
    torch.cuda.empty_cache()
    tcnn.free_temporary_memory()
    
    
def get_rank():
    # SLURM_PROCID can be set even if SLURM is not managing the multiprocessing,
    # therefore LOCAL_RANK needs to be checked first
    rank_keys = ("RANK", "LOCAL_RANK", "SLURM_PROCID", "JSM_NAMESPACE_RANK")
    for key in rank_keys:
        rank = os.environ.get(key)
        if rank is not None:
            return int(rank)
    return 0


def get_device():
    return torch.device(f"cuda:{get_rank()}")


def _distributed_available():
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def barrier():
    if not _distributed_available():
        return
    else:
        torch.distributed.barrier()


def C(value: Any, epoch: int, global_step: int) -> float:
    if isinstance(value, int) or isinstance(value, float):
        pass
    else:
        value = config_to_primitive(value)
        if not isinstance(value, list):
            raise TypeError("Scalar specification only supports list, got", type(value))
        if len(value) == 3:
            value = [0] + value
        assert len(value) == 4
        start_step, start_value, end_value, end_step = value
        if isinstance(end_step, int):
            current_step = global_step
            value = start_value + (end_value - start_value) * max(
                min(1.0, (current_step - start_step) / (end_step - start_step)), 0.0
            )
        elif isinstance(end_step, float):
            current_step = epoch
            value = start_value + (end_value - start_value) * max(
                min(1.0, (current_step - start_step) / (end_step - start_step)), 0.0
            )
    return value


def shifted_expotional_decay(a, b, c, r):
    return a * torch.exp(-b * r) + c