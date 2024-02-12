import torch
import os
import torch.distributed as dist

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False

    if not dist.is_initialized():
        return False

    return True


def get_rank():

    if not is_dist_avail_and_initialized():
        return 0

    return dist.get_rank()

def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1

    return dist.get_world_size()

def is_main_process():

    return get_rank() == 0

def init_distributed():
    if not torch.cuda.is_available() or torch.cuda.device_count() <= 1:
        return

    # Initializes the distributed backend which will take care of synchronizing nodes/GPUs
    dist_url = "env://" # default

    # only works with torch.distributed.launch // torch.run
    rank = int(os.environ["RANK"])
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])
    dist.init_process_group(
            backend="nccl",
            init_method=dist_url,
            world_size=world_size,
            rank=rank)

    # this will make all .cuda() calls work properly
    torch.cuda.set_device(local_rank)

    # synchronizes all the threads to reach this point before moving on
    dist.barrier()