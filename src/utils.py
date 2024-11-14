# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
#
import os
import re
import sys
import pickle
import random
import socket
import inspect
import getpass
import argparse
import itertools
import subprocess
from logging import getLogger
import torch
from torch import optim
import time
import numpy as np
from logger import create_logger
from math import ceil


FALSY_STRINGS = {'off', 'false', '0'}
TRUTHY_STRINGS = {'on', 'true', '1'}

DUMP_PATH = '/checkpoint/%s/dumped' if 'private' in __file__ else '/mnt/vol/gfsai-east/ai-group/users/%s/dumped'
DUMP_PATH = DUMP_PATH % getpass.getuser()


logger = getLogger()

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """
    Save the training model
    """
    torch.save(state, filename)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def log_metrics(args, best_prec1, train_time, save_dir, experiment_id, train_losses, val_losses, train_accuracies, val_accuracies, final_epsilon):
    """
    Logs the training metrics to a file.
    """
    # Save metrics to a log file
    log_file_path = os.path.join(save_dir, "metrics.log")
    with open(log_file_path, "w") as log_file:
        log_file.write(f"Experiment ID: {experiment_id}\n")
        log_file.write(f"Training epochs: {args.epochs}\n")
        log_file.write(f"Batch size: {args.batch_size}\n")
        log_file.write(f"Learning rate: {args.lr}\n")
        log_file.write(f"Target epsilon: {args.target_epsilon}\n")
        log_file.write(f"Final epsilon: {final_epsilon:.2f}\n")
        log_file.write(f"Total training time: {train_time:.2f} seconds\n")
        log_file.write(f"Best accuracy: {best_prec1:.2f}%\n")
        log_file.write(f"Model saved at: {os.path.abspath(args.save_dir)}\n")
        log_file.write(f"Training losses: {train_losses}\n")
        log_file.write(f"Validation losses: {val_losses}\n")
        log_file.write(f"Training accuracies: {train_accuracies}\n")
        log_file.write(f"Validation accuracies: {val_accuracies}\n")

    print(f"Metrics saved in log file: {log_file_path}")











def init_distributed_mode(params):
    """
    Handle single and multi-GPU / multi-node / SLURM jobs.
    Initialize the following variables:
        - n_nodes
        - node_id
        - local_rank
        - global_rank
        - world_size
    """
    params.is_slurm_job = 'SLURM_JOB_ID' in os.environ and not params.debug_slurm
    logger.info("SLURM job: %s" % str(params.is_slurm_job))

    # SLURM job
    if params.is_slurm_job:

        assert params.local_rank == -1   # on the cluster, this is handled by SLURM

        SLURM_VARIABLES = [
            'SLURM_JOB_ID',
            'SLURM_JOB_NODELIST', 'SLURM_JOB_NUM_NODES', 'SLURM_NTASKS', 'SLURM_TASKS_PER_NODE',
            'SLURM_MEM_PER_NODE', 'SLURM_MEM_PER_CPU',
            'SLURM_NODEID', 'SLURM_PROCID', 'SLURM_LOCALID', 'SLURM_TASK_PID'
        ]

        PREFIX = "%i - " % int(os.environ['SLURM_PROCID'])
        for name in SLURM_VARIABLES:
            value = os.environ.get(name, None)
            logger.info(PREFIX + "%s: %s" % (name, str(value)))

        # # job ID
        params.job_id = os.environ['SLURM_JOB_ID']

        # number of nodes / node ID
        params.n_nodes = int(os.environ['SLURM_JOB_NUM_NODES'])
        params.node_id = int(os.environ['SLURM_NODEID'])

        # local rank on the current node / global rank
        params.local_rank = int(os.environ['SLURM_LOCALID'])
        params.global_rank = int(os.environ['SLURM_PROCID'])

        # number of processes / GPUs per node
        params.world_size = int(os.environ['SLURM_NTASKS'])
        params.n_gpu_per_node = params.world_size // params.n_nodes

        # define master address and master port
        hostnames = subprocess.check_output(['scontrol', 'show', 'hostnames', os.environ['SLURM_JOB_NODELIST']])
        params.master_addr = hostnames.split()[0].decode('utf-8')
        assert 10001 <= params.master_port <= 20000 or params.world_size == 1
        logger.info(PREFIX + "Master address: %s" % params.master_addr)
        logger.info(PREFIX + "Master port   : %i" % params.master_port)

        # set environment variables for 'env://'
        os.environ['MASTER_ADDR'] = params.master_addr
        os.environ['MASTER_PORT'] = str(params.master_port)
        os.environ['WORLD_SIZE'] = str(params.world_size)
        os.environ['RANK'] = str(params.global_rank)

    # multi-GPU job (local or multi-node) - jobs started with torch.distributed.launch
    elif params.local_rank != -1:

        assert params.master_port == -1

        # read environment variables
        params.global_rank = int(os.environ['RANK'])
        params.world_size = int(os.environ['WORLD_SIZE'])
        params.n_gpu_per_node = int(os.environ['NGPU'])

        # number of nodes / node ID
        params.n_nodes = params.world_size // params.n_gpu_per_node
        params.node_id = params.global_rank // params.n_gpu_per_node

    # local job (single GPU)
    else:
        assert params.local_rank == -1
        assert params.master_port == -1
        params.n_nodes = 1
        params.node_id = 0
        params.local_rank = 0
        params.global_rank = 0
        params.world_size = 1
        params.n_gpu_per_node = 1

    # sanity checks
    assert params.n_nodes >= 1
    assert 0 <= params.node_id < params.n_nodes
    assert 0 <= params.local_rank <= params.global_rank < params.world_size
    assert params.world_size == params.n_nodes * params.n_gpu_per_node

    # define whether this is the master process / if we are in distributed mode
    params.is_master = params.node_id == 0 and params.local_rank == 0
    params.multi_node = params.n_nodes > 1
    params.multi_gpu = params.world_size > 1

    # summary
    PREFIX = "%i - " % params.global_rank
    logger.info(PREFIX + "Number of nodes: %i" % params.n_nodes)
    logger.info(PREFIX + "Node ID        : %i" % params.node_id)
    logger.info(PREFIX + "Local rank     : %i" % params.local_rank)
    logger.info(PREFIX + "Global rank    : %i" % params.global_rank)
    logger.info(PREFIX + "World size     : %i" % params.world_size)
    logger.info(PREFIX + "GPUs per node  : %i" % params.n_gpu_per_node)
    logger.info(PREFIX + "Master         : %s" % str(params.is_master))
    logger.info(PREFIX + "Multi-node     : %s" % str(params.multi_node))
    logger.info(PREFIX + "Multi-GPU      : %s" % str(params.multi_gpu))
    logger.info(PREFIX + "Hostname       : %s" % socket.gethostname())

    # set GPU device
    torch.cuda.set_device(params.local_rank)

    # initialize multi-GPU
    if params.multi_gpu:

        # http://pytorch.apachecn.org/en/0.3.0/distributed.html#environment-variable-initialization
        # 'env://' will read these environment variables:
        # MASTER_PORT - required; has to be a free port on machine with rank 0
        # MASTER_ADDR - required (except for rank 0); address of rank 0 node
        # WORLD_SIZE - required; can be set either here, or in a call to init function
        # RANK - required; can be set either here, or in a call to init function

        logger.info("Initializing PyTorch distributed ...")
        torch.distributed.init_process_group(
            init_method='env://',
            backend='nccl',
        )


def initialize_exp(params):
    """
    Initialize the experience:
    - dump parameters
    - create a logger
    """
    # dump parameters
    get_dump_path(params)
    pickle.dump(params, open(os.path.join(params.dump_path, 'params.pkl'), 'wb'))

    # get running command
    command = ["python", sys.argv[0]]
    for x in sys.argv[1:]:
        if x.startswith('--'):
            assert '"' not in x and "'" not in x
            command.append(x)
        else:
            assert "'" not in x
            if re.match('^[a-zA-Z0-9_]+$', x):
                command.append("%s" % x)
            else:
                command.append("'%s'" % x)
    command = ' '.join(command)
    params.command = command + ' --exp_id "%s"' % params.exp_id

    # check experiment name
    assert len(params.exp_name.strip()) > 0

    # create a logger
    logger = create_logger(os.path.join(params.dump_path, 'train.log'), rank=getattr(params, 'global_rank', 0))
    logger.info("============ Initialized logger ============")
    logger.info("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(params)).items())))
    logger.info("The experiment will be stored in %s\n" % params.dump_path)
    logger.info("Running command: %s" % command)
    logger.info("")
    return logger

def bool_flag(s):
    """
    Parse boolean arguments from the command line.
    """
    if s.lower() in FALSY_STRINGS:
        return False
    elif s.lower() in TRUTHY_STRINGS:
        return True
    else:
        raise argparse.ArgumentTypeError("invalid value for a boolean flag")

def accuracy(preds, labels):
    return (preds == labels).mean()

def get_noise_from_bs(B, ref_noise, ref_B,beta=1):
    """
    output the noise necessary to keep our "physical constant" eta constant.
    """
    return(ref_noise/((ref_B/B)*beta))

def get_epochs_from_bs(B, ref_nb_steps, size_dataset):
    """
    output the approximate number of epochs necessary to keep our "physical constant" eta constant.
    We use a ceil, but please not that the last epoch will stop when we reach 'ref_nb_steps' steps.
    """
    return(ceil(ref_nb_steps*B/size_dataset))

def print_params(net):
    n_params = sum([p.numel() for p in net.parameters() if p.requires_grad])
    if n_params > 1e9:
        print(f"{n_params // 1e9}B parameters")
    elif n_params > 1e6:
        print(f"{n_params // 1e6}M parameters")
    elif n_params > 1e3:
        print(f"{n_params // 1e3}K parameters")
    else:
        print(f"{n_params} parameters")

def trainable_parameters(module):
    """
    Recursively iterates over all parameters, returning those that
    are trainable (ie they want a grad).
    """
    yield from (
        (p_name, p) for (p_name, p) in module.named_parameters() if p.requires_grad
    )