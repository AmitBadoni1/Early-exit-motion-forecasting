import os
import sys
import time
from typing import Any, Dict, List, Tuple, Union
from datetime import datetime
import argparse
import numpy as np
import faulthandler
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from loader import Loader
from utils.utils import AverageMeter, AverageMeterForDict


def parse_arguments() -> Any:
    """Arguments for running the baseline.

    Returns:
        parsed arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="val", type=str, help="Mode, train/val/test")
    parser.add_argument("--features_dir", required=True, default="", type=str, help="Path to the dataset")
    parser.add_argument("--train_batch_size", type=int, default=16, help="Training batch size")
    parser.add_argument("--val_batch_size", type=int, default=16, help="Val batch size")
    parser.add_argument("--use_cuda", action="store_true", help="Use CUDA for acceleration")
    parser.add_argument("--data_aug", action="store_true", help="Enable data augmentation")
    parser.add_argument("--adv_cfg_path", required=True, default="", type=str)
    parser.add_argument("--model_path", required=False, type=str, help="path to the saved model")
    return parser.parse_args()


def main():
    args = parse_arguments()
    print('Args: {}\n'.format(args))

    faulthandler.enable()

    if args.use_cuda and torch.cuda.is_available():
        device = torch.device("cuda", 0)
    else:
        device = torch.device('cpu')

    if not args.model_path.endswith(".tar"):
        assert False, "Model path error - '{}'".format(args.model_path)

    loader = Loader(args, device, is_ddp=False)
    print('[Resume] Loading state_dict from {}'.format(args.model_path))
    loader.set_resmue(args.model_path)
    (train_set, val_set), net, loss_fn, _, evaluator = loader.load()

    num_workers = 8
    pin_memory = device.type == "cuda"
    if os.name == "nt":
        num_workers = 0

    dl_val = DataLoader(val_set,
                        batch_size=args.val_batch_size,
                        shuffle=False,
                        num_workers=num_workers,
                        collate_fn=val_set.collate_fn,
                        drop_last=False,
                        pin_memory=pin_memory)

    net.eval()

    total_scenes = 0
    early_exit_scenes = 0
    full_pass_scenes = 0
    early_exit_time_ms = 0.0
    full_pass_time_ms = 0.0

    with torch.no_grad():
        # * Validation
        val_start = time.time()
        val_eval_meter = AverageMeterForDict()
        for i, data in enumerate(tqdm(dl_val)):
            data_in = net.pre_process(data)
            out = net(data_in)
            _ = loss_fn(out, data)
            post_out = net.post_process(out)

            eval_out = evaluator.evaluate(post_out, data)
            val_eval_meter.update(eval_out, n=data['BATCH_SIZE'])

            if "selective_exit_mask" in post_out:
                exit_mask = post_out["selective_exit_mask"]
                batch_total = int(exit_mask.numel())
                batch_early = int(exit_mask.sum().item())
                batch_full = batch_total - batch_early

                total_scenes += batch_total
                early_exit_scenes += batch_early
                full_pass_scenes += batch_full

                timing = post_out.get("selective_timing")
                if timing is not None:
                    early_exit_time_ms += timing["exit_ms_per_scene"] * batch_early
                    full_pass_time_ms += timing["full_ms_per_scene"] * batch_full
            else:
                total_scenes += int(data['BATCH_SIZE'])
                full_pass_scenes += int(data['BATCH_SIZE'])

        print('\nValidation set finish, cost {:.2f} secs'.format(time.time() - val_start))
        print('-- ' + val_eval_meter.get_info())
        print('-- total_scenes: {}'.format(total_scenes))
        print('-- early_exit_scenes: {}'.format(early_exit_scenes))
        print('-- full_pass_scenes: {}'.format(full_pass_scenes))
        if early_exit_scenes > 0:
            print('-- early_exit_avg_ms: {:.3f}'.format(early_exit_time_ms / early_exit_scenes))
        else:
            print('-- early_exit_avg_ms: n/a')
        if full_pass_scenes > 0 and full_pass_time_ms > 0.0:
            print('-- full_pass_avg_ms: {:.3f}'.format(full_pass_time_ms / full_pass_scenes))
        else:
            print('-- full_pass_avg_ms: n/a')

    print('\nExit...')


if __name__ == "__main__":
    main()
