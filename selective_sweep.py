import argparse
import os
import statistics
import time

import torch
from torch.utils.data import DataLoader

from loader import Loader
from utils.utils import AverageMeterForDict


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="val", type=str)
    parser.add_argument("--features_dir", required=True, type=str)
    parser.add_argument("--adv_cfg_path", required=True, type=str)
    parser.add_argument("--model_path", required=True, type=str)
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--val_batch_size", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--use_cuda", action="store_true")
    parser.add_argument("--data_aug", action="store_true")
    parser.add_argument("--thresholds", type=str, default="0.30,0.40,0.50,0.60,0.70,0.80,0.90")
    parser.add_argument("--num_warmup", type=int, default=20)
    parser.add_argument("--num_batches", type=int, default=100)
    return parser.parse_args()


def run_eval(net, dl_val, loss_fn, evaluator, threshold):
    net.eval()
    net.selective_threshold = threshold

    val_eval_meter = AverageMeterForDict()
    exit_rates = []
    with torch.no_grad():
        for data in dl_val:
            data_in = net.pre_process(data)
            out = net(data_in)
            _ = loss_fn(out, data)
            post_out = net.post_process(out)

            eval_out = evaluator.evaluate(post_out, data)
            val_eval_meter.update(eval_out, n=data["BATCH_SIZE"])

            if "selective_exit_rate" in post_out:
                exit_rates.append(float(post_out["selective_exit_rate"]))

    return val_eval_meter.metrics, statistics.mean(exit_rates) if exit_rates else 0.0


def run_benchmark(net, dl_val, threshold, device, num_warmup, num_batches):
    net.eval()
    net.selective_threshold = threshold
    iterator = iter(dl_val)

    with torch.no_grad():
        for _ in range(num_warmup):
            data = next(iterator)
            data_in = net.pre_process(data)
            _ = net(data_in)
        if device.type == "cuda":
            torch.cuda.synchronize(device)

        batch_times_ms = []
        exit_rates = []
        for _ in range(num_batches):
            data = next(iterator)
            data_in = net.pre_process(data)

            start = time.perf_counter()
            out = net(data_in)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            batch_times_ms.append(elapsed_ms)

            post_out = net.post_process(out)
            if "selective_exit_rate" in post_out:
                exit_rates.append(float(post_out["selective_exit_rate"]))

    return {
        "batch_ms_mean": statistics.mean(batch_times_ms),
        "batch_ms_std": statistics.pstdev(batch_times_ms),
        "sample_ms_mean": statistics.mean(batch_times_ms) / dl_val.batch_size,
        "exit_rate_bench": statistics.mean(exit_rates) if exit_rates else 0.0,
    }


def main():
    args = parse_args()
    thresholds = [float(x) for x in args.thresholds.split(",") if x.strip()]

    if args.use_cuda and torch.cuda.is_available():
        device = torch.device("cuda", 0)
    else:
        device = torch.device("cpu")

    loader = Loader(args, device, is_ddp=False)
    loader.set_resmue(args.model_path)
    (_, val_set), net, loss_fn, _, evaluator = loader.load()
    net.eval_output = "selective"

    num_workers = 0 if os.name == "nt" else 8
    pin_memory = device.type == "cuda"
    dl_val = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=val_set.collate_fn,
        drop_last=False,
        pin_memory=pin_memory,
    )

    print(f"model_path={args.model_path}")
    print(f"branch_config={args.adv_cfg_path}")
    print(f"device={device}")
    print("threshold,minADE_k,minFDE_k,MR_k,brier_fde_k,exit_rate_eval,batch_ms_mean,batch_ms_std,sample_ms_mean,exit_rate_bench")
    for threshold in thresholds:
        metrics, exit_rate_eval = run_eval(net, dl_val, loss_fn, evaluator, threshold)
        bench = run_benchmark(net, dl_val, threshold, device, args.num_warmup, args.num_batches)
        print(
            f"{threshold:.2f},"
            f"{metrics['minade_k'].avg:.3f},"
            f"{metrics['minfde_k'].avg:.3f},"
            f"{metrics['mr_k'].avg:.3f},"
            f"{metrics['brier_fde_k'].avg:.3f},"
            f"{exit_rate_eval:.3f},"
            f"{bench['batch_ms_mean']:.3f},"
            f"{bench['batch_ms_std']:.3f},"
            f"{bench['sample_ms_mean']:.3f},"
            f"{bench['exit_rate_bench']:.3f}"
        )


if __name__ == "__main__":
    main()
