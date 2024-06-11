# Copyright (c) 2023 Michael Hu.
# This project is released under the MIT License.
# See the accompanying LICENSE file for details.

"""Functions to collect system metrics like GPU, CPU, Disk, Network"""

import subprocess
import psutil
import torch
import torch.distributed as dist
from typing import List


def get_system_metrics(gpu: bool = True, cpu: bool = False, disk: bool = False, network: bool = False) -> dict:
    """Collect system metrics like GPU, CPU, Disk, and Network"""
    metrics = {}
    if gpu:
        gpu_metrics = get_gpu_metrics()
        metrics.update(gpu_metrics)
    if cpu:
        cpu_metrics = get_cpu_ram_metrics()
        metrics.update(cpu_metrics)
    if disk:
        disk_metrics = get_disk_metrics()
        metrics.update(disk_metrics)
    if network:
        network_metrics = get_network_metrics()
        metrics.update(network_metrics)

    return metrics


def get_gpu_metrics():

    metrics = {}
    allocated = torch.cuda.memory_allocated()
    max_allocated = torch.cuda.max_memory_allocated()
    metrics['GPU/allocated_RAM_MB'] = round(float(allocated) / (1024**2), 4)
    metrics['GPU/max_allocated_RAM_MB'] = round(float(max_allocated) / (1024**2), 4)

    try:
        # Run nvidia-smi command and parse the output
        result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,temperature.gpu,memory.free,memory.total,power.draw,fan.speed', '--format=csv,nounits,noheader'], stdout=subprocess.PIPE, text=True)
        utilization, temperature, memory_free, memory_total, power_draw, fan_speed = result.stdout.strip().split(',')
        metrics['GPU/utilization'] = float(utilization)
        metrics['GPU/temperature'] = float(temperature)
        metrics['GPU/power_draw'] = float(power_draw)
        metrics['GPU/free_RAM_MB'] = round(float(memory_free), 4)
        # metrics['GPU/total_RAM_MB'] = round(float(memory_total), 4)
    except Exception as e:
        print(f'Error: {str(e)}')

    return metrics


def get_cpu_ram_metrics():
    cpu_usage = psutil.cpu_percent(interval=1)
    ram_usage = psutil.virtual_memory().percent
    return {'CPU/cpu_usage': cpu_usage, 'RAM/ram_usage': ram_usage}


def get_disk_metrics():
    disk_usage = psutil.disk_usage('/').percent
    disk_io = psutil.disk_io_counters()
    return {'Disk/disk_usage': disk_usage, 'Disk/disk_read_bytes': disk_io.read_bytes, 'Disk/disk_write_bytes': disk_io.write_bytes}


def get_network_metrics():
    net_io = psutil.net_io_counters()
    return {'Network/bytes_sent': net_io.bytes_sent, 'Network/bytes_recv': net_io.bytes_recv, 'Network/packets_sent': net_io.packets_sent, 'Network/packets_recv': net_io.packets_recv}


def aggregate_metrics(metrics_list: List[dict]) -> dict:
    """Aggregate a list of metrics collected over N steps."""
    agg_metrics = {}
    for metrics_dict in metrics_list:
        for key, value in metrics_dict.items():
            if key in agg_metrics:
                agg_metrics[key] += value
            else:
                agg_metrics[key] = value

    # Calculate the average of the metrics
    agg_metrics = {k: v / len(metrics_list) for k, v in agg_metrics.items()}
    return agg_metrics


def aggregate_metrics_across_gpus(metrics: dict, average: bool = True) -> dict:
    """Aggregate metrics across all GPUs."""

    if not dist.is_initialized():
        raise RuntimeError('Distributed package not initialized')

    # Synchronize all processes before starting aggregation
    dist.barrier()

    for key in metrics:
        tensor = torch.tensor(metrics[key]).to('cuda')
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        metrics[key] = tensor.item()

    # Synchronize all processes after aggregation
    dist.barrier()

    num_gpus = dist.get_world_size() if average else 1
    aggregated_metrics = {k: v / num_gpus for k, v in metrics.items()}
    return aggregated_metrics
