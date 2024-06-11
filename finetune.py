# Copyright (c) 2023 Michael Hu.
# This project is released under the MIT License.
# See the accompanying LICENSE file for details.

"""Runs LoRA fine-tuning on a single GPU using basic PyTorch"""
import argparse
import os
import gc
import time
import functools
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.utils.checkpoint import checkpoint

from llama3.core.model import Transformer
from llama3.core.tokenizer import Tokenizer
from llama3.core.dataset import FineTuneDataset, custom_collate_fn
from llama3.core.schedule import CosineDecayWithWarmupLRScheduler
from llama3.core.lora import get_lora_state_dict
from llama3.core.evaluation import masked_cross_entropy_loss, compute_evaluation_metrics
from llama3.utils.file_helper import read_json_file, save_json_file
from llama3.utils.logger import create_logger, pretty_print, log_metrics
from llama3.utils.profiler import get_system_metrics, aggregate_metrics
from llama3.utils.train_helper import set_seed, create_lora_model_and_load_checkpoint, create_optimizer


logger = None


def parse_runtime_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Fine-tune Llama3 script.')
    parser.add_argument(
        '--log-level',
        type=str,
        required=False,
        default='info',
        help='Log level, default info',
    )

    parser.add_argument(
        '--config-file',
        type=str,
        required=False,
        default='./configs/finetune.json',
        help='Path to the json configuration file',
    )

    return parser.parse_args()


def run_validation(model: Transformer, val_dataloader: DataLoader, num_val_steps: int, device: str, writer: SummaryWriter, global_step: int) -> None:
    model.eval()
    val_metrics_list = []
    num_batches = 0

    logger.info(f'Run {num_val_steps} validation steps')
    with torch.no_grad():
        for val_batch in val_dataloader:
            if num_batches >= num_val_steps:
                break

            for k in val_batch.keys():
                val_batch[k] = val_batch[k].to(device)

            val_outputs = model(val_batch['inputs'])
            val_loss = masked_cross_entropy_loss(logits=val_outputs, labels=val_batch['labels'], loss_mask=val_batch['loss_mask'])
            metrics = compute_evaluation_metrics(loss=val_loss, logits=val_outputs, labels=val_batch['labels'], loss_mask=val_batch['loss_mask'], prefix='Validation/')
            val_metrics_list.append(metrics)
            num_batches += 1

    torch.cuda.empty_cache()
    gc.collect()

    val_metrics = aggregate_metrics(val_metrics_list)
    print(f'Validation metrics at train step {global_step}')
    pretty_print(val_metrics)
    log_metrics(writer, global_step, val_metrics)

    model.train()

    assert model.training


def main():
    if not torch.version.cuda or not torch.cuda.is_bf16_supported():
        raise SystemError('This script requires CUDA.')

    FLAGS = parse_runtime_args()

    global logger
    logger = create_logger(FLAGS.log_level)
    # Load configuration for model and datasets
    config = read_json_file(FLAGS.config_file)
    model_config = config['model_config']
    lora_config = config['lora_config']
    set_seed(config['seed'])

    logger.info('Loading datasets...')
    tokenizer = Tokenizer(config['tokenizer_file'])
    model_config['vocab_size'] = tokenizer.vocab_size

    _collate_fn = functools.partial(
        custom_collate_fn,
        pad_id=tokenizer.eos_id,
        max_seq_len=model_config['max_seq_len'],
    )

    train_dataset = FineTuneDataset(data_sources=config['train']['dataset_sources'], max_seq_len=config['model_config']['max_seq_len'])
    val_dataset = FineTuneDataset(data_sources=config['validation']['dataset_sources'], max_seq_len=config['model_config']['max_seq_len'])
    logger.info(f'Train dataset: {train_dataset.get_metadata()}')
    logger.info(f'Validation dataset: {val_dataset.get_metadata()}')

    loader_kwargs = {'collate_fn': _collate_fn, 'shuffle': True, 'pin_memory': True, 'sampler': None}

    train_dataloader = DataLoader(train_dataset, batch_size=config['train']['micro_batch_size'], **loader_kwargs)
    val_dataloader = DataLoader(train_dataset, batch_size=config['validation']['batch_size'], **loader_kwargs)

    model = create_lora_model_and_load_checkpoint(model_config=model_config, lora_config=lora_config, ckpt_file=config['model_ckpt_file'])

    # enable activation checkpointing
    model.enable_activation_checkpointing(functools.partial(checkpoint, use_reentrant=False))

    runtime_device = 'cuda'
    compute_dtype = torch.bfloat16
    model = model.to(compute_dtype).to(device=runtime_device)

    optimizer = create_optimizer(model=model, lr=config['optimizer']['lr'], eps=config['optimizer']['eps'], weight_decay=config['optimizer']['weight_decay'], betas=config['optimizer']['betas'], fused=config['optimizer']['fused'])

    scheduler = CosineDecayWithWarmupLRScheduler(
        optimizer=optimizer,
        init_lr=config['scheduler']['min_lr'],
        max_lr=config['scheduler']['max_lr'],
        min_lr=config['scheduler']['min_lr'],
        warmup_steps=config['scheduler']['warmup_num_steps'],
        max_decay_steps=config['scheduler']['total_num_steps'],
    )

    log_dir = os.path.join(config['tensorboard']['output_path'], config['tensorboard']['job_name'])
    ckpt_dir = os.path.join(config['checkpoint']['output_path'], config['checkpoint']['job_name'])
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)

    # Training loop
    num_epochs = config['train']['num_epochs']
    logger.info(f'Starting to run {num_epochs} training epochs')
    elapsed_time = 0
    global_step = 0
    micro_step = 0
    num_samples = 0

    for epoch in range(num_epochs):
        logger.info(f'Epoch {epoch}')

        # Store metrics over gradient accumulation steps
        train_metrics_list = []
        for batch in train_dataloader:
            t0 = time.time()
            for k in batch.keys():
                batch[k] = batch[k].to(runtime_device)
            outputs = model(batch['inputs'])
            loss = masked_cross_entropy_loss(logits=outputs, labels=batch['labels'], loss_mask=batch['loss_mask'])
            scaled_loss = loss / config['train']['gradient_accumulation_steps']
            scaled_loss.backward()

            # Collect metrics for current micro batch
            metrics = compute_evaluation_metrics(loss=loss, logits=outputs, labels=batch['labels'], loss_mask=batch['loss_mask'], prefix='Train/')
            train_metrics_list.append(metrics)

            micro_step += 1
            num_samples += outputs.size(0)
            elapsed_time += time.time() - t0

            if micro_step % len(train_dataloader) == 0 or micro_step % config['train']['gradient_accumulation_steps'] == 0:
                t1 = time.time()
                grad_norm_clip = config['train']['gradient_clipping']

                if grad_norm_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_norm_clip)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                global_step += 1
                elapsed_time += time.time() - t1

                train_metrics = aggregate_metrics(train_metrics_list)
                # Reset for next step
                train_metrics_list = []

                # Add more metrics
                more_metrics = {
                    'Throughput/num_samples': num_samples,
                    'Throughput/samples_per_second': num_samples / elapsed_time if elapsed_time > 0 else 0,
                }
                train_metrics.update(more_metrics)

                # Log metrics
                if global_step % config['steps_per_print'] == 0:
                    train_metrics['Train/lr'] = optimizer.param_groups[0]['lr']

                    system_metrics = get_system_metrics()
                    train_metrics.update(system_metrics)
                    log_metrics(writer, global_step, train_metrics)

                    # Log to console
                    if global_step % 200 == 0:
                        logger.info(f'Train metrics at train step {global_step}')
                        pretty_print(train_metrics)

                # Save checkpoint
                if global_step % config['checkpoint']['save_interval'] == 0:
                    lora_state = get_lora_state_dict(model.state_dict(), train_bias=lora_config['train_bias'])
                    ckpt_file = os.path.join(ckpt_dir, f'lora_state_steps_{global_step}.pth')
                    logger.info(f'Saving LoRA model checkpoint to {ckpt_file!r}')
                    torch.save(lora_state, ckpt_file)

                    meta_file = os.path.join(ckpt_dir, 'params.json')
                    lora_file = os.path.join(ckpt_dir, 'lora.json')
                    if not os.path.exists(meta_file):
                        save_json_file(model_config, meta_file)
                        save_json_file(lora_config, lora_file)

                # Validation
                if config['validation']['enabled'] and global_step % config['validation']['interval'] == 0:
                    run_validation(model=model, val_dataloader=val_dataloader, num_val_steps=config['validation']['steps'], device=runtime_device, writer=writer, global_step=global_step)

    # Create final checkpoint
    lora_state = get_lora_state_dict(model.state_dict(), train_bias=lora_config['train_bias'])
    ckpt_file = os.path.join(ckpt_dir, f'lora_state_steps_{global_step}.pth')

    writer.close()


if __name__ == '__main__':
    main()
