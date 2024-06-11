# Copyright (c) 2023 Michael Hu.
# This project is released under the MIT License.
# See the accompanying LICENSE file for details.

"""Runs LoRA fine-tuning using DeepSpeed"""
import argparse
import os
import gc
import time
import functools
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
import deepspeed

from llama3.core.model import ModelArgs, Transformer
from llama3.core.tokenizer import Tokenizer
from llama3.core.dataset import FineTuneDataset, custom_collate_fn
from llama3.core.lora import replace_with_lora_layers, mark_only_lora_as_trainable, get_lora_state_dict
from llama3.core.evaluation import masked_cross_entropy_loss, compute_evaluation_metrics
from llama3.utils.file_helper import read_json_file, save_json_file
from llama3.utils.logger import create_logger, pretty_print, log_metrics
from llama3.utils.profiler import get_system_metrics, aggregate_metrics, aggregate_metrics_across_gpus
from llama3.utils.train_helper import set_seed, create_lora_model_and_load_checkpoint, load_model_checkpoint, get_model_parameters_for_optimizer


logger = None


def parse_runtime_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='DeepSpeed Fine-tune Llama3 script.')

    parser.add_argument('--local_rank', type=int, default=-1, help='local rank passed from distributed launcher')
    parser.add_argument('--log-level', type=str, required=False, default='info', help='Log level, default info')
    parser.add_argument(
        '--config-file',
        type=str,
        required=False,
        default='./configs/ds_finetune.json',
        help='Path to the json configuration file',
    )

    # Include DeepSpeed configuration arguments
    parser = deepspeed.add_config_arguments(parser)

    return parser.parse_args()


def run_validation(model_engine: deepspeed.DeepSpeedEngine, val_dataloader: DataLoader, num_val_steps: int) -> None:
    model_engine.eval()
    val_metrics_list = []
    num_batches = 0

    logger.info(f'Run {num_val_steps} validation steps')
    with torch.no_grad():
        for val_batch in val_dataloader:
            if num_batches >= num_val_steps:
                break

            for k in val_batch.keys():
                val_batch[k] = val_batch[k].to(model_engine.local_rank)

            val_outputs = model_engine(val_batch['inputs'])
            val_loss = masked_cross_entropy_loss(logits=val_outputs, labels=val_batch['labels'], loss_mask=val_batch['loss_mask'])
            metrics = compute_evaluation_metrics(loss=val_loss, logits=val_outputs, labels=val_batch['labels'], loss_mask=val_batch['loss_mask'], prefix='Validation/GlobalSteps/')
            val_metrics_list.append(metrics)
            num_batches += 1

    torch.cuda.empty_cache()
    gc.collect()

    local_val_metrics = aggregate_metrics(val_metrics_list)
    val_metrics = aggregate_metrics_across_gpus(local_val_metrics)
    dist.barrier()

    if dist.get_rank() == 0:
        summary_events = [(k, v, model_engine.global_steps) for k, v in val_metrics.items()]
        model_engine.monitor.write_events(summary_events)

        logger.info(f'Validation metrics at global train steps {model_engine.global_steps}')
        pretty_print(val_metrics)

    model_engine.train()
    assert model_engine.training


def main():

    # Initialize distributed training
    deepspeed.init_distributed()

    FLAGS = parse_runtime_args()

    global logger
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    logger = create_logger(level=FLAGS.log_level, rank=rank)

    # Load configuration for model, datasets, and deepspeed
    config = read_json_file(FLAGS.config_file)
    model_config = config['model_config']
    lora_config = config['lora_config']
    ds_config = config['deepspeed_config']
    set_seed(config['seed'])

    logger.info('Loading datasets...')

    tokenizer = Tokenizer(config['tokenizer_file'])
    model_config['vocab_size'] = tokenizer.vocab_size

    _collate_fn = functools.partial(
        custom_collate_fn,
        pad_id=tokenizer.eos_id,
        max_seq_len=model_config['max_seq_len'],
    )

    train_dataset = FineTuneDataset(data_sources=config['train']['dataset_sources'], max_seq_len=model_config['max_seq_len'])
    val_dataset = FineTuneDataset(data_sources=config['validation']['dataset_sources'], max_seq_len=model_config['max_seq_len'])
    logger.info(f'Train dataset: {train_dataset.get_metadata()}')
    logger.info(f'Validation dataset: {val_dataset.get_metadata()}')

    sampler_kwargs = {'num_replicas': world_size, 'rank': rank, 'seed': config['seed'], 'shuffle': True, 'drop_last': True}
    train_sampler = DistributedSampler(train_dataset, **sampler_kwargs)
    val_sampler = DistributedSampler(val_dataset, **sampler_kwargs)
    loader_kwargs = {'collate_fn': _collate_fn, 'shuffle': False, 'pin_memory': False}
    train_dataloader = DataLoader(train_dataset, batch_size=ds_config['train_micro_batch_size_per_gpu'], sampler=train_sampler, **loader_kwargs)
    val_dataloader = DataLoader(train_dataset, batch_size=config['validation']['batch_size'], sampler=val_sampler, **loader_kwargs)

    # This will not work when we try to load the pretrained weights, as we'll get the following error
    # "size mismatch for norm.weight: copying a param with shape torch.Size([4096]) from checkpoint, the shape in current model is torch.Size([0])."
    # with deepspeed.zero.Init():
    #     model_args: ModelArgs = ModelArgs(**model_config)
    #     model = Transformer(model_args)

    model = create_lora_model_and_load_checkpoint(model_config=model_config, lora_config=lora_config, ckpt_file=config['model_ckpt_file'])

    # enable activation checkpointing
    model.enable_activation_checkpointing(deepspeed.checkpointing.checkpoint)
    model_params = get_model_parameters_for_optimizer(model, ds_config['optimizer']['params']['weight_decay'])

    # Force set 'requires_grad' on the embedding weights if not apply LoRA to train embedding
    # This is to avoid error "RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn"
    # when using DeepSpeed's activation checkpointing
    if not lora_config['train_embedding']:
        # This will not actual train the embedding layer, as we didn't include the weights in the model_params for optimizer
        model.tok_embeddings.weight.requires_grad_(True)

    model_engine: deepspeed.DeepSpeedEngine
    model_engine, optimizer, _, lr_scheduler = deepspeed.initialize(model=model, model_parameters=model_params, config=ds_config)

    # Force to use custom logging logic, by default DeepSpeed logging is based on global_samples, we want to log by global_steps
    model_engine.monitor.enabled = False

    is_rank0 = rank == 0
    if is_rank0:  # Only log on rank 0 to avoid duplication
        log_dir = os.path.join(ds_config['tensorboard']['output_path'], ds_config['tensorboard']['job_name'])
        ckpt_dir = os.path.join(ds_config['checkpoint']['output_path'], ds_config['checkpoint']['job_name'])
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(ckpt_dir, exist_ok=True)

    # Training loop
    num_epochs = config['train']['num_epochs']
    logger.info(f'Starting to run {num_epochs} training epochs')
    model_engine.train()
    elapsed_time = 0
    num_samples = 0
    model_engine.global_steps = 0
    # Store metrics over N gradient accumulation steps
    train_metrics_list = []
    for epoch in range(num_epochs):
        logger.info(f'Epoch {epoch}')
        # Set epoch for distributed sampler
        train_sampler.set_epoch(epoch)
        val_sampler.set_epoch(epoch)

        for batch in train_dataloader:
            t0 = time.time()
            for k in batch.keys():
                batch[k] = batch[k].to(model_engine.local_rank)

            outputs = model_engine(batch['inputs'])
            loss = masked_cross_entropy_loss(logits=outputs, labels=batch['labels'], loss_mask=batch['loss_mask'])
            model_engine.backward(loss)
            model_engine.step()

            # Collect metrics for current micro batch
            metrics = compute_evaluation_metrics(loss=loss, logits=outputs, labels=batch['labels'], loss_mask=batch['loss_mask'], prefix='Train/GlobalSteps/')
            train_metrics_list.append(metrics)

            num_samples += outputs.size(0)
            elapsed_time += time.time() - t0

            if model_engine.is_gradient_accumulation_boundary():
                if model_engine.global_steps < 1:
                    continue

                # Aggregate over N gradient accumulation steps locally
                train_metrics = aggregate_metrics(train_metrics_list)
                # Aggregate over GPU nodes
                train_metrics = aggregate_metrics_across_gpus(train_metrics, average=True)
                # Reset for next step
                train_metrics_list = []
                # Add throughput metrics
                more_metrics = {
                    'Throughput/samples_per_second': num_samples / elapsed_time if elapsed_time > 0 else 0,
                }
                more_metrics = aggregate_metrics_across_gpus(more_metrics, average=False)
                train_metrics.update(more_metrics)
                train_metrics['Throughput/total_samples'] = model_engine.global_samples

                # Log metrics
                if is_rank0 and model_engine.global_steps % ds_config['steps_per_print'] == 0:
                    learning_rates = model_engine.get_lr()
                    if learning_rates:
                        train_metrics['Train/GlobalSteps/lr'] = learning_rates[0]
                    # Add GPU metrics
                    system_metrics = get_system_metrics()
                    train_metrics.update(system_metrics)

                    # Write custom metrics
                    summary_events = [(k, v, model_engine.global_steps) for k, v in train_metrics.items()]
                    model_engine.monitor.write_events(summary_events)

                    # Log to console
                    if model_engine.global_steps % 200 == 0:
                        logger.info(f'Train metrics at global train steps {model_engine.global_steps}')
                        pretty_print(train_metrics)

                # Save checkpoint
                if model_engine.global_steps % ds_config['checkpoint']['save_interval'] == 0:
                    # This will only save trainable weights
                    model_engine.save_checkpoint(ckpt_dir, tag=f'global_steps_{model_engine.global_steps}', exclude_frozen_parameters=True)

                    if is_rank0:
                        meta_file = os.path.join(ckpt_dir, 'params.json')
                        lora_file = os.path.join(ckpt_dir, 'lora.json')
                        if not os.path.exists(meta_file):
                            save_json_file(model_config, meta_file)
                            save_json_file(lora_config, lora_file)

                # Validation, not working for stage3, as it will break LoRA weights
                if ds_config['zero_optimization']['stage'] != 3 and config['validation']['enabled'] and model_engine.global_steps % config['validation']['interval'] == 0:
                    run_validation(model_engine=model_engine, val_dataloader=val_dataloader, num_val_steps=config['validation']['steps'])

    # Create final checkpoint
    model_engine.save_checkpoint(ckpt_dir, tag=f'global_steps_{model_engine.global_steps}', exclude_frozen_parameters=True)

    dist.barrier()


if __name__ == '__main__':
    main()
