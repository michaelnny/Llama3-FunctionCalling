"""
Convert our custom fine-tuned llama3 model weights to Meta's format (shard using fairscale layers)

This is basically undo the job of 'convert_from_meta_weights.py'.
This could be useful if we wish to deploy our custom tine-tuned model to Triton inference server using the TensorRT-LLM backend.
Because the TensorRT-LLM project has script to convert Meta's weights to build TRT engine.
"""

import argparse
import json
import gc
import torch

from tqdm import tqdm
import os
import shutil

# support running without installing as a package
from pathlib import Path
import sys

wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from llama3.core.model import ModelArgs, Transformer
from llama3.utils.logger import create_logger

logger = create_logger()


def convert(config_file: str, ckpt_file: str, num_shards: int, output_dir: str):
    assert os.path.exists(config_file) and os.path.isfile(config_file), 'Invalid custom llama3 model config path'
    assert os.path.exists(ckpt_file) and os.path.isfile(ckpt_file), 'Invalid custom llama3 model checkpoint path'
    assert 1 <= num_shards <= 8, 'Invalid number of shards'

    os.makedirs(output_dir, exist_ok=True)
    torch.set_default_tensor_type(torch.HalfTensor)

    logger.info('Creating model instance...')
    with open(config_file, 'r') as f:
        params = json.loads(f.read())
        params['max_seq_len'] = 2048
        params['max_batch_size'] = 32

    model_args = ModelArgs(**params)
    model = Transformer(model_args)
    # Make sure we can load the checkpoint
    logger.info('Loading model checkpoint...')
    checkpoint = torch.load(ckpt_file, map_location='cpu')
    model.load_state_dict(checkpoint)
    logger.info('Finished loading model checkpoint')
    del checkpoint
    gc.collect()

    # # optional, convert to float16
    # model = model.to(torch.float16)

    # Mapping from key to weights shard dimension, dimension of 'None' means no shard is applied
    key_to_shard_dim = {
        'w1': 0,
        'w2': 1,
        'w3': 0,
        'wo': 1,
        'wq': 0,
        'wk': 0,
        'wv': 0,
        'output': 0,
        'tok_embeddings': 1,
        'ffn_norm': None,
        'attention_norm': None,
        'norm': None,
        'rope': None,
    }

    for i in tqdm(range(num_shards), total=num_shards):
        shard_state_dict = {}
        for parameter_name, parameter in model.named_parameters():
            short_name = parameter_name.split('.')[-2]

            assert short_name in key_to_shard_dim, f"Short name {short_name} not found in key_to_shard_dim"
            shard_dim = key_to_shard_dim[short_name]
            assert shard_dim in [0, 1, None], f"Invalid shard dimension {shard_dim} for parameter name {parameter_name}"
            assert not parameter_name in shard_state_dict

            if shard_dim is None:
                # for layers like RMSNorm, there are no shard, as these are identical across different checkpoints
                # so only need to copy to each shard
                shard_state_dict[parameter_name] = parameter.clone()
            else:
                size = parameter.size(shard_dim)
                assert size % num_shards == 0, f"Parameter size {size} not divisible by num_shards {num_shards}"
                shard_size = size // num_shards
                slice_start_idx = shard_size * i
                slice_end_idx = shard_size * (i + 1)
                logger.debug(f'Shard {i} - Layer {parameter_name} - Slicing from {slice_start_idx} to {slice_end_idx}')
                if shard_dim == 0:
                    # Shard weights on dimension 0
                    shard_state_dict[parameter_name] = parameter[slice_start_idx:slice_end_idx, :].clone()
                elif shard_dim == 1:
                    # Shard weights on dimension 1
                    shard_state_dict[parameter_name] = parameter[:, slice_start_idx:slice_end_idx].clone()
                logger.debug(f'{short_name}: {parameter.size()} --> {shard_state_dict[parameter_name].size()}')

        shard_sequence = f"0{i}" if i < 10 else f"{i}"
        shard_ckpt_file = os.path.join(output_dir, f"consolidated.{shard_sequence}.pth")
        logger.info(f'Saving shard checkpoint to {shard_ckpt_file!r}')
        torch.save(shard_state_dict, shard_ckpt_file)

    meta_output_file = os.path.join(output_dir, 'params.json')
    logger.info(f'Saving model params.json file to {meta_output_file!r}')
    with open(meta_output_file, 'w') as f:
        f.write(json.dumps(params, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-file', type=str, required=True, help='Path contains model para.json file')
    parser.add_argument('--ckpt-file', type=str, required=True, help='Path contains custom llama3 model checkpoint')
    parser.add_argument('--num-shards', type=int, required=True, help='Number of shard to use for the output checkpoint')
    parser.add_argument('--output-dir', type=str, required=True, help='Where to save the shard checkpoint')
    args = parser.parse_args()

    convert(args.config_file, args.ckpt_file, args.num_shards, args.output_dir)
