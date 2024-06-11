"""
Convert Meta's llama3 model weights to pure PyTorch, without using fairscale layers.

Code adapted from vanilla-llama project:
https://github.com/galatolofederico/vanilla-llama/blob/main/convert.py
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


def convert(llama3_dir: str, output_dir: str):
    assert os.path.exists(llama3_dir) and os.path.isdir(llama3_dir), 'Invalid Meta llama3 model checkpoints dir'
    src_files = os.listdir(llama3_dir)
    assert len(src_files) > 1, "Llama3 dir can not be empty"
    assert 'tokenizer.model' in src_files, "Tokenizer model 'tokenizer.model' not found in llama3 path"
    assert 'params.json' in src_files, "Model config file 'params.json' not found in llama3 path"

    os.makedirs(output_dir, exist_ok=True)

    if 'tokenizer.model' not in os.listdir(output_dir):
        logger.info(f'Copy tokenizer checkpoint file to {output_dir!r}')
        shutil.copy(os.path.join(llama3_dir, 'tokenizer.model'), output_dir)

    torch.set_default_tensor_type(torch.HalfTensor)

    checkpoints = sorted(Path(llama3_dir).glob('*.pth'))
    logger.info(f'Found {len(checkpoints)} checkpoint files from {llama3_dir!r}')
    with open(Path(llama3_dir) / 'params.json', 'r') as f:
        params = json.loads(f.read())
        params['max_seq_len'] = 2048
        params['max_batch_size'] = 1

    model_args = ModelArgs(**params)
    model = Transformer(model_args)

    # Mapping from key to weights shard dimension, dimension of 'None' means no shard is applied
    key_to_shard_dim = {
        'w1': 0,
        'w2': -1,
        'w3': 0,
        'wo': -1,
        'wq': 0,
        'wk': 0,
        'wv': 0,
        'output': 0,
        'tok_embeddings': -1,
        'ffn_norm': None,
        'attention_norm': None,
        'norm': None,
        'rope': None,
    }

    converted_state_dict = {}

    for i, ckpt in tqdm(enumerate(checkpoints), total=len(checkpoints)):
        checkpoint = torch.load(ckpt, map_location='cpu')
        for parameter_name, parameter in model.named_parameters():
            if parameter_name not in converted_state_dict:
                # Create a place holder, which will eventually be overwritten by the following operations
                converted_state_dict[parameter_name] = torch.zeros_like(parameter, device='cpu')
            short_name = parameter_name.split('.')[-2]
            if key_to_shard_dim[short_name] is None and i == 0:
                # for layers like RMSNorm, there are no shard, as these are identical across different checkpoints
                # so only need to copy from the very first checkpoint
                converted_state_dict[parameter_name] = checkpoint[parameter_name]
            elif key_to_shard_dim[short_name] == 0:
                # weights was shaded on dimension 0
                size = checkpoint[parameter_name].size(0)
                converted_state_dict[parameter_name][size * i : size * (i + 1), :] = checkpoint[parameter_name]
            elif key_to_shard_dim[short_name] == -1:
                # weights was shaded on dimension -1, which is the same on dimension 1, as it's a NxM matrix
                size = checkpoint[parameter_name].size(-1)
                converted_state_dict[parameter_name][:, size * i : size * (i + 1)] = checkpoint[parameter_name]
            del checkpoint[parameter_name]
        del checkpoint
        gc.collect()

    logger.info('Verifying that we can load the state dict with strict mode')
    model.load_state_dict(converted_state_dict, strict=True)

    meta_output_file = os.path.join(output_dir, 'params.json')
    logger.info(f'Saving model params.json file to {meta_output_file!r}')
    with open(meta_output_file, 'w') as f:
        f.write(json.dumps(params, indent=2))

    ckpt_output_file = os.path.join(output_dir, 'consolidated.pth')
    logger.info(f'Saving consolidated model checkpoint file to {ckpt_output_file!r}')
    torch.save(converted_state_dict, ckpt_output_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--llama3-dir', type=str, required=True, help='Path contains Meta llama3 model checkpoints')
    parser.add_argument('--output-dir', type=str, required=True, help='Where to save the converted checkpoint')
    args = parser.parse_args()

    convert(args.llama3_dir, args.output_dir)
