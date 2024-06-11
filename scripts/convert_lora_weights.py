# Copyright (c) 2023 Michael Hu.
# This project is released under the MIT License.
# See the accompanying LICENSE file for details.


"""Merge LoRA fine-tunned checkpoint and pretrained checkpoint into a single checkpoint file."""
import gc
import os
import argparse
import json
import torch
import torch.nn as nn
from tqdm import tqdm

# support running without installing as a package
from pathlib import Path
import sys

wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from llama3.core.model import Transformer, ModelArgs
from llama3.core.lora import replace_with_lora_layers


def get_clean_state_dict(model: nn.Module):
    """Clean up lora weights and return cleaned state dict."""
    model_dict = model.state_dict()
    key_to_delete = [k for k in model_dict if 'lora_' in k]
    for del_key in key_to_delete:
        del model_dict[del_key]
    return model_dict


def main(config_dir: str, base_ckpt_path: str, lora_ckpt_path: str, output_dir: str) -> None:
    """Merges LoRA weights with pretrained base model.

    Args:
        config_dir: Path contains the model configure and LoRA configure files used for training.
        base_ckpt_path: The base checkpoint (like pre-trained or fine-tuned) used for training with lora.
        lora_ckpt_path: Path to the checkpoint with trained LoRA weights.
        output_dir: target path to save the merged stat_dict.
    """

    if not os.path.exists(config_dir):
        raise ValueError(f'Configure dir {config_dir!r} does not exist, aborting ...')
    if not os.path.exists(base_ckpt_path):
        raise ValueError(f'Pretrained checkpoint dir {base_ckpt_path!r} does not exist, aborting ...')
    if not os.path.exists(lora_ckpt_path):
        raise ValueError(f'LoRA checkpoint file {lora_ckpt_path!r} does not exist, aborting ...')

    model_config_path = os.path.join(config_dir, 'params.json')
    lora_config_path = os.path.join(config_dir, 'lora.json')
    if not os.path.exists(model_config_path):
        raise ValueError(f'Model configuration file {model_config_path!r} does not exist, aborting ...')
    if not os.path.exists(lora_config_path):
        raise ValueError(f'LoRA configuration file {lora_config_path!r} does not exist, aborting ...')

    # Create the output directory if necessary
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, mode=0o777, exist_ok=True)

    # torch.set_default_tensor_type(torch.HalfTensor)

    with open(model_config_path, 'r') as f:
        meta_params = json.load(f)
        meta_params['embed_dropout'] = 0.0
        meta_params['attention_dropout'] = 0.0
        meta_params['hidden_dropout'] = 0.0

    with open(lora_config_path, 'r') as f:
        lora_config = json.load(f)

    model_args = ModelArgs(**meta_params)
    model = Transformer(model_args)

    replace_with_lora_layers(model, **lora_config)

    # 1. Load the pretrained weights
    print(f'Loading base model checkpoint {base_ckpt_path!r}...')
    base_checkpoint = torch.load(base_ckpt_path, map_location='cpu')
    model.load_state_dict(base_checkpoint, strict=False)
    del base_checkpoint
    gc.collect()

    # 2. Load the fine-tuned lora weights
    print(f'Loading LoRA model checkpoint {lora_ckpt_path!r}...')
    lora_checkpoint = torch.load(lora_ckpt_path, map_location='cpu')
    model.load_state_dict(lora_checkpoint, strict=False)
    del lora_checkpoint
    gc.collect()

    # if os.path.isfile(lora_ckpt_path) and (lora_ckpt_path.endswith('.pth') or lora_ckpt_path.endswith('.pt')):
    #     lora_checkpoints = [lora_ckpt_path]
    # else:
    #     # when using deepspeed, could have multiple checkpoints from different GPUs
    #     lora_checkpoints = sorted(Path(lora_ckpt_path).glob('*model_states.pt'))

    # print(f'Found {len(lora_checkpoints)} LoRA checkpoint files from {lora_ckpt_path!r}')

    # for i, ckpt_file in tqdm(enumerate(lora_checkpoints), total=len(lora_checkpoints)):
    #     print(f'Loading LoRA model checkpoint {ckpt_file!r}...')
    #     lora_checkpoint = torch.load(ckpt_file, map_location='cpu')
    #     model.load_state_dict(lora_checkpoint, strict=False)

    # 3. merge LoRA weights, which was handled inside the LoRALinear.train() method
    model.eval()

    # 4. optional, convert to float16
    model = model.to(torch.float16)

    # 5. Remove LoRA parameters from the model state
    state_dict = get_clean_state_dict(model)

    ckpt_output_file = os.path.join(output_dir, 'consolidated.pth')
    print(f'Saving merged model weights to {ckpt_output_file!r} ...')
    torch.save(state_dict, ckpt_output_file)

    meta_output_file = os.path.join(output_dir, 'params.json')
    if not os.path.exists(meta_output_file):
        del_keys = ('lora', 'dropout', 'use_cache', 'deepspeed', 'checkpointing')
        meta = {k: v for k, v in meta_params.items() if all([n not in k for n in del_keys])}

        print(f'Saving model metadata to {meta_output_file!r} ...')
        with open(meta_output_file, 'w', encoding='utf-8') as f:
            json.dump(meta, f, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-dir', type=str, required=True, help='Path for model params.json and lora.json files')
    parser.add_argument('--base-ckpt-path', type=str, required=True, help='Path contains Meta model checkpoint')
    parser.add_argument('--lora-ckpt-path', type=str, required=True, help='LoRA model checkpoint file')
    parser.add_argument('--output-dir', type=str, required=True, help='Where to save the converted model checkpoint')
    args = parser.parse_args()

    main(config_dir=args.config_dir, base_ckpt_path=args.base_ckpt_path, lora_ckpt_path=args.lora_ckpt_path, output_dir=args.output_dir)
