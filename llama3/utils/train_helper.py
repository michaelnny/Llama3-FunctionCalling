# Copyright (c) 2023 Michael Hu.
# This project is released under the MIT License.
# See the accompanying LICENSE file for details.

from typing import Tuple, List
import torch
import math
import os
import gc
import time
from logging import getLogger


from llama3.core.lora import replace_with_lora_layers, mark_only_lora_as_trainable
from llama3.core.model import ModelArgs, Transformer

logger = getLogger(__name__)


def set_seed(seed):
    # seed must be the same in all processes
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def load_model_checkpoint(model: Transformer, ckpt_file: str) -> None:
    assert os.path.exists(ckpt_file), f'Invalid checkpoint file {ckpt_file}'
    t0 = time.time()
    checkpoint = torch.load(ckpt_file, map_location='cpu')
    model.load_state_dict(checkpoint, strict=False)
    logger.info(f'Loaded model in {time.time() - t0:.2f} seconds')
    del checkpoint
    gc.collect()


def create_lora_model_and_load_checkpoint(model_config: dict, lora_config: dict, ckpt_file: str) -> Transformer:
    assert os.path.exists(ckpt_file), 'Invalid checkpoint file'

    logger.info('Creating model ...')
    model_args: ModelArgs = ModelArgs(**model_config)
    model = Transformer(model_args)

    # Must replace layer before load weights
    logger.info('Replacing nn.Linear with LoRALinear layers ...')
    replace_with_lora_layers(model=model, **lora_config)

    logger.info('Loading model checkpoint ...')
    load_model_checkpoint(model, ckpt_file)

    mark_only_lora_as_trainable(model, lora_config['train_bias'])

    num_trainable, num_frozen = compute_num_trainable_params(model)
    logger.info(f'Number of trainable parameters: {num_trainable:,}')
    logger.info(f'Number of frozen parameters: {num_frozen:,}')

    return model


def compute_num_trainable_params(model: torch.nn.Module) -> Tuple[int, int]:
    num_trainable = 0
    num_frozen = 0

    for p_name, params in model.named_parameters():
        is_trainable = params.requires_grad
        is_quantized = hasattr(params, 'quant_state')

        # quantized layer is not trainable
        if not is_trainable and is_quantized:
            num_params = math.prod(params.quant_state.shape)
        else:
            num_params = params.numel()

        num_trainable += num_params if is_trainable else 0
        num_frozen += num_params if not is_trainable else 0

    return num_trainable, num_frozen


def get_model_parameters_for_optimizer(
    model: Transformer,
    weight_decay: float,
) -> List[dict]:
    """
    Returns the trainable parameters of the model,
    where we skip apply weight decay to layer norm, embedding, and all bias,
    and apply weight decay to the reset of parameters.
    """

    # Create empty lists to store parameters for weight decay and no weight decay.
    decay = []
    no_decay = []

    for p_name, params in model.named_parameters():
        is_trainable = params.requires_grad

        if is_trainable:
            # Check for parameters corresponding to torch.nn.LayerNorm or torch.nn.Embedding.
            if p_name.endswith('bias') or p_name.endswith('norm.weight') or p_name.endswith('tok_embeddings.weight'):
                no_decay.append(params)
            else:
                decay.append(params)

    if weight_decay > 0:
        num_decay = sum(p.numel() for p in decay)
        num_nodecay = sum(p.numel() for p in no_decay)
        logger.info(f'Number of decayed parameters: {num_decay:,}')
        logger.info(f'Number of non-decayed parameters: {num_nodecay:,}')

    optim_groups = [
        {'params': decay, 'weight_decay': weight_decay},
        {'params': no_decay, 'weight_decay': 0.0},
    ]

    return optim_groups


def create_optimizer(
    model: torch.nn.Module,
    lr: float,
    eps: float,
    weight_decay: float,
    betas: Tuple[float],
    fused: bool = False,
) -> torch.optim.Adam:
    """
    Returns Adam optimizer, where we skip apply weight decay to layer norm, embedding, and all bias,
    and apply weight decay to the reset of parameters.
    """

    optim_groups = get_model_parameters_for_optimizer(model, weight_decay)

    kwargs = {'lr': lr, 'eps': eps, 'betas': betas, 'fused': fused}
    optimizer = torch.optim.Adam(optim_groups, **kwargs)
    return optimizer
