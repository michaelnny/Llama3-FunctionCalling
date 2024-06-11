"""Deep clean glaive-function-calling-v2 dataset and turn them into our custom prompt/response format for fine-tuning."""

import argparse
from typing import List, Tuple, Union
import random
import os
import pickle
import json
import time
import numpy as np
from tqdm import tqdm

# support running without installing as a package
from pathlib import Path
import sys

wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from llama3.core.tokenizer import Tokenizer, ChatFormat
from llama3.core.prompt import insert_functions_to_system_message
from llama3.utils.file_helper import read_json_file, save_json_file

ROLE_MAPPING = [
    {'keyword': 'USER:', 'to_role': 'user'},
    {'keyword': 'ASSISTANT:', 'to_role': 'assistant'},
    {'keyword': 'FUNCTION RESPONSE:', 'to_role': 'tool'},
]

FUNC_KEYWORDS = 'with access to the following functions. Use them if required -'


def parse_runtime_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-v',
        '--verbose',
        action='store_true',
        required=False,
        default=False,
        help='Enable verbose output',
    )

    parser.add_argument(
        '--source-file',
        type=str,
        required=True,
        help='Path to the glaive-function-calling-v2.json file',
    )

    parser.add_argument(
        '--tokenizer-file',
        type=str,
        required=True,
        help='Path to the tiktoken model checkpoint file',
    )

    parser.add_argument(
        '--val-ratio',
        type=float,
        required=False,
        default=0.15,
        help='Ratio of train and validation datasets split',
    )

    parser.add_argument(
        '--save-dir',
        type=str,
        required=True,
        help='Path to save the extracted train and validation dataset files',
    )

    parser.add_argument(
        '--seed',
        type=int,
        required=False,
        default=3,
        help='Runtime seed',
    )

    return parser.parse_args()


def split_and_tokenize_prompt_response(dataset: List[dict], chat_formatter: ChatFormat) -> None:
    """Split messages into prompt and response, then tokenize them"""
    if dataset is None or chat_formatter is None:
        raise SystemError('Invalid dataset or chat_formatter')

    print(f'Tokenizing {len(dataset)} items in the dataset')

    for i, sample in tqdm(enumerate(dataset), total=len(dataset)):
        prompt = []

        # Inject functions into system prompt, also mixing a small amount of samples without function calling
        if len(sample['tools']) > 0 or (len(sample['tools']) == 0 and random.random() > 0.9):
            system_msg = insert_functions_to_system_message(sample['tools'])
            if sample['messages'][0]['role'] == 'system':
                mixed_sys_msg = system_msg + '\n\n' + sample['messages'][0]['content']
                sample['messages'][0]['content'] = mixed_sys_msg
            else:
                prompt.append({'role': 'system', 'content': system_msg})

        for item in sample['messages'][:-1]:
            prompt.append(item)

        response = sample['messages'][-1]

        assert prompt[-1]['role'] != 'assistant'
        assert response['role'] == 'assistant'

        prompt_ids = chat_formatter.encode_dialog_prompt(prompt)
        response_ids = chat_formatter.encode_dialog_response(response)

        sample['prompt_len'] = len(prompt_ids)
        sample['response_len'] = len(response_ids)

        # Input ids contains full prompt + response (except the last response token), this is for model input
        sample['inputs'] = prompt_ids + response_ids[:-1]

        # Label ids consists of prompt (except the first token), plus full response ids, this is for compute crossentropy loss during training
        sample['labels'] = prompt_ids[1:] + response_ids

        # Optional loss mask, with 0s mark for prompt and 1s mark for response
        sample['loss_mask'] = [0] * (len(prompt_ids) - 1) + [1] * len(response_ids)

        assert len(sample['inputs']) == len(sample['labels']) == len(sample['loss_mask'])


def compute_seq_statistics(dataset: List[dict]) -> dict:
    result = {}

    prompt_lens = []
    response_lens = []
    sequence_lens = []

    for sample in dataset:
        prompt_lens.append(sample['prompt_len'])
        response_lens.append(sample['response_len'])
        sequence_lens.append(len(sample['inputs']))

    prompt_lens = np.array(prompt_lens)
    response_lens = np.array(response_lens)
    sequence_lens = np.array(sequence_lens)

    def compute_percentiles(array: np.ndarray) -> Tuple[int, int, int, int]:
        p80 = round(np.percentile(array, 80), 2)
        p90 = round(np.percentile(array, 90), 2)
        p99 = round(np.percentile(array, 99), 2)

        return (p80, p90, p99)

    prompt_len_p80, prompt_len_p90, prompt_len_p99 = compute_percentiles(prompt_lens)
    response_len_p80, response_len_p90, response_len_p99 = compute_percentiles(response_lens)
    sequence_len_p80, sequence_len_p90, sequence_len_p99 = compute_percentiles(sequence_lens)

    result['prompt_len_p80'] = prompt_len_p80
    result['prompt_len_p90'] = prompt_len_p90
    result['prompt_len_p99'] = prompt_len_p99
    result['response_len_p80'] = response_len_p80
    result['response_len_p90'] = response_len_p90
    result['response_len_p99'] = response_len_p99
    result['sequence_len_p80'] = sequence_len_p80
    result['sequence_len_p90'] = sequence_len_p90
    result['sequence_len_p99'] = sequence_len_p99

    return result


def main():

    FLAGS = parse_runtime_args()
    np.random.seed(FLAGS.seed)
    random.seed(FLAGS.seed)

    os.makedirs(FLAGS.save_dir, exist_ok=True)
    train_out_file = os.path.join(FLAGS.save_dir, 'train.pk')
    val_out_file = os.path.join(FLAGS.save_dir, 'validation.pk')
    meta_out_file = os.path.join(FLAGS.save_dir, 'metadata.json')

    tokenizer = Tokenizer(FLAGS.tokenizer_file)
    chat_formatter = ChatFormat(tokenizer)

    start_time = time.time()
    dataset = read_json_file(FLAGS.source_file)

    split_and_tokenize_prompt_response(dataset, chat_formatter)

    for _ in range(2):
        random.shuffle(dataset)

    split_idx = int(len(dataset) * FLAGS.val_ratio)

    train_dataset = dataset[split_idx:]
    val_dataset = dataset[:split_idx]

    print(f'Found {len(train_dataset)} items in training dataset')
    print(f'Found {len(val_dataset)} items in validation dataset')

    print(f'Saving train dataset to {train_out_file!r} ...')
    pickle.dump(train_dataset, open(train_out_file, 'wb'))
    print(f'Saving validation dataset to {val_out_file!r} ...')
    pickle.dump(val_dataset, open(val_out_file, 'wb'))

    train_stats = compute_seq_statistics(train_dataset)
    val_stats = compute_seq_statistics(val_dataset)

    meta = {
        'name': 'glaive-function-calling-v2-cleaned',
        'description': 'Deep cleaned version of glaive-function-calling-v2 for fine-tuning llama3 model',
        'structure': {
            'tools': 'a list contains function metadata',
            'messages': "a list of cleaned and well formatted chat history, where each turn has 'role' and 'content' properties",
            'inputs': 'a list token ids for the model input, contains prompt_ids + response_ids[:-1]',
            'labels': 'a list token ids for compute loss, contains prompt_ids[1:] + response_ids',
            'loss_mask': 'a list of 0s and 1s, with 0s for prompt id and 1s for response id, computed as: [0] * (len(prompt_ids)-1) + [1] * len(response_ids)',
        },
        'statistics': {
            'number_samples': {
                'train': len(train_dataset),
                'validation': len(val_dataset),
            },
            'sequence_length': {
                'train': train_stats,
                'validation': val_stats,
            },
        },
    }

    print(f'Saving metadata to {meta_out_file!r} ...')
    save_json_file(meta, meta_out_file)

    print(f'Finished building dataset in {time.time() - start_time:.2f} seconds')


if __name__ == '__main__':
    main()
