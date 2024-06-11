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

from llama3.core.prompt import insert_functions_to_system_message, check_validity_of_chat_messages
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
        '--save-file',
        type=str,
        required=True,
        help='Path to save the cleaned dataset file',
    )
    parser.add_argument(
        '--remove-no-fc',
        action='store_true',
        required=False,
        default=False,
        help='Remove samples do not have function calls',
    )
    parser.add_argument(
        '--seed',
        type=int,
        required=False,
        default=3,
        help='Runtime seed',
    )

    return parser.parse_args()


def extract_function_metadata(input_text: str) -> List[dict]:

    results = []
    if input_text is None or len(input_text) < 1 or FUNC_KEYWORDS not in input_text:
        return results

    function_splits = input_text.split(FUNC_KEYWORDS)[1].split('\n\n')
    for func in function_splits:
        function_metadata = json.loads(func)
        results.append({'type': 'function', 'function': function_metadata})

    return results


def extract_and_clean_content(input_text: str) -> Tuple[str, Union[str, dict]]:
    for item in ROLE_MAPPING:
        keyword = item['keyword']
        role = item['to_role']
        if input_text.startswith(keyword):
            content = input_text.replace(keyword, '').strip()
            content = content.replace('but as an AI, ', '')
            if role == 'tool':
                content = [json.loads(content)]
            elif content.startswith('<functioncall>'):  # build structured data for function call
                # try to turn function call from raw text to structured data
                content = content.replace('<functioncall>', '').strip()
                # replace single quotes with double quotes for valid JSON
                clean_content = content.replace("'{", '{').replace("'}", '}')
                data_json = json.loads(clean_content)
                # Make it compatible with openAI prompt format
                func_call = {'recipient_name': f"functions.{data_json['name']}", 'parameters': data_json['arguments']}
                content = {'tool_uses': [func_call]}
            return (role, content)

    return (None, None)


def extract_dataset_from_json_file(file: str, remove_no_fc: bool, verbose: bool) -> List[dict]:
    """Extract items with clean role structure from the original dataset json file"""

    if not os.path.exists(file):
        raise SystemError(f'The source json file {file} does not exist')

    print(f'Loading items from file {file}')
    raw_dataset = read_json_file(file)
    print(f'Found {len(raw_dataset)} items from file {file}')

    dataset = []

    for item in raw_dataset:
        sys_strings = item['system'].strip()
        chat_strings = item['chat'].strip()
        extracted_messages = []

        # Extract function metadata from system message, this will also filter out invalid items
        try:
            functions = extract_function_metadata(sys_strings)
            insert_functions_to_system_message(functions)
        except Exception as error:
            print(f'Error when try to extract function metadata {str(error)}')
            continue

        # prepare for split, we still want to preserve the role keywords after the split
        new_turn = '<new_turn>'
        for keyword in [item['keyword'] for item in ROLE_MAPPING]:
            chat_strings = chat_strings.replace(keyword, new_turn + keyword)

        # split and filter out empty strings
        chat_messages = [item.replace('<|endoftext|>', '').strip() for item in chat_strings.split(new_turn) if len(item.strip()) > 1]

        try:
            for i in range(len(chat_messages)):
                current_chat = chat_messages[i]
                role, content = extract_and_clean_content(current_chat)
                if role is None or content is None:
                    continue
                chat = {'role': role, 'content': content}
                extracted_messages.append(chat)
        except Exception as error:
            continue  # skip bad sample

        if not check_validity_of_chat_messages(extracted_messages):
            continue

        if verbose:
            print('-' * 80)
            for turn in extracted_messages:
                print(turn)
            print('\n')

        if remove_no_fc and not functions:  # skip sample with no functions
            continue

        dataset.append({'tools': functions, 'messages': extracted_messages})

    print(f'Found {len(dataset)} items after clean-up')
    return dataset


def make_tool_call_as_response(dataset: List[dict]) -> None:
    """Make sure the samples contain proper function calling as response.

    This make sure the model should be about to generate the proper tool call json as response"""
    count = 0

    for i in range(len(dataset)):
        sample = dataset[i]
        tool_call_indices = [i for i, chat in enumerate(sample['messages']) if chat['role'] == 'assistant' and isinstance(chat['content'], dict) and 'tool_uses' in chat['content']]
        if len(tool_call_indices) == 0:
            continue
        elif random.random() < 0.4:  # keep 40% of samples intact
            continue

        cut_idx = random.choice(tool_call_indices)
        messages = sample['messages'][: cut_idx + 1]  # +1 to make it inclusive
        assert messages[-1]['role'] == 'assistant' and isinstance(messages[-1]['content'], dict) and 'tool_uses' in messages[-1]['content']
        sample['messages'] = messages
        count += 1

    print(f'Found {count} items with tool call as response')


def make_asking_for_more_info_as_response(dataset: List[dict]) -> None:
    """Make sure the samples contain proper follow up question from the assistant as response.

    This make sure the model should asking for following up questions before generate function calls, if user has not provided already
    """

    count = 0
    for i in range(len(dataset)):
        sample = dataset[i]
        follow_up_indices = []
        j = 0
        while j < (len(sample['messages']) - 3):
            user_turn = sample['messages'][j]
            assistant_turn = sample['messages'][j + 1]
            user_followup_turn = sample['messages'][j + 2]
            assistant_fc_turn = sample['messages'][j + 3]

            if user_turn['role'] == 'user' and assistant_turn['role'] == 'assistant' and user_followup_turn['role'] == 'user' and assistant_fc_turn['role'] == 'assistant':
                if isinstance(assistant_fc_turn['content'], dict) and 'tool_uses' in assistant_fc_turn['content']:
                    follow_up_indices.append(j + 1)
                    j += 3
            j += 1

        if len(follow_up_indices) == 0:
            continue
        elif random.random() < 0.5:  # keep 50% of samples intact
            continue

        cut_idx = random.choice(follow_up_indices)
        messages = sample['messages'][: cut_idx + 1]  # +1 to make it inclusive
        assert messages[-1]['role'] == 'assistant', print(messages)
        sample['messages'] = messages
        count += 1

    print(f'Found {count} items with follow up question from the assistant as response')


def cutoff_low_quality_response(dataset: List[dict]) -> None:
    """Remove low quality response from the chat messages. This does not remove the sample.

    Any thing ends with the assistant saying "You're welcome ... feel free to ask." is considered low quality.
    """

    if not dataset:
        raise SystemError('Invalid dataset')

    count = 0
    for i in range(len(dataset)):
        sample = dataset[i]

        if len(sample['messages']) < 4:
            continue

        last_turn = sample['messages'][-1]
        assert last_turn['role'] == 'assistant'

        if isinstance(last_turn['content'], str):
            msg: str = last_turn['content'].lower()
            if msg.startswith("I'm sorry, ".lower()) or ((msg.startswith("you're welcome") or msg.startswith('great choice')) and 'feel free to ask' in msg):
                cut_messages = sample['messages'][:-2]
                assert cut_messages[-1]['role'] == 'assistant', print(f"{sample['messages']} \n\n {cut_messages}")
                sample['messages'] = cut_messages
                count += 1

    print(f'Found {count} items with low quality response')


def remove_low_quality_sample_in_place(lst: List[dict]):
    """
    Remove low quality sample from a list in place.
    """

    user_phases_to_skip = [
        "tell me the current stock price of",
        "need a random number between",
        "need to generate a QR code for my website",
        "need a QR code for my website",
        "I need to convert ",
        "I need to know the distance between",
        "I need help with calculating the tip",
    ]
    assistant_phases_to_skip = [
        "How long would you like your password to be",
    ]

    i = 0
    count = 0
    while i < len(lst):
        item = lst[i]

        if len(item['messages']) == 2:
            user_msg: str = item['messages'][0]['content'].lower()
            assistant_msg: str = item['messages'][1]['content'].lower()

            # Remove 90% of these low quality samples
            if user_msg.startswith("Can you ".lower()) and assistant_msg.startswith("I'm sorry, ".lower()) and random.random() >= 0.1:
                del lst[i]
                count += 1
            else:
                i += 1
        else:
            user_msg: str = item['messages'][0]['content'].lower()
            # Remove 80% of these low quality samples
            if any([phase.lower() in user_msg for phase in user_phases_to_skip]) and random.random() >= 0.2:
                del lst[i]
                count += 1

            elif not isinstance(item['messages'][1]['content'], dict):
                assistant_msg: str = item['messages'][1]['content'].lower()
                if any([phase.lower() in assistant_msg for phase in assistant_phases_to_skip]) and random.random() >= 0.1:
                    del lst[i]
                    count += 1
                else:
                    i += 1
            else:
                i += 1

    print(f'Found {count} low quality items')


def make_hashable(d):
    """
    Convert a dictionary to a hashable type (tuple of sorted items).
    """
    if isinstance(d, dict):
        return tuple(sorted((k, make_hashable(v)) for k, v in d.items()))
    if isinstance(d, list):
        return tuple(make_hashable(x) for x in d)
    return d


def remove_duplicates_in_place(lst: List[dict]):
    """
    Remove sample from a list in place.
    """
    seen = set()
    i = 0
    count = 0
    while i < len(lst):
        h = make_hashable(lst[i]['messages'])
        if h in seen:
            del lst[i]
            count += 1
        else:
            seen.add(h)
            i += 1

    print(f'Found {count} duplicate items')


def main():

    FLAGS = parse_runtime_args()
    np.random.seed(FLAGS.seed)
    random.seed(FLAGS.seed)
    os.makedirs(os.path.dirname(FLAGS.save_file), exist_ok=True)
    start_time = time.time()
    dataset = extract_dataset_from_json_file(FLAGS.source_file, FLAGS.remove_no_fc, FLAGS.verbose)

    remove_low_quality_sample_in_place(dataset)
    cutoff_low_quality_response(dataset)
    make_asking_for_more_info_as_response(dataset)
    make_tool_call_as_response(dataset)
    remove_duplicates_in_place(dataset)

    print(f'Saving cleaned dataset with {len(dataset)} items to {FLAGS.save_file!r} ...')
    save_json_file(dataset, FLAGS.save_file)
    print(f'Finished cleanup dataset in {time.time() - start_time:.2f} seconds')


if __name__ == '__main__':
    main()
