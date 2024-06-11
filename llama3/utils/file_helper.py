# Copyright (c) 2023 Michael Hu.
# This project is released under the MIT License.
# See the accompanying LICENSE file for details.

import os
import json


def read_json_file(input_file: str) -> dict:
    """Returns json objects or None if input file not exists or is not .json file."""
    if not os.path.exists(input_file) or not os.path.isfile(input_file) or not input_file.endswith('.json'):
        return None

    with open(input_file, 'r', encoding='utf-8') as file:
        content = file.read()
        return json.loads(content)


def save_json_file(content: dict, output_file: str) -> None:

    with open(output_file, 'w') as f:
        f.write(json.dumps(content, indent=2))
