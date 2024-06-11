# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.

from typing import List, Optional

import fire
import torch
from llama3.core.generation import Dialog, Llama
from llama3.core.prompt import insert_functions_to_system_message


def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 1024,
    max_batch_size: int = 16,
    max_gen_len: Optional[int] = None,
):
    """
    Examples to run with the models finetuned for chat. Prompts correspond of chat
    turns between the user and assistant with the final one always being the user.

    An optional system prompt at the beginning to control how the model should respond
    is also supported.

    The context window of llama3 models is 8192 tokens, so `max_seq_len` needs to be <= 8192.

    `max_gen_len` is optional because finetuned models are able to stop generations naturally.
    """
    generator = Llama.build(ckpt_dir=ckpt_dir, tokenizer_path=tokenizer_path, max_seq_len=max_seq_len, max_batch_size=max_batch_size, device='cuda' if torch.cuda.is_available() else 'cpu')

    dialogs_with_tools: List[dict] = [
        {
            'tools': [
                {
                    'type': 'function',
                    'function': {
                        'name': 'rate_movie',
                        'description': "Rate a movie based on user's review",
                        'parameters': {
                            'type': 'object',
                            'properties': {'movie_title': {'type': 'string', 'description': 'The title of the movie'}, 'rating': {'type': 'number', 'description': 'The rating given by the user (1-10)'}, 'review': {'type': 'string', 'description': 'The review of the movie'}},
                            'required': ['movie_title', 'rating', 'review'],
                        },
                    },
                }
            ],
            'messages': [
                {'role': 'user', 'content': 'I just watched the movie "Inception" and I would like to rate it.'},
            ],
        },
        {
            'tools': [
                {
                    'type': 'function',
                    'function': {
                        'name': 'rate_movie',
                        'description': "Rate a movie based on user's review",
                        'parameters': {
                            'type': 'object',
                            'properties': {'movie_title': {'type': 'string', 'description': 'The title of the movie'}, 'rating': {'type': 'number', 'description': 'The rating given by the user (1-10)'}, 'review': {'type': 'string', 'description': 'The review of the movie'}},
                            'required': ['movie_title', 'rating', 'review'],
                        },
                    },
                }
            ],
            'messages': [
                {'role': 'user', 'content': 'I just watched the movie "Inception" and I would like to rate it.'},
                {'role': 'assistant', 'content': 'Sure, I can help you with that. Could you please provide a rating from 1 to 10 and a brief review of the movie?'},
                {'role': 'user', 'content': 'I would rate it 9. It was a mind-bending journey through the architecture of the mind. The plot was complex but engaging, and the performances were top-notch.'},
            ],
        },
        {
            'tools': [
                {
                    'type': 'function',
                    'function': {
                        'name': 'rate_movie',
                        'description': "Rate a movie based on user's review",
                        'parameters': {
                            'type': 'object',
                            'properties': {'movie_title': {'type': 'string', 'description': 'The title of the movie'}, 'rating': {'type': 'number', 'description': 'The rating given by the user (1-10)'}, 'review': {'type': 'string', 'description': 'The review of the movie'}},
                            'required': ['movie_title', 'rating', 'review'],
                        },
                    },
                }
            ],
            'messages': [
                {'role': 'user', 'content': 'I just watched the movie "Inception" and I would like to rate it.'},
                {'role': 'assistant', 'content': 'Sure, I can help you with that. Could you please provide a rating from 1 to 10 and a brief review of the movie?'},
                {'role': 'user', 'content': 'I would rate it 9. It was a mind-bending journey through the architecture of the mind. The plot was complex but engaging, and the performances were top-notch.'},
                {
                    'role': 'assistant',
                    'content': "{'tool_uses': [{'recipient_name': 'functions.rate_movie', 'parameters': {'movie_title': 'Inception', 'rating': 9, 'review': 'It was a mind-bending journey through the architecture of the mind. The plot was complex but engaging, and the performances were top-notch.'}}]}",
                },
                {'role': 'tool', 'content': '[{\'status\': \'success\', \'message\': "Your rating and review for the movie \'Inception\' has been successfully recorded. Thank you for your feedback."}]'},
            ],
        },
        {
            'tools': [
                {
                    'type': 'function',
                    'function': {
                        'name': 'calculate_tip',
                        'description': 'Calculate the tip amount for a given bill',
                        'parameters': {'type': 'object', 'properties': {'bill_amount': {'type': 'number', 'description': 'The total bill amount'}, 'tip_percentage': {'type': 'number', 'description': 'The tip percentage'}}, 'required': ['bill_amount', 'tip_percentage']},
                    },
                }
            ],
            'messages': [
                {'role': 'user', 'content': 'Hi, I need help with calculating a tip. My bill amount is $50 and I want know how much should I tip if a leave a 20% tip.'},
            ],
        },
        {
            'tools': [
                {
                    'type': 'function',
                    'function': {
                        'name': 'search_books',
                        'description': 'Search for books based on title, author, or genre',
                        'parameters': {
                            'type': 'object',
                            'properties': {'keyword': {'type': 'string', 'description': 'The keyword to search for in book title, author, or genre'}},
                            'required': ['keyword'],
                        },
                    },
                }
            ],
            'messages': [{'role': 'user', 'content': 'I am looking for a book but I can\'t remember the full title. It had the word "Sun" in it.'}],
        },
        {
            'tools': [
                {
                    'type': 'function',
                    'function': {
                        'name': 'validate_email',
                        'description': 'Validate if an email address is valid',
                        'parameters': {'type': 'object', 'properties': {'email': {'type': 'string', 'format': 'email', 'description': 'The email address to validate'}}, 'required': ['email']},
                    },
                },
                {
                    'type': 'function',
                    'function': {
                        'name': 'convert_currency',
                        'description': 'Convert currency from one type to another',
                        'parameters': {
                            'type': 'object',
                            'properties': {'amount': {'type': 'number', 'description': 'The amount to convert'}, 'from_currency': {'type': 'string', 'description': 'The currency to convert from'}, 'to_currency': {'type': 'string', 'description': 'The currency to convert to'}},
                            'required': ['amount', 'from_currency', 'to_currency'],
                        },
                    },
                },
            ],
            'messages': [
                {'role': 'user', 'content': 'Hi, can you check if my email address is valid?'},
            ],
        },
        {
            'tools': [
                {
                    'type': 'function',
                    'function': {
                        'name': 'validate_email',
                        'description': 'Validate if an email address is valid',
                        'parameters': {'type': 'object', 'properties': {'email': {'type': 'string', 'format': 'email', 'description': 'The email address to validate'}}, 'required': ['email']},
                    },
                },
                {
                    'type': 'function',
                    'function': {
                        'name': 'convert_currency',
                        'description': 'Convert currency from one type to another',
                        'parameters': {
                            'type': 'object',
                            'properties': {'amount': {'type': 'number', 'description': 'The amount to convert'}, 'from_currency': {'type': 'string', 'description': 'The currency to convert from'}, 'to_currency': {'type': 'string', 'description': 'The currency to convert to'}},
                            'required': ['amount', 'from_currency', 'to_currency'],
                        },
                    },
                },
            ],
            'messages': [
                {'role': 'user', 'content': 'Hi, can you check if my email address is valid?'},
                {'role': 'assistant', 'content': 'Sure, I can help with that. Please provide me with the email address you want to validate.'},
                {'role': 'user', 'content': 'The email address is john.doe@example.com.'},
            ],
        },
        {
            'tools': [],
            'messages': [
                {'role': 'user', 'content': 'I am going to Paris, what should I see?'},
                {
                    'role': 'assistant',
                    'content': """\
Paris, the capital of France, is known for its stunning architecture, art museums, historical landmarks, and romantic atmosphere. Here are some of the top attractions to see in Paris:

1. The Eiffel Tower: The iconic Eiffel Tower is one of the most recognizable landmarks in the world and offers breathtaking views of the city.
2. The Louvre Museum: The Louvre is one of the world's largest and most famous museums, housing an impressive collection of art and artifacts, including the Mona Lisa.
3. Notre-Dame Cathedral: This beautiful cathedral is one of the most famous landmarks in Paris and is known for its Gothic architecture and stunning stained glass windows.

These are just a few of the many attractions that Paris has to offer. With so much to see and do, it's no wonder that Paris is one of the most popular tourist destinations in the world.""",
                },
                {'role': 'user', 'content': 'What is so great about #1?'},
            ],
        },
        {
            'tools': [],
            'messages': [
                {
                    'role': 'system',
                    'content': 'Always answer with emojis',
                },
                {'role': 'user', 'content': 'How to go from Beijing to NY?'},
            ],
        },
    ]

    dialogs = []

    for item in dialogs_with_tools:
        if item['tools'] and len(item['tools']) > 0:
            system_msg = insert_functions_to_system_message(item['tools'])
            system_turn = {'role': 'system', 'content': system_msg}
            dialog = [system_turn] + item['messages']
        else:
            dialog = item['messages']

        dialogs.append(dialog)

    results = generator.chat_completion(
        dialogs,
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
    )

    for dialog, result in zip(dialogs, results):
        for msg in dialog:
            if msg['role'] == 'system':
                continue
            print(f"{msg['role'].capitalize()}: {msg['content']}\n")
        print(f"> {result['generation']['role'].capitalize()}: {result['generation']['content']}")
        print('\n==================================\n')


if __name__ == '__main__':
    fire.Fire(main)
