# Copyright (c) 2023 Michael Hu.
# This project is released under the MIT License.
# See the accompanying LICENSE file for details.


from typing import List
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset


class FineTuneDataset(Dataset):
    """For supervised fune-tuning, where we have pair of prompt:completion tokens."""

    def __init__(self, data_sources: List[str], max_seq_len: int = 2048) -> None:
        """
        Args:
            data_sources: a list of string path to where to load the dataset .pk files.
            max_seq_len: prompt_tokens + completion_tokens length greater than this will be discarded.
        """

        assert len(data_sources) > 0
        assert max_seq_len > 128

        self.data_sources = data_sources
        self.max_seq_len = max_seq_len

        self.data = []
        seq_lengths = []
        # Load datasets
        for source in data_sources:
            samples = pickle.load(open(source, 'rb'))
            for sample in samples:
                if len(sample['inputs']) != len(sample['labels']) != len(sample['loss_mask']):
                    continue
                elif len(sample['inputs']) > self.max_seq_len:
                    continue
                else:
                    self.data.append(
                        {
                            'inputs': sample['inputs'],
                            'labels': sample['labels'],
                            'loss_mask': sample['loss_mask'],
                        }
                    )
                    seq_lengths.append(len(sample['inputs']))

        # track sequence length statistics
        seq_lengths = np.array(seq_lengths)
        self.stats = {
            'data_sources': self.data_sources,
            'num_samples': len(self),
            'num_tokens': np.sum(seq_lengths),
            'sequence_lengths': {
                'p80': round(np.percentile(seq_lengths, 80), 2),
                'p90': round(np.percentile(seq_lengths, 90), 2),
                'p99': round(np.percentile(seq_lengths, 99), 2),
            },
        }

        del seq_lengths

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        sample = self.data[idx]

        return {
            'inputs': torch.tensor(sample['inputs'], dtype=torch.long),
            'labels': torch.tensor(sample['labels'], dtype=torch.long),
            'loss_mask': torch.tensor(sample['loss_mask'], dtype=torch.bool),
        }

    def get_metadata(self) -> dict:
        return self.stats


def custom_collate_fn(batch: List[dict], pad_id: int, max_seq_len: int) -> dict:
    """
    Custom collate function to pad the sequence to maximum length in the batch,
    and compute the loss mask for the batch.
    """

    batch_size = len(batch)

    max_batch_seq_len = max([len(item['inputs']) for item in batch])
    assert max_batch_seq_len <= max_seq_len

    batch_inputs = torch.full((batch_size, max_batch_seq_len), pad_id, dtype=torch.long)
    batch_labels = torch.full((batch_size, max_batch_seq_len), pad_id, dtype=torch.long)
    batch_loss_mask = torch.full((batch_size, max_batch_seq_len), 0, dtype=torch.bool)

    for i, item in enumerate(batch):
        curr_inputs = item['inputs']
        curr_labels = item['labels']
        curr_loss_mask = item['loss_mask']

        # right padding, a simplified example where 0s are pad id: [1, 2, 3] -> [1, 2, 3, 0, 0]
        batch_inputs[i, : len(curr_inputs)] = curr_inputs
        batch_labels[i, : len(curr_labels)] = curr_labels
        batch_loss_mask[i, : len(curr_loss_mask)] = curr_loss_mask

    return {'inputs': batch_inputs, 'labels': batch_labels, 'loss_mask': batch_loss_mask}
