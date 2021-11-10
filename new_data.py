import torch
import torchtext
import pandas as pd
import numpy as np
import os
from torch.utils.data import IterableDataset
from torch.utils.data import DataLoader
import io
import sys
import csv
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))


def to_map_style_dataset(iter_data):
    r"""Convert iterable-style dataset to map-style dataset.
    args:
        iter_data: An iterator type object. Examples include Iterable datasets, string list, text io, generators etc.
    Examples:
        >>> from torchtext.datasets import IMDB
        >>> from torchtext.data import to_map_style_dataset
        >>> train_iter = IMDB(split='train')
        >>> train_dataset = to_map_style_dataset(train_iter)
        >>> file_name = '.data/EnWik9/enwik9'
        >>> data_iter = to_map_style_dataset(open(file_name,'r'))
    """

    # Inner class to convert iterable-style to map-style dataset
    class _MapStyleDataset(torch.utils.data.Dataset):

        def __init__(self, iter_data):
            # TODO Avoid list issue #1296
            self._data = list(iter_data)

        def __len__(self):
            return len(self._data)

        def __getitem__(self, idx):
            return self._data[idx]

    return _MapStyleDataset(iter_data)


class _RawTextIterableDataset(torch.utils.data.IterableDataset):
    """Defines an abstraction for raw text iterable datasets.
    """

    def __init__(self, description, full_num_lines, iterator):
        """Initiate the dataset abstraction.
        """
        super(_RawTextIterableDataset, self).__init__()
        self.description = description
        self.full_num_lines = full_num_lines
        self._iterator = iterator
        self.num_lines = full_num_lines
        self.current_pos = None

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_pos == self.num_lines - 1:
            raise StopIteration
        item = next(self._iterator)
        if self.current_pos is None:
            self.current_pos = 0
        else:
            self.current_pos += 1
        return item

    def __len__(self):
        return self.num_lines

    def pos(self):
        """
        Returns current position of the iterator. This returns None
        if the iterator hasn't been used yet.
        """
        return self.current_pos

    def __str__(self):
        return self.description


def unicode_csv_reader(unicode_csv_data, **kwargs):
    # Fix field larger than field limit error
    maxInt = sys.maxsize
    while True:
        # decrease the maxInt value by factor 10
        # as long as the OverflowError occurs.
        try:
            csv.field_size_limit(maxInt)
            break
        except OverflowError:
            maxInt = int(maxInt / 10)
    csv.field_size_limit(maxInt)

    for line in csv.reader(unicode_csv_data, **kwargs):
        yield line


def _create_data_from_csv(data_path):
    with io.open(data_path, encoding="utf8") as f:
        next(f)
        reader = unicode_csv_reader(f)
        for row in reader:
            yield row[1], row[4], row[6]


file_name = '/home/dongxx/projects/def-mercer/dongxx/mimiciii/mimic3/final.csv'
iterator = _create_data_from_csv(file_name)
DATASET_NAME = 'minic'
NUM_LINES = 52722
dataset = _RawTextIterableDataset(DATASET_NAME, NUM_LINES, iterator)
tokenizer = get_tokenizer('basic_english')


def yield_tokens(data_iter):
    for _, _, text in data_iter:
        yield tokenizer(text)


vocab = build_vocab_from_iterator(yield_tokens(dataset))
iterator = _create_data_from_csv(file_name)
dataset = _RawTextIterableDataset(DATASET_NAME, NUM_LINES, iterator)


def pad_sequence(sequences, ksz, batch_first=False, padding_value=0.0):
    max_size = sequences[0].size()
    trailing_dims = max_size[1:]
    max_len = max([s.size(0) for s in sequences])
    if max_len < ksz:
        max_len = ksz
    if batch_first:
        out_dims = (len(sequences), max_len) + trailing_dims
    else:
        out_dims = (max_len, len(sequences)) + trailing_dims

    out_tensor = sequences[0].new_full(out_dims, padding_value)
    for i, tensor in enumerate(sequences):
        length = tensor.size(0)
        # use index notation to prevent duplicate references to the tensor
        if batch_first:
            out_tensor[i, :length, ...] = tensor
        else:
            out_tensor[:length, i, ...] = tensor

    return out_tensor


train_dataset = to_map_style_dataset(dataset)


def convert_text_tokens(text):
    text = tokenizer(text)
    filtered_text = [word for word in text if word not in stop_words]

    tokens = torch.tensor([vocab[token] for token in filtered_text])
    return tokens


def process_label(label):
    label = label.replace("'", "").replace('\n', '')
    label = label.lstrip("[").rstrip("]")
    label = list(label.split(" "))
    return label


def collate_batch(batch):
    text_list = []
    labels = []
    for (ICD9_CODE, D_ICD9_CODE, text) in batch:
        ICD9_CODE = process_label(ICD9_CODE)
        D_ICD9_CODE = process_label(D_ICD9_CODE)
        label = ICD9_CODE + D_ICD9_CODE
        if '' in label:
            label.remove('')

        labels.append(label)
        processed_text = convert_text_tokens(text)
        text_list.append(processed_text)
    text_length = [len(seq) for seq in text_list]
    pad_text = pad_sequence(text_list, ksz=10, batch_first=True)

    return pad_text, text_length, labels


dataloader = DataLoader(train_dataset, batch_size=4, collate_fn=collate_batch)
for i, (text, text_length, labels) in enumerate(dataloader):
    if i == 0:
        print(text.size())
        print(text_length)
        print(labels)
