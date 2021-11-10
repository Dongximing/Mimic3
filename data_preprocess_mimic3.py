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
            yield  row[1],row[4],row[6]
file_name='/home/dongxx/projects/def-mercer/dongxx/mimiciii/mimic3/final.csv'
iterator = _create_data_from_csv(file_name)
DATASET_NAME = 'minic'
NUM_LINES = 52722
dataset = _RawTextIterableDataset(DATASET_NAME, NUM_LINES, iterator)
tokenizer = get_tokenizer('basic_english')
def yield_tokens(data_iter):
    for _,_, text in data_iter:
        yield tokenizer(text)
vocab = build_vocab_from_iterator(yield_tokens(dataset))
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
    label =label.replace("'", "").replace('\n', '')
    label= label.lstrip("[").rstrip("]")
    label = list(label.split(" "))
    return label
def collate_batch(batch):
    text_list = []
    labels =[]
    for (ICD9_CODE,D_ICD9_CODE, text) in batch:
        ICD9_CODE = process_label(ICD9_CODE)
        D_ICD9_CODE = process_label(D_ICD9_CODE)
        label = ICD9_CODE+D_ICD9_CODE
        if '' in label:
            label.remove('')
            
        labels.append(label)
        processed_text = convert_text_tokens(text)
        text_list.append(processed_text)
    text_length = [len(seq) for seq in text_list]
    pad_text = pad_sequence(text_list, ksz=10, batch_first=True)
    
    return pad_text,text_length,labels 
dataloader = DataLoader(train_dataset, batch_size = 4,collate_fn=collate_batch)
for i,(text,text_length,labels) in enumerate(dataloader):
    if i == 0:
        print(text.size()) 
        print(text_length)
        print(labels)



        
# #
# # import pandas as pd
# # import numpy as np
# # import os
# # import sys
# # # note_events
# # import argparse
# # from mimic3benchmark.util import dataframe_from_csv
# # def load_list_from_txt(filepath):
# #     with open(filepath, 'r') as f:
# #         return f.read().split()
# # df = pd.read_csv('/home/dongxx/projects/def-mercer/dongxx/mimiciii/final.csv')
# # DATA_DIR = '/home/dongxx/projects/def-mercer/dongxx/caml-mimic/mimicdata/mimic3/'
# # val_ids = load_list_from_txt(f'{DATA_DIR}dev_full_hadm_ids.csv')
# # test_ids = load_list_from_txt(f'{DATA_DIR}test_full_hadm_ids.csv')
# # train_ids = load_list_from_txt(f'{DATA_DIR}train_full_hadm_ids.csv')
# # hadm_ids = [train_ids, val_ids, test_ids]
# # print(hadm_ids)
# # full_train,full_val,full_test = [df[df['HADM_ID'].isin(ids)] for ids in hadm_ids]
# # print(len(full_train))
# # print(len(full_val))
# # print(len(full_test))

# # print(train_ids)
# # parser = argparse.ArgumentParser(description='Extract per-subject data from MIMIC-III CSV files.')
# # parser.add_argument('--mimic3_path', type=str,default="/home/dongxx/projects/def-mercer/dongxx/mimiciii/1.4")
# # parser.add_argument('--output_path', type=str,default="/home/dongxx/projects/def-mercer/dongxx/mimiciii/1.4")
# # args = parser.parse_args()
# # def index_default(i, char):
# #
# #     try:
# #         retval = i.index(char)
# #     except ValueError:
# #         retval = 100000
# #     return retval
# # #
# # # def read_icd_diagnoses_table(mimic3_path):
# # #     codes = dataframe_from_csv(os.path.join(mimic3_path, 'D_ICD_DIAGNOSES.csv'))
# # #     codes = codes[['ICD9_CODE', 'LONG_TITLE']]
# # #     diagnoses = dataframe_from_csv(os.path.join(mimic3_path, 'DIAGNOSES_ICD.csv'))
# # #     diagnoses = diagnoses[["SUBJECT_ID", "HADM_ID", "ICD9_CODE"]]
# # #     print("code", len(codes))
# # #     print("diagnoses", len(diagnoses))
# # #     diagnoses = diagnoses.merge(codes, how='inner', left_on='ICD9_CODE', right_on='ICD9_CODE')
# # #
# # #     diagnoses[['SUBJECT_ID', 'HADM_ID']] = diagnoses[['SUBJECT_ID', 'HADM_ID']].astype(int)
# # #     return diagnoses
# # # def read_icd_procedures_table(mimic3_path):
# # #     codes = dataframe_from_csv(os.path.join(mimic3_path, 'D_ICD_PROCEDURES.csv'))
# # #     codes = codes[['ICD9_CODE', 'LONG_TITLE']]
# # #     procedures = dataframe_from_csv(os.path.join(mimic3_path, 'PROCEDURES_ICD.csv'))
# # #
# # #     procedures = procedures[["SUBJECT_ID","HADM_ID","ICD9_CODE"]]
# # #     print("code", len(codes))
# # #     print("procedures", len(procedures))
# # #
# # #     procedures = procedures.merge(codes, how='inner', left_on='ICD9_CODE', right_on='ICD9_CODE')
# # #
# # #     procedures[['SUBJECT_ID', 'HADM_ID']] = procedures[['SUBJECT_ID', 'HADM_ID']].astype(int)
# # #     return procedures
# # #
# # # def combine_procedures_diagnoses(procedures,diagnoses):
# # #     procedures_and_diagnoses = procedures.merge(diagnoses, how='inner', left_on=['SUBJECT_ID', 'HADM_ID'], right_on=['SUBJECT_ID', 'HADM_ID'])
# # #     return procedures_and_diagnoses
# # #
# # # note_events = dataframe_from_csv(os.path.join(args.mimic3_path, 'NOTEEVENTS.csv'))
# # # note_event_text = note_events[['SUBJECT_ID', 'HADM_ID', 'TEXT']].to_csv(os.path.join(args.output_path, 'all_note_event.csv'), index=False)#SUBJECT_ID,HADM_ID,TEXT
# # note_event_text = dataframe_from_csv(os.path.join(args.mimic3_path, 'all_note_event.csv'))
# # # diagnoses = read_icd_diagnoses_table(args.mimic3_path)
# # # procedures = read_icd_procedures_table(args.mimic3_path)
# # # print("diagnoses",len(diagnoses))
# # # print("procedures",len(procedures))
# # # diagnoses.to_csv(os.path.join(args.output_path, 'all_diagnoses.csv'), index=False)#SUBJECT_ID,HADM_ID,ICD9_CODE,LONG_TITLE
# # # procedures.to_csv(os.path.join(args.output_path, 'all_procedures.csv'), index=False)#SUBJECT_ID,HADM_ID,ICD9_CODE,LONG_TITLE
# # # procedures_and_diagnoses = combine_procedures_diagnoses(procedures,diagnoses)
# # # procedures_and_diagnoses.to_csv(os.path.join(args.output_path, 'all_diagnoses_with_procedures.csv'), index=False)
# # # print("procedures_and_diagnoses",len(procedures_and_diagnoses))
# #
# #
# # texts = []
# # def split_log_line(i):
# #     """Splits a line at either a period or a colon, depending on which appears
# #     first in the line.
# #     """
# #     if index_default(i, "chief complaint") < index_default(i, "history of present illness"):
# #         a = i.split('chief complaint')[1]
# #         b = 'Chief complaint' + a
# #         texts.append(b)
# #     elif index_default(i, "history of present illness") < 100000:
# #         a = i.split('history of present illness')[1]
# #         b = 'History of present illness' + a
# #         texts.append(b)
# #     else:
# #         texts.append('')
# #
# #
# # note_event_text['TEXT'] = note_event_text['TEXT'].str.lower()
# # for text in note_event_text['TEXT']:
# #     split_log_line(text)
# #
# # note_event_text['Require_text'] = texts
# # note_event_text['Require_text'] = note_event_text['Require_text'] .replace(r'^\s*$', np.NaN, regex=True)
# #
# # note_event_require_text =note_event_text.dropna(subset=['HADM_ID','Require_text']).reset_index()
# #
# #
# # note_event_require_text = note_event_require_text[['SUBJECT_ID', 'HADM_ID','Require_text']]#
# # print("len",len(note_event_require_text))
# # note_event_require_text = note_event_require_text[['SUBJECT_ID', 'HADM_ID', 'Require_text']].to_csv(os.path.join(args.output_path, 'all_require_text_event.csv'), index=False)#SUBJECT_ID,HADM_ID,TEXT
# import pandas as pd
# import numpy as np
# import os
# from tqdm import tqdm
# final = pd.read_csv(os.path.join('/home/dongxx/projects/def-mercer/dongxx/mimiciii/final.csv'))

# for i in range(len(final)):

#     if isinstance(final['D_ICD9_CODE'][i],float):
#         continue
#     final['D_ICD9_CODE'][i] =final['D_ICD9_CODE'][i].replace("'", "").replace('\n', '')
#     final['D_ICD9_CODE'][i]= final['D_ICD9_CODE'][i].lstrip("[").rstrip("]")
#     final['D_ICD9_CODE'][i] = list(final['D_ICD9_CODE'][i].split(" "))
# for i in range(len(final)):

#     # print(type(final['ICD9_CODE'][i]))
#     if isinstance(final['ICD9_CODE'][i],list):
#         continue
#     if isinstance(final['ICD9_CODE'][i],float):
#         continue
#     final['ICD9_CODE'][i] =final['ICD9_CODE'][i].replace("'", "").replace('\n', '')
#     final['ICD9_CODE'][i]= final['ICD9_CODE'][i].lstrip("[").rstrip("]")
#     final['ICD9_CODE'][i] = list(final['ICD9_CODE'][i].split(" "))
# print(final.ICD9_CODE.explode().nunique())
# print(final.D_ICD9_CODE.explode().nunique())
# val_texts = []
# p_val_labels = []
# d_val_labels = []
# val_text = final.TEXT
# p_val_label = final.ICD9_CODE
# d_val_label = final.D_ICD9_CODE
# for i,(text,p_label,d_label) in enumerate(zip(val_text,p_val_label,d_val_label)):
#     if i <= 1:
#         val_texts.append(text)
#         p_val_labels.append(p_label)
#         d_val_labels.append(d_label)
#     else:
#         break
# print(type(val_texts[0]))
# print(type(p_val_labels[1]))
# print(type(d_val_labels[1]))

