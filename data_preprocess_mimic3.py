import csv
import pandas as pd
import numpy as np
import os
import sys
# note_events
import argparse
from mimic3benchmark.util import dataframe_from_csv

parser = argparse.ArgumentParser(description='Extract per-subject data from MIMIC-III CSV files.')
parser.add_argument('--mimic3_path', type=str,default="/home/dongxx/projects/def-mercer/dongxx/mimiciii/1.4")
parser.add_argument('--output_path', type=str,default="/home/dongxx/projects/def-mercer/dongxx/mimiciii/1.4")
args = parser.parse_args()
def index_default(i, char):

    try:
        retval = i.index(char)
    except ValueError:
        retval = 100000
    return retval
#
# def read_icd_diagnoses_table(mimic3_path):
#     codes = dataframe_from_csv(os.path.join(mimic3_path, 'D_ICD_DIAGNOSES.csv'))
#     codes = codes[['ICD9_CODE', 'LONG_TITLE']]
#     diagnoses = dataframe_from_csv(os.path.join(mimic3_path, 'DIAGNOSES_ICD.csv'))
#     diagnoses = diagnoses[["SUBJECT_ID", "HADM_ID", "ICD9_CODE"]]
#     print("code", len(codes))
#     print("diagnoses", len(diagnoses))
#     diagnoses = diagnoses.merge(codes, how='inner', left_on='ICD9_CODE', right_on='ICD9_CODE')
#
#     diagnoses[['SUBJECT_ID', 'HADM_ID']] = diagnoses[['SUBJECT_ID', 'HADM_ID']].astype(int)
#     return diagnoses
# def read_icd_procedures_table(mimic3_path):
#     codes = dataframe_from_csv(os.path.join(mimic3_path, 'D_ICD_PROCEDURES.csv'))
#     codes = codes[['ICD9_CODE', 'LONG_TITLE']]
#     procedures = dataframe_from_csv(os.path.join(mimic3_path, 'PROCEDURES_ICD.csv'))
#
#     procedures = procedures[["SUBJECT_ID","HADM_ID","ICD9_CODE"]]
#     print("code", len(codes))
#     print("procedures", len(procedures))
#
#     procedures = procedures.merge(codes, how='inner', left_on='ICD9_CODE', right_on='ICD9_CODE')
#
#     procedures[['SUBJECT_ID', 'HADM_ID']] = procedures[['SUBJECT_ID', 'HADM_ID']].astype(int)
#     return procedures
#
# def combine_procedures_diagnoses(procedures,diagnoses):
#     procedures_and_diagnoses = procedures.merge(diagnoses, how='inner', left_on=['SUBJECT_ID', 'HADM_ID'], right_on=['SUBJECT_ID', 'HADM_ID'])
#     return procedures_and_diagnoses
#
# note_events = dataframe_from_csv(os.path.join(args.mimic3_path, 'NOTEEVENTS.csv'))
# note_event_text = note_events[['SUBJECT_ID', 'HADM_ID', 'TEXT']].to_csv(os.path.join(args.output_path, 'all_note_event.csv'), index=False)#SUBJECT_ID,HADM_ID,TEXT
note_event_text = dataframe_from_csv(os.path.join(args.mimic3_path, 'all_note_event.csv'))
# diagnoses = read_icd_diagnoses_table(args.mimic3_path)
# procedures = read_icd_procedures_table(args.mimic3_path)
# print("diagnoses",len(diagnoses))
# print("procedures",len(procedures))
# diagnoses.to_csv(os.path.join(args.output_path, 'all_diagnoses.csv'), index=False)#SUBJECT_ID,HADM_ID,ICD9_CODE,LONG_TITLE
# procedures.to_csv(os.path.join(args.output_path, 'all_procedures.csv'), index=False)#SUBJECT_ID,HADM_ID,ICD9_CODE,LONG_TITLE
# procedures_and_diagnoses = combine_procedures_diagnoses(procedures,diagnoses)
# procedures_and_diagnoses.to_csv(os.path.join(args.output_path, 'all_diagnoses_with_procedures.csv'), index=False)
# print("procedures_and_diagnoses",len(procedures_and_diagnoses))


texts = []
def split_log_line(i):
    """Splits a line at either a period or a colon, depending on which appears
    first in the line.
    """
    if index_default(i, "chief complaint") < index_default(i, "history of present illness"):
        a = i.split('chief complaint')[1]
        b = 'Chief complaint' + a
        texts.append(b)
    elif index_default(i, "history of present illness") < 100000:
        a = i.split('history of present illness')[1]
        b = 'History of present illness' + a
        texts.append(b)
    else:
        texts.append('')


note_event_text['TEXT'] = note_event_text['TEXT'].str.lower()
for text in note_event_text['TEXT']:
    split_log_line(text)

note_event_text['require_text'] = texts
note_event_require_text = note_event_text['require_text'].replace(r'^\s*$', np.nan, regex=True, inplace = True)
# print(note_event_text.isna().sum())
note_event_require_text = note_event_require_text.dropna(subset=['HADM_ID','require_text'])
print(note_event_require_text[:1])
note_event_require_text = note_event_require_text[['SUBJECT_ID', 'HADM_ID','require_text']]#

note_event_require_text = note_event_require_text[['SUBJECT_ID', 'HADM_ID', 'require_text']].to_csv(os.path.join(args.output_path, 'all_require_text_event.csv'), index=False)#SUBJECT_ID,HADM_ID,TEXT



